import torch
import pypose as pp
# from iplanner.dynamics import Bicycle, LinearDynamic
from dynamics import Bicycle, LinearDynamic

torch.set_default_dtype(torch.float32)

class TrajPlanner:
    def __init__(self, is_train=False):
        self.T = 14
        self.dt = 0.1
        self.is_train = is_train
        if self.is_train:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.mpc_config()
        # self.linear_mpc_config()
    
    def getSE3(self, xyzrpy):
        xyz = xyzrpy[..., :3]
        rpy = xyzrpy[..., 3:]
        q = pp.euler2SO3(rpy).tensor()
        return pp.SE3(torch.cat([xyz, q], dim=-1))
        
    def mpc_config(self):
        self.dynamics = Bicycle(dt = self.dt)

        if self.is_train:
            n_batch = 128
        else:
            n_batch = 1
        n_state, n_ctrl = 6, 2
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1, 1))
        Q[..., n_state:, n_state:] *= 1e-2
        p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1))
 
        stepper = pp.utils.ReduceToBason(steps=1, verbose=False)
        self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)
        
    def linear_mpc_config(self):
        self.dynamics = LinearDynamic(dt = self.dt)
        
        if self.is_train:
            n_batch = 128
        else:
            n_batch = 1
        n_state, n_ctrl = 3, 3
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1, 1))
        Q[..., n_state:, n_state:] *= 1e-2
        p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1))
 
        stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
        self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)
                
    def oriAppend(self, waypoints):
        # m, n = waypoints.shape[0], waypoints.shape[1]
        # append_mtx = torch.tensor([0.0,0.0,0.0,1.0], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(m,n,1)
        # waypoints = torch.cat((waypoints, append_mtx), dim=2)
        # init_pts = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0,1.0]]], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(m,1,1)
        # waypoints_SE3 = pp.SE3(torch.cat((init_pts, waypoints), dim = 1))
        # print("The waypoints_SE3 shape is: ", waypoints_SE3.shape)
        # return waypoints_SE3
    
        batch_size, num_p, _ = waypoints.shape
        zero_0 = torch.zeros(batch_size, 1, 3, device=waypoints.device)
        padded_waypoints = torch.cat((zero_0, waypoints), dim=1)
        delta = padded_waypoints[:, 1:] - padded_waypoints[:, :-1]
        angles = torch.atan2(delta[:, :, 1], delta[:, :, 0]).unsqueeze(-1)
        zeros_1 = torch.zeros(batch_size, 1, 1, device=waypoints.device)
        angles = torch.cat((zeros_1, angles), dim=1)
        zeros_2 = torch.zeros((angles.shape[0], angles.shape[1], 2), device=waypoints.device, dtype=waypoints.dtype)
        orientations = torch.cat((zeros_2, angles), dim=-1)
        waypoints_ori = torch.cat((padded_waypoints, orientations), dim=2)
        waypoints_SE3 = self.getSE3(waypoints_ori)
        return waypoints_SE3
    
    def trajGenerate(self, waypoints):
        # Biycle model
        waypoints = self.oriAppend(waypoints)
        traj = pp.bspline(waypoints, interval=0.75, extrapolate=True)
        self.dynamics.set_reftrajectory(traj)
        
        x_init = traj[...,0,:].Log()
        
        # Linear Model
        # waypoints = self.oriAppend(waypoints)
        # traj = pp.bspline(waypoints, interval=0.75, extrapolate=True).translation()
        # self.dynamics.set_reftrajectory(traj)
                
        # x_init = traj[...,0,:]
        _, u_mpc, cost = self.MPC(self.dt, x_init)
        
        x_traj = x_init.unsqueeze(-2).repeat((1, self.T+1, 1))
        self.dynamics.recover_dynamics()
        
        for i in range(self.T):
            x_traj[...,i+1,:], _ = self.dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])
                        
        # return x_traj, cost
        return x_traj.translation(), cost