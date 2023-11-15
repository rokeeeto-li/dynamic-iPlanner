import torch
import pypose as pp

torch.set_default_dtype(torch.float32)
        
class Bicycle(pp.module.NLS):
    def __init__(self, dt=None):
        super().__init__()
        self.ref_traj = None
        self.dt = dt

    def origin_state_transition(self, state, input, t=None):
        v, w = input[..., 0:1], input[..., 1:2]
        zeros = torch.zeros_like(v.repeat(1,4), dtype=torch.float32, requires_grad=True)
        xyzrpy = torch.cat((v, zeros, w), dim=-1)*self.dt
        rt = self.getSE3(xyzrpy)
        return (state.Exp()*rt).Log()

    def error_state_transition(self, error_state, input, t=None):
        v, w = input[..., 0:1], input[..., 1:2]
        zeros = torch.zeros_like(v.repeat(1,4), dtype=torch.float32, requires_grad=True)
        xyzrpy = torch.cat((v, zeros, w), dim=-1)*self.dt
        rt = self.getSE3(xyzrpy)

        ref_SE3 = self.ref_traj[...,t,:]
        next_ref_SE3 = self.ref_traj[...,t+1,:]

        return (next_ref_SE3.Inv()*ref_SE3*error_state.Exp()*rt).Log()

    def state_transition(self, state, input, t):
        if self.ref_traj is None:
            return self.origin_state_transition(state, input, t)
        else:
            return self.error_state_transition(state, input, t)

    def observation(self, state, input, t=None):
        return state

    def set_reftrajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def recover_dynamics(self):
        self.ref_traj = None
        
    def getSE3(self, xyzrpy):
        xyz = xyzrpy[..., :3]
        rpy = xyzrpy[..., 3:]
        q = pp.euler2SO3(rpy).tensor()
        return pp.SE3(torch.cat([xyz, q], dim=-1))
    

class TrajPlanner:
    def __init__(self, is_train=False):
        self.T = 14
        self.dt = 0.1
        self.is_train = is_train
        self.dynamics = Bicycle(dt = 1)
        if self.is_train:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.mpc_config()
        
    def mpc_config(self):
        if self.is_train:
            n_batch = 128
        else:
            n_batch = 1
        n_state, n_ctrl = 6, 2
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1, 1))
        Q[..., n_state:, n_state:] *= 1e-1
        p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1))

        stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
        self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)
                
    def oriAppend(self, waypoints):
        m, n = waypoints.shape[0], waypoints.shape[1]
        # xy, theta = waypoints[..., :2], waypoints[..., 2:3]
        # append_mtx = torch.zeros(m,n,3, requires_grad=True, device=waypoints.device, dtype=waypoints.dtype)
        append_mtx = torch.tensor([0.0,0.0,0.0,1.0], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(m,n,1)
        waypoints = torch.cat((waypoints, append_mtx), dim=2)
        init_pts = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0,1.0]]], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(m,1,1)
        return pp.SE3(torch.cat((init_pts, waypoints), dim = 1))
    
    def trajGenerate(self, waypoints):
        waypoints = self.oriAppend(waypoints)
        traj = pp.bspline(waypoints, interval=0.75, extrapolate=True)
        self.dynamics.set_reftrajectory(traj)
        
        x_init = traj[...,0,:].Log()
        _, u_mpc, cost = self.MPC(self.dt, x_init)
        
        x_traj = x_init.unsqueeze(-2).repeat((1, self.T+1, 1))
        self.dynamics.recover_dynamics()

        for i in range(self.T):
            x_traj[...,i+1,:], _ = self.dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])
        return x_traj.Exp()[..., :3], cost