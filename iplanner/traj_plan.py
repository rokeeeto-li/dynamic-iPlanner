import torch
import pypose as pp
# from iplanner.dynamics import Bicycle, LinearDynamic
# from dynamics import Bicycle, LinearDynamic, NonLinearDynamic, SecondOrderLinear

torch.set_default_dtype(torch.float32)

class TrajPlanner2():
    def __init__(self, is_train=False, gpu_id=0, v_ref=1.5, dynamic=None):
        self.v_ref = v_ref
        self.dynamic = dynamic
        self.is_train = is_train
        if self.is_train:
            self.device = torch.device("cuda:" + str(gpu_id))
        else:
            self.device = torch.device("cpu")

    def set_dynamic(self, dynamic):
        self.dynamic = dynamic

    def ref_generate(self, waypoints, init_vel=None, spline_step = 0.1):
        waypoints = waypoints.to(self.device)

        waypoints = torch.cat((torch.zeros(waypoints.shape[0], 1, 3, device=self.device), waypoints), dim=1)
        splined = pp.chspline(waypoints, spline_step)
        # def calculate_closest_distance(waypoints, splined):
        #     batch_size, num_points, _ = waypoints.shape
        #     _, num_splined, _ = splined.shape

        #     waypoints = waypoints.unsqueeze(2).expand(batch_size, num_points, num_splined, 3)
        #     splined = splined.unsqueeze(1).expand(batch_size, num_points, num_splined, 3)

        #     distances = torch.norm(waypoints - splined, dim=-1)
        #     closest_distances, _ = torch.min(distances, dim=-1)

        #     return closest_distances
        
        # closest_distances = calculate_closest_distance(waypoints, splined)

        direction_vectors = splined[...,1:,:] - splined[...,:-1,:]
        magnitudes = torch.sqrt(torch.sum(direction_vectors**2, dim=-1, keepdim=True))
        desired_v = direction_vectors / magnitudes * self.v_ref

        if init_vel is None:
            init_v = torch.zeros(desired_v.shape[0], 1, 3, device = self.device)
        else:
            init_v = init_vel.unsqueeze(1).to(self.device)

        end_v = torch.zeros(desired_v.shape[0], 1, 3, device = self.device)
        desired_v = torch.cat((init_v, desired_v[...,1:,:], end_v), dim=-2)

        self.ref_traj = torch.cat((splined, desired_v), dim=-1)

        self.batch, T, _ = self.ref_traj.shape
        self.T = T-1
        self.mpc_configs(self.batch, self.T)

        avg_v = (desired_v[:,1:,:] + desired_v[:,:-1,:])/2
        t = magnitudes / avg_v
        total_time = torch.sum(t, dim=-2)
        
        total_length = torch.sum(magnitudes, dim=-2)
        total_time = (magnitudes[:,0,:]+magnitudes[:,-1,:]) / self.v_ref + total_length / self.v_ref
        # self.dynamic.dt = self.dt = total_length / self.v_ref / self.T
        # print("dt shape:", self.dt)
        # print("dt2 shape:", (total_time/self.T))
        self.dynamic.dt = self.dt = total_time / self.T
    
    def mpc_configs(self, n_batch, T):
        n_state, n_ctrl = self.dynamic.state_dim, self.dynamic.ctrl_dim
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (n_batch, T, 1, 1))
        Q[..., 0, 0] *= 0.3
        Q[..., 3:, 3:] *= 0.1
        # Q[..., n_state:, n_state:] *= 0.05
        # Q[..., -1, :, :] *= 10
        p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (n_batch, T, 1))

        stepper = pp.utils.ReduceToBason(steps=1, verbose=False)
        self.MPC = pp.module.MPC(self.dynamic, Q, p, T, stepper=stepper)
        self.MPC.Q = Q
        return
    
    def planning(self, waypoints, vel = None):
        self.ref_generate(waypoints, init_vel = vel)
        self.dynamic.set_reftrajectory(self.ref_traj)
        x_init = self.ref_traj[...,0,:]
        x_error, motion, cost = self.MPC(self.dt, x_init)
        x_traj = x_error + self.ref_traj
        return x_traj, motion 


# class TrajPlanner:
#     def __init__(self, is_train=False, gpu_id=0, n_batch=1):
#         self.T = 14
#         self.dt = 0.1
#         self.is_train = is_train
#         self.n_batch = n_batch
#         if self.is_train:
#             self.device = torch.device("cuda:" + str(gpu_id))
#         else:
#             self.device = torch.device("cpu")
#         # self.mpc_config()
#         self.linear_mpc_config()
#         # self.quadratic_config()
#         # self.SecondOrderLinear_config()
    
#     def getSE3(self, xyzrpy):
#         xyz = xyzrpy[..., :3]
#         rpy = xyzrpy[..., 3:]
#         q = pp.euler2SO3(rpy).tensor()
#         return pp.SE3(torch.cat([xyz, q], dim=-1))
        
#     def mpc_config(self):
#         self.dynamics = Bicycle(dt = self.dt)

#         n_state, n_ctrl = 6, 2
#         Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (self.n_batch, self.T, 1, 1))
#         Q[..., 3:, 3:] *= 1e-2
#         p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (self.n_batch, self.T, 1))
 
#         stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
#         self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)
        
#     def linear_mpc_config(self):
#         self.dynamics = LinearDynamic(dt = self.dt)
        
#         n_state, n_ctrl = 3, 3
#         Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (self.n_batch, self.T, 1, 1))
#         Q[..., n_state:, n_state:] *= 1e-2
#         p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (self.n_batch, self.T, 1))
 
#         stepper = pp.utils.ReduceToBason(steps=1, verbose=False)
#         self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)

#     def quadratic_config(self):
#         self.dynamics = NonLinearDynamic(dt = self.dt)
        
#         n_state, n_ctrl = 3, 3
#         Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (self.n_batch, self.T, 1, 1))
#         Q[..., n_state:, n_state:] *= 1e-2
#         p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (self.n_batch, self.T, 1))
 
#         stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
#         self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)

#     def oriAppend(self, waypoints):
#         m, n = waypoints.shape[0], waypoints.shape[1]
#         append_mtx = torch.tensor([0.0,0.0,0.0,1.0], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(m,n,1)
#         waypoints = torch.cat((waypoints, append_mtx), dim=2)
#         init_pts = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0,1.0]]], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(m,1,1)
#         waypoints_SE3 = pp.SE3(torch.cat((init_pts, waypoints), dim = 1))
#         return waypoints_SE3
    
#         # batch_size, num_p, _ = waypoints.shape
#         # zero_0 = torch.zeros(batch_size, 1, 3, device=waypoints.device)
#         # padded_waypoints = torch.cat((zero_0, waypoints), dim=1)
#         # delta = padded_waypoints[:, 1:] - padded_waypoints[:, :-1]
#         # angles = torch.atan2(delta[:, :, 1], delta[:, :, 0]).unsqueeze(-1)
#         # zeros_1 = torch.zeros(batch_size, 1, 1, device=waypoints.device)
#         # angles = torch.cat((zeros_1, angles), dim=1)
#         # zeros_2 = torch.zeros((angles.shape[0], angles.shape[1], 2), device=waypoints.device, dtype=waypoints.dtype)
#         # orientations = torch.cat((zeros_2, angles), dim=-1)
#         # waypoints_ori = torch.cat((padded_waypoints, orientations), dim=2)
#         # waypoints_SE3 = self.getSE3(waypoints_ori)
#         # return waypoints_SE3
    
#     def trajGenerate(self, waypoints):
#         # Biycle model
#         # waypoints = self.oriAppend(waypoints)
#         # traj = pp.bspline(waypoints, interval=0.75, extrapolate=True)
#         # self.dynamics.set_reftrajectory(traj)
        
#         # x_init = traj[...,0,:].Log()
        
#         # Linear Model
#         waypoints = self.oriAppend(waypoints)
#         traj = pp.bspline(waypoints, interval=0.75, extrapolate=True).translation()
#         self.dynamics.set_reftrajectory(traj)
                
#         x_init = traj[...,0,:]
#         _, u_mpc, cost = self.MPC(self.dt, x_init)
        
#         x_traj = x_init.unsqueeze(-2).repeat((1, self.T+1, 1))
#         self.dynamics.recover_dynamics()
        
#         for i in range(self.T):
#             x_traj[...,i+1,:], _ = self.dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])

#         # print("x_traj:", x_traj)                
#         return x_traj, cost
#         # return x_traj.translation(), cost