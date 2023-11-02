import torch
import pypose as pp

torch.set_default_dtype(torch.float32)

class anySE3(pp.module.NLS):
    def __init__(self, dt=None):
        super().__init__()
        self.ref_traj = None
        self.dt = dt

    def state_transition(self, state, input, t):
        if self.ref_traj is None:
            return state.Retr(pp.se3(input))
        else:
            ref_SE3 = self.ref_traj[...,t,:]
            next_ref_SE3 = self.ref_traj[...,t+1,:]
            next_SE3 = (ref_SE3@state).Retr(pp.se3(input))
            return next_ref_SE3.Inv()*next_SE3

    def observation(self, state, input, t=None):
        return state

    def set_reftrajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def recover_dynamics(self):
        self.ref_traj = None
        
class TrajPlanner:
    def __init__(self):
        self.T = 14
        self.dt = 0.1
        self.dynamics = anySE3()
        self.device = torch.device("cpu")  
        self.mpc_config()
        
    def mpc_config(self):
        n_batch = 1
        n_state, n_ctrl = 7, 6
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1, 1))
        Q[..., n_state:, n_state:] *= 1e-1
        p = torch.tile(torch.zeros(n_state + n_ctrl, device=self.device), (n_batch, self.T, 1))

        stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
        self.MPC = pp.module.MPC(self.dynamics, Q, p, self.T, stepper=stepper)
        
    def oriAppend(self, waypoints):
        m = waypoints.shape[1]
        append_mtx = torch.tensor([0.0,0.0,0.0,1.0], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype).repeat(1,m,1)
        waypoints = torch.cat((waypoints, append_mtx), dim=2)
        init_pts = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0,1.0]]], requires_grad=True, device=waypoints.device, dtype=waypoints.dtype)
        return pp.SE3(torch.cat((init_pts, waypoints), dim = 1))
    
    def trajGenerate(self, waypoints):
        waypoints = self.oriAppend(waypoints)
        traj = pp.bspline(waypoints, interval=0.75, extrapolate=True)
        print("the traj shape is: ", traj.shape)
        self.dynamics.set_reftrajectory(traj)
        
        x_init = traj[...,0,:]
        _, u_mpc, cost = self.MPC(self.dt, x_init)
        
        x_traj = x_init.unsqueeze(-2).repeat((1, self.T+1, 1))
        self.dynamics.recover_dynamics()

        for i in range(self.T):
            x_traj[...,i+1,:], _ = self.dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])
        return x_traj[..., :3], cost