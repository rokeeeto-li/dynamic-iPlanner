import torch
import pypose as pp

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
    
class LinearDynamic(pp.module.NLS):
    def __init__(self, dt=None):
        super().__init__()
        self.dt = dt

    def origin_state_transition(self, state, input, t=None):
        return state + input*self.dt
    
    def error_state_transition(self, error_state, input, t=None):
        ref = self.ref_traj[...,t,:]
        next_ref = self.ref_traj[...,t+1,:]

        next_err = -next_ref + ref + error_state + input*self.dt
        return next_err

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
