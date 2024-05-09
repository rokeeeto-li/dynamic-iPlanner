import torch
import pypose as pp
import numpy as np
import matplotlib.pyplot as plt

class second_order_linear(pp.module.NLS):
    def __init__(self, dt=None, v_ref=5.0):
        super().__init__()
        self.dt = dt
        self.v_ref = v_ref
        self.T = None
        self.state_dim = 6
        self.ctrl_dim = 3
    
    def state_transition(self, state, input, t):
        state_mask = torch.cat((torch.cat((torch.zeros(3,3), torch.eye(3)), dim=-1), torch.zeros(3,6)), dim=-2).repeat(state.shape[0], 1, 1)
        input_mask = torch.cat((torch.zeros(3,3), torch.eye(3)), dim=-2).repeat(state.shape[0], 1, 1)

        if self.ref_traj is None:
            state = state + pp.bmv(state_mask, state)*self.dt + pp.bmv(input_mask, input)*self.dt
        else:
            ref = self.ref_traj[...,t,:]
            next_ref = self.ref_traj[...,t+1,:]
            state = state + ref
            state = state + pp.bmv(state_mask, state)*self.dt + pp.bmv(input_mask, input)*self.dt - next_ref
        return state
        
    def observation(self, state, input, t=None):
        return state

    def set_reftrajectory(self, ref_traj):
        self.ref_traj = ref_traj
        self.T = ref_traj.shape[1]-1

    def recover_dynamics(self):
        self.ref_traj = None

class traj_planner():
    def __init__(self, v_ref=5.0, dynamic=None):
        self.v_ref = v_ref
        self.dynamic = dynamic

    def set_dynamic(self, dynamic):
        self.dynamic = dynamic

    def ref_generate(self, waypoints, spline_step = 0.3):
        splined = pp.chspline(waypoints, spline_step)

        direction_vectors = splined[...,1:,:] - splined[...,:-1,:]
        magnitudes = torch.sqrt(torch.sum(direction_vectors**2, dim=-1, keepdim=True))
        desired_v = direction_vectors / magnitudes * self.v_ref

        init_v = torch.zeros(desired_v.shape[0], 1, 3)
        end_v = torch.zeros(desired_v.shape[0], 1, 3)
        desired_v = torch.cat((init_v, desired_v[...,1:,:], end_v), dim=-2)

        self.ref_traj = torch.cat((splined, desired_v), dim=-1)

        self.batch, T, _ = self.ref_traj.shape
        self.T = T-1
        self.mpc_configs(self.batch, self.T)

        total_length = torch.sum(magnitudes, dim=-2)
        self.dynamic.dt = self.dt = total_length / self.v_ref / self.T
    
    def mpc_configs(self, n_batch, T):
        n_state, n_ctrl = self.dynamic.state_dim, self.dynamic.ctrl_dim
        Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
        Q[..., n_state:, n_state:] *= 0.1
        Q[..., -1, :, :] *= 10
        p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))

        stepper = pp.utils.ReduceToBason(steps=1, verbose=False)
        self.MPC = pp.module.MPC(self.dynamic, Q, p, T, stepper=stepper)
        return
    
    def planning(self, waypoints):
        self.ref_generate(waypoints)
        self.dynamic.set_reftrajectory(self.ref_traj)
        x_init = self.ref_traj[...,0,:]
        x_error, _, _ = self.MPC(self.dt, x_init)
        x_traj = x_error + self.ref_traj
        return self.ref_traj, x_traj


def waypoints_configs():
    r, sqrt2 = 6, 2**0.5
    waypoints = torch.tensor([[[0, 0, 0], [r-r/sqrt2, r/sqrt2, 0], [r, r, 0], 
                            [r+r/sqrt2, r/sqrt2, 0], [2*r, 0, 0], [r+r/sqrt2, -r/sqrt2, 0],
                            [r, -r, 0], [r-r/sqrt2, -r/sqrt2, 0], [0, 0, 0],
                            [r/sqrt2-r, r/sqrt2, 0], [-r, r, 0], [-r/sqrt2-r, r/sqrt2, 0],
                            [-2*r, 0, 0], [-r/sqrt2-r, -r/sqrt2, 0], [-r, -r, 0],
                            [r/sqrt2-r, -r/sqrt2, 0], [0, 0, 0]]], requires_grad=True)
    return waypoints

def visulization(waypoints, ref_traj, planned_traj):
    waypoints = waypoints.squeeze().detach().numpy()
    ref_traj = ref_traj.squeeze().detach().numpy()
    planned_traj = planned_traj.squeeze().detach().numpy()

    ref_pos, ref_v = ref_traj[...,0:3], ref_traj[...,3:6]
    planned_pos, planned_v = planned_traj[...,0:3], planned_traj[...,3:6]
    ref_v_mag = np.linalg.norm(ref_v, axis=1)
    planned_v_mag = np.linalg.norm(planned_v, axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    # Plot Trajectories
    axs[0].set_aspect('equal')
    axs[0].scatter(waypoints[...,0], waypoints[...,1], c='r', marker='o', label='Waypoints', alpha=0.8)
    axs[0].plot(ref_pos[...,0], ref_pos[...,1], 'o-', label='Reference Trajectory', alpha=0.1)
    axs[0].plot(planned_pos[...,0], planned_pos[...,1], 'o-', label='Planned Trajectory', alpha=0.8)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Trajectories')
    axs[0].legend()
    # Plot Velocity Magnitudes
    axs[1].plot(ref_v_mag, label='Reference Velocity', alpha=0.6)
    axs[1].plot(planned_v_mag, label='Planned Velocity', alpha=0.8)
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Velocity Magnitude')
    axs[1].set_title('Velocity Magnitudes')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    waypoints = waypoints_configs()
    dynamic = second_order_linear()
    planner = traj_planner(dynamic=dynamic)
    ref_traj, x_traj = planner.planning(waypoints)
    visulization(waypoints, ref_traj, x_traj)


if __name__ == "__main__":
    main()