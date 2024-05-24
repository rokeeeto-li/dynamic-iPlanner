# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
import pypose as pp
from tsdf_map import TSDF_Map
from traj_opt import TrajOpt
# from traj_plan import TrajPlanner
from traj_plan import TrajPlanner2
from dynamics import SecondOrderLinear
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

# class TrajCost:
#     def __init__(self, gpu_id=0, n_batch=1):
#         self.tsdf_map = TSDF_Map(gpu_id)
#         self.planner = TrajPlanner(is_train=True, gpu_id=gpu_id, n_batch=n_batch)
#         self.opt = TrajOpt()
#         self.is_map = False
#         return None

#     def TransformPoints(self, odom, points):
#         batch_size, num_p, _ = points.shape
#         world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
#         world_ps.tensor()[:, :, 0:3] = points
#         world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
#         return world_ps
    
#     def SetMap(self, root_path, map_name):
#         self.tsdf_map.ReadTSDFMap(root_path, map_name)
#         self.is_map = True
#         return

#     def CostofTraj(self, waypoints, odom, goal, ahead_dist, alpha=2, beta=1.0, gamma=2.0, delta=5.0, obstalce_thred=0.5):
#         batch_size, num_p, _ = waypoints.shape
#         if self.is_map:
#             world_ps = self.TransformPoints(odom, waypoints)
#             norm_inds, _ = self.tsdf_map.Pos2Ind(world_ps)
#             # Obstacle Cost
#             cost_grid = self.tsdf_map.cost_array.T.expand(batch_size, 1, -1, -1)
#             oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
#             oloss_M = oloss_M.to(torch.float32)
#             oloss = torch.mean(torch.sum(oloss_M, axis=1))

#             # Terrian Height loss
#             height_grid = self.tsdf_map.ground_array.T.expand(batch_size, 1, -1, -1)
#             hloss_M = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
#             hloss_M = torch.abs(waypoints[:, :, 2] - hloss_M)
#             hloss = torch.mean(torch.sum(hloss_M, axis=1))

#         # Goal Cost
#         gloss = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
#         gloss = torch.mean(torch.log(gloss + 1.0))
#         # gloss = torch.mean(gloss)
        
#         # Motion Loss
#         desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0/(num_p-1)) 
#         desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
#         wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0:num_p-1, :], dim=2)
#         mloss = torch.abs(desired_ds - wp_ds)
#         mloss = torch.sum(mloss, axis=1)
#         mloss = torch.mean(mloss)
                
#         # Fear labels
#         goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
#         floss_M = torch.clone(oloss_M)[:, 1:]
#         floss_M[goal_dists > ahead_dist] = 0.0
#         fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
#         fear_labels = (fear_labels > obstalce_thred).to(torch.float32)
        
#         total_loss = alpha*oloss + beta*hloss + delta*gloss
#         # total_loss = alpha*oloss + beta*hloss + gamma*mloss + delta*gloss
        
#         # print("\nthe obstacle cost is:", oloss)
#         # print("the height cost is:", hloss)
#         # # print("the motion cost is:", mloss)
#         # print("the goal cost is:", gloss)
#         # print("the fear labels is:", torch.sum(fear_labels))
        
#         return total_loss, fear_labels
    
class MulLayerCost:
    def __init__(self, gpu_id=0, n_batch=1, alpha=5.0, beta=1.0, gamma=1.0, delta=2.0, epi=1.0, zeta=1.0, obstalce_thred=0.5):
        self.tsdf_map = TSDF_Map(gpu_id)
        self.dynamic = SecondOrderLinear()
        self.planner = TrajPlanner2(is_train=True, gpu_id=gpu_id, dynamic=self.dynamic)
        # self.planner = TrajPlanner(is_train=True, gpu_id=gpu_id, n_batch=n_batch)
        self.opt = TrajOpt()
        self.is_map = False

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epi = epi
        self.zeta = zeta
        self.obstalce_thred = obstalce_thred
        return None

    def TransformPoints(self, odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps
    
    def SetMap(self, root_path, map_name):
        self.tsdf_map.ReadTSDFMap(root_path, map_name)
        self.is_map = True
        return
    
    def CalCost(self, preds, odom, goal, ahead_dist, fear, vel = None):
        traj, motion = self.planner.planning(preds, vel)
        n_state = self.dynamic.state_dim
        # traj_v = traj[..., 3:6]
        traj = traj[..., :3]
        # traj, mpc_cost = self.planner.trajGenerate(preds)
        if pp.is_lietensor(traj):
            traj = traj.translation()
        batch_size, num_p, _ = traj.shape

        if self.is_map:
            world_pred = self.TransformPoints(odom, preds)
            world_traj = self.TransformPoints(odom, traj)
            norm_inds_pred, _ = self.tsdf_map.Pos2Ind(world_pred)
            norm_inds_traj, _ = self.tsdf_map.Pos2Ind(world_traj)
            
            # Pred Obstacle Cost
            pred_cost_grid = self.tsdf_map.cost_array.T.expand(batch_size, 1, -1, -1)
            oloss_P = F.grid_sample(pred_cost_grid, norm_inds_pred[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            oloss_P = oloss_P.to(torch.float32)
            ploss = torch.mean(torch.sum(oloss_P, axis=1))

            # Traj Obstacle Cost
            traj_cost_grid = self.tsdf_map.cost_array.T.expand(batch_size, 1, -1, -1)
            oloss_M = F.grid_sample(traj_cost_grid, norm_inds_traj[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            oloss_M = oloss_M.to(torch.float32)
            oloss = torch.mean(torch.sum(oloss_M, axis=1))

            # Terrian Height loss
            height_grid = self.tsdf_map.ground_array.T.expand(batch_size, 1, -1, -1)
            hloss_M = F.grid_sample(height_grid, norm_inds_traj[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            hloss_M = torch.abs(traj[:, :, 2] - hloss_M)
            hloss = torch.mean(torch.sum(hloss_M, axis=1))

        # mtxR = self.planner.MPC.Q[..., n_state:, n_state:]
        # motion_cost = pp.bvmv(motion, mtxR, motion).sum(dim=-1)
        motion_cost = torch.mean(torch.norm(motion, dim=-1))
        mpcloss = torch.mean(motion_cost)

        # Motion Loss
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(preds[:, -1, 0:3].unsqueeze(-2), step=1.0/(num_p-1)) 
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
        wp_ds = torch.norm(traj[:, 1:num_p, :] - traj[:, 0:num_p-1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.mean(torch.log(5*mloss + 1.0), axis=1)
        mloss = torch.mean(mloss)
        
        # Goal Cost
        dis = torch.norm(goal[:, :3] - preds[:, -1, :], dim=1)
        gloss = torch.mean(dis)
        # gloss = torch.mean(torch.log(dis + 1.0))

        # # Length Cost
        # length = self.planner.length.squeeze()
        # str_length = torch.norm(traj[:, -1, :], dim=-1)
        # dis = torch.max(length-str_length, torch.zeros_like(str_length))
        # lloss = torch.mean(torch.log(dis + 1.0))
                
        # Fear labels
        wp_ds = torch.norm(traj[:, 1:num_p, :] - traj[:, 0:num_p-1, :], dim=2)
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        floss_M = torch.clone(oloss_M)[:, 1:]
        floss_M[goal_dists > ahead_dist] = 0.0
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        fear_labels = (fear_labels > self.obstalce_thred).to(torch.float32)
        
        loss1 = self.alpha*ploss + self.beta*oloss + self.gamma*hloss + self.delta*gloss + self.epi*mloss + self.zeta*mpcloss
        # loss1 = self.alpha*ploss + self.beta*oloss + self.gamma*hloss + self.delta*gloss + self.epi*mloss
        loss2 = F.binary_cross_entropy(fear, fear_labels)

        total_loss = loss1 + loss2

        # total_loss = loss1 + loss2 + self.zeta*mpc_cost.mean()

        # # Pack the individual loss components and other metrics into a dictionary
        # loss_dict = {
        #     "ploss": self.alpha*ploss,
        #     "oloss": self.beta*oloss,
        #     "hloss": self.gamma*hloss,
        #     "gloss": self.delta*gloss,
        #     "mloss": self.epi*mloss,
        #     "fear_loss": loss2,
        #     "mpc_cost": self.zeta*mpc_cost.mean()
        # }

        # # Return the total loss, trajectory, and the dictionary of individual loss components
        # return total_loss, traj, loss_dict

        return total_loss, traj, self.alpha*ploss, self.beta*oloss, self.gamma*hloss, self.delta*gloss, self.epi*mloss, loss2, self.zeta*mpcloss
