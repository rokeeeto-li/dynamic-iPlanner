# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================


import PIL
import json
import math
import torch
# import pypose as pp
import torchvision.transforms as transforms

# from iplanner import traj_opt
from iplanner import traj_plan
from dynamics import SecondOrderLinear

class IPlannerAlgo:
    def __init__(self, args):
        super(IPlannerAlgo, self).__init__()
        self.config(args)
        
        self.depth_transform = transforms.Compose([
            transforms.Resize(tuple(self.crop_size)),
            transforms.ToTensor()])

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.idx = 0
        # self.traj_generate = traj_opt.TrajOpt()
        self.dynamic = SecondOrderLinear()
        self.traj_planner = traj_plan.TrajPlanner2(is_train=False, dynamic=self.dynamic)
        # self.traj_planner = traj_plan.TrajPlanner(is_train=False)
        return None

    def config(self, args):
        self.model_save = args.model_save
        self.crop_size  = args.crop_size
        self.sensor_offset_x = args.sensor_offset_x
        self.sensor_offset_y = args.sensor_offset_y
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
        return None
    
    def save_traj(self, waypoints, traj, cost):
        waypoints_json = waypoints.tolist()
        traj_json = traj.tolist()
        cost_json = cost.tolist()
        
        data = {
            "waypoints": waypoints_json,
            "x_traj": traj_json,
            "cost": cost_json
        }

        json_data = json.dumps(data)
        json_path = "/home/qihang/iplanner_ws/src/iPlanner/iplanner/trajs/traj_%d.json"%(self.idx)

        with open(json_path, "w") as file:
            file.write(json_data)
            print("The traj info stored")
            self.idx += 1
    
    def plan(self, image, goal_robot_frame, vel):
        img = PIL.Image.fromarray(image)
        img = self.depth_transform(img).expand(1, 3, -1, -1)
        if torch.cuda.is_available():
            img = img.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
            vel = vel.cuda()
        with torch.no_grad():
            if vel.norm() > 1.5:
                vel_rate = vel / vel.norm()
            else:
                vel_rate = vel/1.5
            keypoints, fear = self.net(img, goal_robot_frame, vel_rate)
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat((torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), axis=1)
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        # traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        # return keypoints, traj, fear, img
            
        mpc_traj, _ = self.traj_planner.planning(keypoints)
        mpc_vel = mpc_traj[..., 3:].to(torch.device("cuda"))
        traj_gpu = mpc_traj[..., :3].to(torch.device("cuda"))

        return keypoints, mpc_vel[:,0,...], traj_gpu, fear, img
        # mpc_traj, _ = self.traj_planner.trajGenerate(keypoints)
        
        # keypts_cpu = keypoints.to(torch.device("cpu"))
        # mpc_traj, cost = self.traj_planner.trajGenerate(keypts_cpu)
        # traj_gpu = mpc_traj[..., :3].to(torch.device("cuda"))
        
        # self.save_traj(keypts_cpu, mpc_traj, cost)
        return keypoints, mpc_traj, fear, img
