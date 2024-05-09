# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import tqdm
import time
import torch
import json
import wandb
import random
import argparse
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms

from planner_net import PlannerNet
from dataloader import PlannerData, MultiEpochsDataLoader
from torchutil import EarlyStopScheduler
# from traj_cost import TrajCost
from traj_cost import MulLayerCost
from traj_viz import TrajViz

torch.set_default_dtype(torch.float32)

class PlannerNetTrainer:
    def __init__(self):
        self.root_folder = os.getenv('EXPERIMENT_DIRECTORY', os.getcwd())
        self.load_config()
        self.parse_args()
        self.prepare_model()
        self.prepare_data()
        if self.args.training == True:
            self.init_wandb()
        else:
            print("Testing Mode")
        
    def init_wandb(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        root_len = len(self.root_folder)
        model_name = self.args.model_save[root_len+8:-3]
        print("Model Name: ", model_name)
        # Initialize wandb
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="dynamic_iPlanner",
            # Set the run name to current date and time
            name=model_name+date_time_str + "adamW",
            config={
                "learning_rate": self.args.lr,
                "architecture": "PlannerNet",  # Replace with your actual architecture
                "dataset": self.args.data_root,  # Assuming this holds the dataset name
                "epochs": self.args.epochs,
                "goal_step": self.args.goal_step,
                "max_episode": self.args.max_episode,
                "fear_ahead_dist": self.args.fear_ahead_dist,
                "cost-alpha": self.args.alpha,
                "cost-beta": self.args.beta,
                "cost-gamma": self.args.gamma,
                "cost-delta": self.args.delta,
                "cost-epi": self.args.epi,
                "cost-zeta": self.args.zeta,
                "obs_threshold": self.args.obs_threshold,
            }
        )

    def load_config(self):
        with open(os.path.join(os.path.dirname(self.root_folder), 'config', 'training_config.json')) as json_file:
            self.config = json.load(json_file)

    def prepare_model(self):
        self.net = PlannerNet(self.args.in_channel, k=self.args.knodes)
        if self.args.resume == True or not self.args.training:
            self.net, self.best_loss = torch.load(self.args.model_save, map_location=torch.device("cpu"))
            print("Resume training from best loss: {}".format(self.best_loss))
        else:
            self.best_loss = float('Inf')

        if torch.cuda.is_available():
            print("Available GPU list: {}".format(list(range(torch.cuda.device_count()))))
            print("Runnin on GPU: {}".format(self.args.gpu_id))
            self.net = self.net.cuda(self.args.gpu_id)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        self.scheduler = EarlyStopScheduler(self.optimizer, factor=self.args.factor, verbose=True, min_lr=self.args.min_lr, patience=self.args.patience)

    def prepare_data(self):
        ids_path = os.path.join(self.args.data_root, self.args.env_id)
        with open(ids_path) as f:
            self.env_list = [line.rstrip() for line in f.readlines()]

        depth_transform = transforms.Compose([
            transforms.Resize((self.args.crop_size)),
            transforms.ToTensor()])
        
        total_img_data = 0
        track_id = 0
        test_env_id = min(self.args.test_env_id, len(self.env_list)-1)
        
        self.train_loader_list = []
        self.val_loader_list   = []
        self.traj_cost_list    = []
        self.traj_viz_list     = []
        
        for env_name in tqdm.tqdm(self.env_list):
            if not self.args.training and track_id != test_env_id:
                track_id += 1
                continue
            is_anymal_frame = False
            sensorOffsetX = 0.0
            camera_tilt = 0.0
            if 'anymal' in env_name:
                is_anymal_frame = True
                sensorOffsetX = self.args.sensor_offsetX_ANYmal
                camera_tilt = self.args.camera_tilt
            elif 'tilt' in env_name:
                camera_tilt = self.args.camera_tilt
            data_path = os.path.join(*[self.args.data_root, self.args.env_type, env_name])

            train_data = PlannerData(root=data_path,
                                     train=True, 
                                     transform=depth_transform,
                                     sensorOffsetX=sensorOffsetX,
                                     is_robot=is_anymal_frame,
                                     goal_step=self.args.goal_step,
                                     max_episode=self.args.max_episode,
                                     max_depth=self.args.max_camera_depth,
                                     v_ref=self.args.v_ref,
                                     v_num=self.args.v_num)
            
            total_img_data += len(train_data)
            train_loader = MultiEpochsDataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
            self.train_loader_list.append(train_loader)

            val_data = PlannerData(root=data_path,
                                   train=False,
                                   transform=depth_transform,
                                   sensorOffsetX=sensorOffsetX,
                                   is_robot=is_anymal_frame,
                                   goal_step=self.args.goal_step,
                                   max_episode=self.args.max_episode,
                                   max_depth=self.args.max_camera_depth,
                                   v_ref=self.args.v_ref,
                                   v_num=self.args.v_num)

            val_loader = MultiEpochsDataLoader(val_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
            self.val_loader_list.append(val_loader)

            # Load Map and Trajectory Class
            map_name = "tsdf1"
            traj_cost = MulLayerCost(self.args.gpu_id, self.args.batch_size, alpha=self.args.alpha, beta=self.args.beta, 
                                     gamma=self.args.gamma, delta=self.args.delta, epi=self.args.epi, zeta=self.args.zeta, obstalce_thred=self.args.obs_threshold)
            # traj_cost = TrajCost(self.args.gpu_id, self.args.batch_size)
            traj_cost.SetMap(data_path, map_name)

            self.traj_cost_list.append(traj_cost)
            self.traj_viz_list.append(TrajViz(data_path, map_name=map_name, cameraTilt=camera_tilt))
            track_id += 1

            # break 
                        
        print("Data Loading Completed!")
        print("Number of image: %d | Number of goal-image pairs: %d"%(total_img_data, total_img_data * (int)(self.args.max_episode / self.args.goal_step)))
        
        return None

    # def MapObsLoss(self, preds, fear, traj_cost, odom, goal, step=0.1):
    #     # return traj_cost.CalCost(preds, odom, goal, self.args.fear_ahead_dist, fear)
        
    #     # waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)
    #     # loss1, fear_labels = traj_cost.CostofTraj(waypoints, odom, goal, ahead_dist=self.args.fear_ahead_dist)
    #     traj, mpc_cost = traj_cost.planner.trajGenerate(preds)
    #     loss1, fear_labels = traj_cost.CostofTraj(traj, odom, goal, ahead_dist=self.args.fear_ahead_dist)
    #     loss2 = F.binary_cross_entropy(fear, fear_labels)
    #     # return loss1+loss2, traj
    #     return loss1+loss2+mpc_cost.mean(), traj
    
    def train_epoch(self, epoch):
        loss_sum = 0.0
        p_sum, o_sum, h_sum, g_sum, m_sum, f_sum, mpc_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        env_num = len(self.train_loader_list)
        
        # Zip the lists and convert to a list of tuples
        combined = list(zip(self.train_loader_list, self.traj_cost_list))
        # Shuffle the combined list
        random.shuffle(combined)

        # Iterate through shuffled pairs
        for env_id, (loader, traj_cost) in enumerate(combined):
            train_loss, batches = 0, len(loader)
            train_ploss, train_oloss, train_hloss, train_gloss, train_mloss, train_fearloss, train_mpc_cost = 0, 0, 0, 0, 0, 0, 0

            enumerater = tqdm.tqdm(enumerate(loader))
            for batch_idx, inputs in enumerater:
                if torch.cuda.is_available():
                    image = inputs[0].cuda(self.args.gpu_id)
                    odom  = inputs[1].cuda(self.args.gpu_id)
                    goal  = inputs[2].cuda(self.args.gpu_id)
                    vel   = inputs[3].cuda(self.args.gpu_id)
                self.optimizer.zero_grad()
                # normlized_vel = vel / self.args.v_ref
                preds, fear = self.net(image, goal, vel)
                # print("fear is", torch.sum(fear))

                loss, _, ploss, oloss, hloss, gloss, mloss, fearloss, mpc_cost = traj_cost.CalCost(preds, odom, goal, self.args.fear_ahead_dist, fear)
                # loss, _ = self.MapObsLoss(preds, fear, traj_cost, odom, goal)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_ploss += ploss.item()
                train_oloss += oloss.item()
                train_hloss += hloss.item()
                train_gloss += gloss.item()
                train_mloss += mloss.item()
                train_fearloss += fearloss.item()
                train_mpc_cost += mpc_cost.mean().item()
                enumerater.set_description("Epoch: %d in Env: (%d/%d) - train loss: %.4f on %d/%d" % (epoch, env_id+1, env_num, train_loss/(batch_idx+1), batch_idx, batches))
            
            loss_sum += train_loss/(batch_idx+1)
            p_sum += train_ploss/(batch_idx+1)
            o_sum += train_oloss/(batch_idx+1)
            h_sum += train_hloss/(batch_idx+1)
            g_sum += train_gloss/(batch_idx+1)
            m_sum += train_mloss/(batch_idx+1)
            f_sum += train_fearloss/(batch_idx+1)
            mpc_sum += train_mpc_cost/(batch_idx+1)

            wandb.log({"Running Loss": train_loss/(batch_idx+1)})
            
        loss_sum /= env_num
        p_sum /= env_num
        o_sum /= env_num
        h_sum /= env_num
        g_sum /= env_num
        m_sum /= env_num
        f_sum /= env_num
        mpc_sum /= env_num

        return loss_sum, p_sum, o_sum, h_sum, g_sum, m_sum, f_sum, mpc_sum
        
    def train(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        
        self.args.log_save += (date_time_str + ".txt")
        open(self.args.log_save, 'w').close()

        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_loss, train_p, train_o, train_h, train_g, train_m, train_f, train_mpc = self.train_epoch(epoch)
            val_loss, val_p, val_o, val_h, val_g, val_m, val_f, val_mpc= self.evaluate(is_visualize=False)
            duration = (time.time() - start_time) / 60 # minutes

            self.log_message("Epoch: %d | Training Loss: %f | Val Loss: %f | Duration: %f" % (epoch, train_loss, val_loss, duration))
            # Log metrics to wandb
            wandb.log({"Avg Training Loss": train_loss, 
                       "Validation Loss": val_loss, 
                       "Duration (min)": duration})
            wandb.log({"train":
                       {"Train-Pred obs": train_p, 
                       "Train-Traj obs": train_o, 
                       "Train-Traj height": train_h, 
                       "Train-Goal": train_g, 
                       "Train-Motion": train_m, 
                       "Train-Fear": train_f, 
                       "Train-MPC": train_mpc},
                       "val":
                       {"Val-Pred obs": val_p,
                        "Val-Traj obs": val_o,
                        "Val-Traj height": val_h,
                        "Val-Goal": val_g,
                        "Val-Motion": val_m,
                        "Val-Fear": val_f,
                        "Val-MPC": val_mpc}})
            
            if val_loss < self.best_loss:
                self.log_message("Save model of epoch %d" % epoch)
                torch.save((self.net, val_loss), self.args.model_save)
                self.best_loss = val_loss
                self.log_message("Current val loss: %.4f" % self.best_loss)
                self.log_message("Epoch: %d model saved | Current Min Val Loss: %f" % (epoch, val_loss))

            self.log_message("------------------------------------------------------------------------")
            if self.scheduler.step(val_loss):
                self.log_message('Early Stopping!')
                break
            
         # Close wandb run at the end of training
        self.wandb_run.finish()
    
    def log_message(self, message):
        with open(self.args.log_save, 'a') as f:
            f.writelines(message)
            f.write('\n')
        print(message)

    def evaluate(self, is_visualize=False):
            self.net.eval()
            test_loss = 0   # Declare and initialize test_loss
            test_ploss = 0
            test_oloss = 0
            test_hloss = 0
            test_gloss = 0
            test_mloss = 0
            test_fearloss = 0
            test_mpc_cost = 0

            total_batches = 0  # Count total number of batches
            with torch.no_grad():
                for _, (val_loader, traj_cost, traj_viz) in enumerate(zip(self.val_loader_list, self.traj_cost_list, self.traj_viz_list)):
                    preds_viz = []
                    wp_viz = []
                    for batch_idx, inputs in enumerate(val_loader):
                        total_batches += 1  # Increment total number of batches
                        if torch.cuda.is_available():
                            image = inputs[0].cuda(self.args.gpu_id)
                            odom  = inputs[1].cuda(self.args.gpu_id)
                            goal  = inputs[2].cuda(self.args.gpu_id)
                            vel   = inputs[3].cuda(self.args.gpu_id)

                        # normlized_vel = vel / self.args.v_ref
                        preds, fear = self.net(image, goal, vel)
                        loss, waypoints, ploss, oloss, hloss, gloss, mloss, fearloss, mpc_cost = traj_cost.CalCost(preds, odom, goal, self.args.fear_ahead_dist, fear, vel=vel)
                        # loss, waypoints = self.MapObsLoss(preds, fear, traj_cost, odom, goal)
                        test_loss += loss.item()
                        test_ploss += ploss.item()
                        test_oloss += oloss.item()
                        test_hloss += hloss.item()
                        test_gloss += gloss.item()
                        test_mloss += mloss.item()
                        test_fearloss += fearloss.item()
                        test_mpc_cost += mpc_cost.mean().item()

                        if is_visualize and len(preds_viz) < self.args.visual_number:
                            if batch_idx == 0:
                                image_viz = image
                                odom_viz = odom
                                goal_viz = goal
                                fear_viz = fear
                            else:
                                image_viz = torch.cat((image_viz, image), dim=0)
                                odom_viz  = torch.cat((odom_viz, odom),   dim=0)
                                goal_viz  = torch.cat((goal_viz, goal),   dim=0)
                                fear_viz  = torch.cat((fear_viz, fear),   dim=0)
                            preds_viz.extend(preds.tolist())
                            wp_viz.extend(waypoints.tolist())

                    if is_visualize:
                        max_n = min(len(wp_viz), self.args.visual_number)
                        preds_viz = torch.tensor(preds_viz[:max_n])
                        wp_viz    = torch.tensor(wp_viz[:max_n])
                        odom_viz  = odom_viz[:max_n].cpu()
                        goal_viz  = goal_viz[:max_n].cpu()
                        fear_viz  = fear_viz[:max_n, :].cpu()
                        image_viz = image_viz[:max_n].cpu()
                        # visual trajectory and images
                        traj_viz.VizTrajectory(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz)
                        traj_viz.VizImages(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, image_viz)

                test_loss /= total_batches
                test_ploss /= total_batches
                test_oloss /= total_batches
                test_hloss /= total_batches
                test_gloss /= total_batches
                test_mloss /= total_batches
                test_fearloss /= total_batches
                test_mpc_cost /= total_batches

                return test_loss, test_ploss, test_oloss, test_hloss, test_gloss, test_mloss, test_fearloss, test_mpc_cost
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Training script for PlannerNet')

        # dataConfig
        parser.add_argument("--data-root", type=str, default=os.path.join(self.root_folder, self.config['dataConfig'].get('data-root')), help="dataset root folder")
        parser.add_argument('--env-id', type=str, default=self.config['dataConfig'].get('env-id'), help='environment id list')
        parser.add_argument('--env_type', type=str, default=self.config['dataConfig'].get('env_type'), help='the dataset type')
        parser.add_argument('--crop-size', nargs='+', type=int, default=self.config['dataConfig'].get('crop-size'), help='image crop size')
        parser.add_argument('--max-camera-depth', type=float, default=self.config['dataConfig'].get('max-camera-depth'), help='maximum depth detection of camera, unit: meter')

        # modelConfig
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, self.config['modelConfig'].get('model-save')), help="model save point")
        parser.add_argument('--resume', type=str, default=self.config['modelConfig'].get('resume'))
        parser.add_argument('--in-channel', type=int, default=self.config['modelConfig'].get('in-channel'), help='goal input channel numbers')
        parser.add_argument("--knodes", type=int, default=self.config['modelConfig'].get('knodes'), help="number of max nodes predicted")
        parser.add_argument("--goal-step", type=int, default=self.config['modelConfig'].get('goal-step'), help="number of frames betwen goals")
        parser.add_argument("--max-episode", type=int, default=self.config['modelConfig'].get('max-episode-length'), help="maximum episode frame length")
        parser.add_argument("--v-ref", type=float, default=self.config['modelConfig'].get('v-ref'), help="reference velocity")
        parser.add_argument("--v-num", type=int, default=self.config['modelConfig'].get('v-num'), help="velocity level amount")

        # costConfig
        parser.add_argument("--alpha", type=float, default=self.config['costConfig'].get('alpha'), help="weight of pred obstacle cost")
        parser.add_argument("--beta", type=float, default=self.config['costConfig'].get('beta'), help="weight of traj obstacle cost")
        parser.add_argument("--gamma", type=float, default=self.config['costConfig'].get('gamma'), help="weight of traj height cost")
        parser.add_argument("--delta", type=float, default=self.config['costConfig'].get('delta'), help="weight of goal cost")
        parser.add_argument("--epi", type=float, default=self.config['costConfig'].get('epi'), help="weight of motion cost")
        parser.add_argument("--zeta", type=float, default=self.config['costConfig'].get('zeta'), help="weight of controller cost")
        parser.add_argument("--obs-threshold", type=float, default=self.config['costConfig'].get('obs-threshold'), help="obstacle threshold")
        
        # trainingConfig
        parser.add_argument('--training', type=str, default=self.config['trainingConfig'].get('training'))
        parser.add_argument("--lr", type=float, default=self.config['trainingConfig'].get('lr'), help="learning rate")
        parser.add_argument("--factor", type=float, default=self.config['trainingConfig'].get('factor'), help="ReduceLROnPlateau factor")
        parser.add_argument("--min-lr", type=float, default=self.config['trainingConfig'].get('min-lr'), help="minimum lr for ReduceLROnPlateau")
        parser.add_argument("--patience", type=int, default=self.config['trainingConfig'].get('patience'), help="patience of epochs for ReduceLROnPlateau")
        parser.add_argument("--epochs", type=int, default=self.config['trainingConfig'].get('epochs'), help="number of training epochs")
        parser.add_argument("--batch-size", type=int, default=self.config['trainingConfig'].get('batch-size'), help="number of minibatch size")
        parser.add_argument("--w-decay", type=float, default=self.config['trainingConfig'].get('w-decay'), help="weight decay of the optimizer")
        parser.add_argument("--num-workers", type=int, default=self.config['trainingConfig'].get('num-workers'), help="number of workers for dataloader")
        parser.add_argument("--gpu-id", type=int, default=self.config['trainingConfig'].get('gpu-id'), help="GPU id")

        # logConfig
        parser.add_argument("--log-save", type=str, default=os.path.join(self.root_folder, self.config['logConfig'].get('log-save')), help="train log file")
        parser.add_argument('--test-env-id', type=int, default=self.config['logConfig'].get('test-env-id'), help='the test env id in the id list')
        parser.add_argument('--visual-number', type=int, default=self.config['logConfig'].get('visual-number'), help='number of visualized trajectories')

        # sensorConfig
        parser.add_argument('--camera-tilt', type=float, default=self.config['sensorConfig'].get('camera-tilt'), help='camera tilt angle for visualization only')
        parser.add_argument('--sensor-offsetX-ANYmal', type=float, default=self.config['sensorConfig'].get('sensor-offsetX-ANYmal'), help='anymal front camera sensor offset in X axis')
        parser.add_argument("--fear-ahead-dist", type=float, default=self.config['sensorConfig'].get('fear-ahead-dist'), help="fear lookahead distance")

        self.args = parser.parse_args()

 
def main():
    trainer = PlannerNetTrainer()
    if trainer.args.training == True:
        trainer.train()
    trainer.evaluate(is_visualize=True)

if __name__ == "__main__":
    main()
