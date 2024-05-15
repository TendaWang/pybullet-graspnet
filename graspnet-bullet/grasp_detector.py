""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='../dataset/checkpoint-rs.tar', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

class Grasper():
    def __init__(self):
        self.data_dir = './doc/example_data'
        self.net=self.get_net()
    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self,data_dir):
        # load data
        color = np.array(Image.open(os.path.join(data_dir, 'rgb1.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth1.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        # camera = CameraInfo(width=1280, height=720, fx=1.206285 * 1280 / 2, fy=2.14450693 * 720 / 2, cx=1280 / 2,
        #                              cy=720 / 2, scale=1)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # workspace_mask=workspace_mask[:,100:1180]
        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud
    def process_data(self,color,depth,seg=None):
        # load data
        color=np.array(color/255.,dtype=np.float32)
        if type(seg)==type(None):
            workspace_mask=np.array([True for i in range(1280*720)]).reshape(720,1280)
        elif type(seg)==list:
            workspace_mask = np.array([False for i in range(1280 * 720)]).reshape(720, 1280)
            for i in range(int(seg[0]*0.9),min(int(1.1*seg[1]),720)):
                for j in range(int(0.9*seg[2]),min(int(1.1*seg[3]),1280)):
                    workspace_mask[i][j]=True
            # workspace_mask[seg[0]:seg[1]][seg[2]:seg[3]]=True
        else:

            workspace_mask=seg
        # workspace_mask = np.array(Image.open(os.path.join('../doc/example_data', 'workspace_mask.png')))
        # workspace_mask = np.array([True for i in range(1280*720)]).reshape(720,1280)

        camera = CameraInfo(width=1280., height=720., fx=1.206285 * 1280 / 2., fy=2.14450693 * 720 / 2., cx=1280 / 2.,
                                     cy=720 / 2., scale=1000.0)
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # workspace_mask=workspace_mask[:,100:1180]
        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        lc=len(cloud_masked)
        if  lc>= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(lc)
            idxs2 = np.random.choice(lc, cfgs.num_point - lc, replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud
    def get_grasps(self,net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self,gg, cloud):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:20]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def demo(self,inputpic=False,color=None,depth=None,seg=None,show=False):
        net = self.net
        if inputpic== False:
            end_points, cloud = self.get_and_process_data(self.data_dir)
        else:
            end_points, cloud = self.process_data(color,depth,seg)
        gg = self.get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        if show:
            self.vis_grasps(gg, cloud)
            return gg, cloud
        else:
            return gg,cloud



if __name__=='__main__':
    grasper=Grasper()
    gg,cloud=grasper.demo(inputpic=True,color=1,depth=1)
    gg1=gg[0]
    print(gg1.translation)
    print(gg1.rotation_matrix)
    print(gg1)

