# Copyright (c) Facebook, Inc. and its affiliates.
'''
Run body mocap on the cropped videos
'''

import os
import sys
import os.path as osp
from pathlib import Path, PurePath
from math import floor, ceil
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
from datetime import datetime

from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import argparse

class DemoOptions():

    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # parser.add_argument('--checkpoint', required=False, default=default_checkpoint, help='Path to pretrained checkpoint')
        default_checkpoint_body_smpl ='./extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
        parser.add_argument('--checkpoint_body_smpl', required=False, default=default_checkpoint_body_smpl, help='Path to pretrained checkpoint')
        default_checkpoint_body_smplx ='./extra_data/body_module/pretrained_weights/smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt'
        parser.add_argument('--checkpoint_body_smplx', required=False, default=default_checkpoint_body_smplx, help='Path to pretrained checkpoint')
        default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        parser.add_argument('--checkpoint_hand', required=False, default=default_checkpoint_hand, help='Path to pretrained checkpoint')

        # input options 
        #parser.add_argument('--input_path', type=str, default=None, help="""Path of video, image, or a folder where image files exists""")
        #parser.add_argument('--start_frame', type=int, default=0, help='given a sequence of frames, set the starting frame')
        #parser.add_argument('--end_frame', type=int, default=float('inf'), help='given a sequence of frames, set the last frame')
        parser.add_argument('--pkl_dir', type=str, help='Path of storing pkl files that store the predicted results')
        parser.add_argument('--openpose_dir', type=str, help='Directory of storing the prediction of openpose prediction')

        # output options
        parser.add_argument('--out_dir', type=str, default=None, help='Folder of output images.')
        # parser.add_argument('--pklout', action='store_true', help='Export mocap output as pkl file')
        parser.add_argument('--save_bbox_output', action='store_true', help='Save the bboxes in json files (bbox_xywh format)')
        parser.add_argument('--save_pred_pkl', action='store_true', help='Save the predictions (bboxes, params, meshes in pkl format')
        parser.add_argument("--save_mesh", action='store_true', help="Save the predicted vertices and faces")
        parser.add_argument("--save_frame", action='store_true', help='Save the extracted frames from video input or webcam')

        # Other options
        parser.add_argument('--single_person', action='store_true', help='Reconstruct only one person in the scene with the biggest bbox')
        parser.add_argument('--no_display', action='store_true', help='Do not visualize output on the screen')
        parser.add_argument('--no_video_out', action='store_true', help='Do not merge rendered frames to video (ffmpeg)')
        parser.add_argument('--smpl_dir', type=str, default='./extra_data/smpl/', help='Folder where smpl files are located.')
        parser.add_argument('--skip', action='store_true', help='Skip there exist already processed outputs')
        parser.add_argument('--video_url', type=str, default=None, help='URL of YouTube video, or image.')
        parser.add_argument('--download', '-d', action='store_true', help='Download YouTube video first (in webvideo folder), and process it')

        # Body mocap specific options
        parser.add_argument('--use_smplx', action='store_true', help='Use SMPLX model for body mocap')

        # Hand mocap specific options
        parser.add_argument('--view_type', type=str, default='third_view', choices=['third_view', 'ego_centric'],
            help = "The view type of input. It could be ego-centric (such as epic kitchen) or third view")
        parser.add_argument('--crop_type', type=str, default='no_crop', choices=['hand_crop', 'no_crop'],
            help = """ 'hand_crop' means the hand are central cropped in input. (left hand should be flipped to right). 
                        'no_crop' means hand detection is required to obtain hand bbox""")
        
        # Whole motion capture (FrankMocap) specific options
        parser.add_argument('--frankmocap_fast_mode', action='store_true', help="Use fast hand detection mode for whole body motion capture (frankmocap)")

        # renderer
        parser.add_argument("--renderer_type", type=str, default="pytorch3d", 
            choices=['pytorch3d', 'opendr', 'opengl_gui', 'opengl'], help="type of renderer to use")

        # cropped data loader
        parser.add_argument('--bbox_path', type=str, default='/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected', help='Path of storing bbox-npy')
        parser.add_argument('--info_3d_path', type=str, default='/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected/3D_info.json',  help='Path of storing bbox-npy & 3D positions')
        parser.add_argument('--cropped_path', type=str, default='/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/cropped', help='Path of storing bbox-npy & 3D positions')
        
        parser.add_argument('--res_video_name', type=str, default='simple_crop_mocap', help='Resultant video name')
        self.parser = parser

    

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt



def crop_body_mocap(args, db, body_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type = 'video'

    video_frame = 0
    timer = Timer()
    for data in db:
        timer.tic()
        image_path = str(data['crop_path'])
        img_original_bgr  = cv2.imread(image_path)
        body_bbox_list = [data['bbox']]
        hand_bbox_list = [None, ] * len(body_bbox_list)
        
        print("--------------------------------------")

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        #Sort the bbox using bbox size 
        # (no need ordering, because single bbox list)
        if args.single_person and len(body_bbox_list)>0:
            body_bbox_list = [body_bbox_list[0], ]       

        # Body Pose Regression
        pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualization
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list = pred_mesh_list, 
            body_bbox_list = body_bbox_list)
        
        # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save result image
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'body'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        timer.toc(bPrint=True,title="Time")
        print(f"Processed : {image_path}")  

    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.res_video_name)

    cv2.destroyAllWindows()


def make_db(info_3d, bbox_path, cropped_path):
    fids = info_3d['fids']
    order_list = info_3d['order_list']

    db = []
    for i, fid in enumerate(fids):
        if order_list[i] == -1:
            # case where there's no bbox in the image
            continue
        data = {}
        data['crop_path'] = cropped_path / (str(fid).zfill(5)+'.png')
        data['3d_info'] = info_3d[str(order_list[i])]
        data['fid'] = fid
        data['ns_data_idx'] = order_list[i]
        
        # absolute position of crop
        x0 = data['3d_info']['x'][0]
        y0 = data['3d_info']['y'][0]

        # get bounding box's relative position in ours
        bbox_np = np.load(str(bbox_path/(str(fid).zfill(5)+'.npy')))[0]

        # absolute position of bbox 
        x1 = int(floor(bbox_np[0]))
        x2 = int(floor(bbox_np[2]))
        y1 = int(ceil(bbox_np[1]))
        y2 = int(ceil(bbox_np[3]))

        w = int(x2 - x1)
        h = int(y2 - y1)

        # add bbox in relative position
        data['bbox'] = np.array((x1-x0, y1-y0, w, h)).astype(np.int32) 

        db.append(data)

    return db


def main():
    args = DemoOptions().parse()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Get valid image ids
    if not Path(args.info_3d_path).exists():
        assert 0, "invalid info_3d_path. use proper json path"
    if not Path(args.bbox_path).exists():
        assert 0, "invalid bbox_path. use proper path of bbox"
    if not Path(args.cropped_path).exists():
        assert 0, "invalid info_3d_path. use proper path of cropped_path"
    
    
    with open(args.info_3d_path, 'r') as f:
        info_3d = json.load(f)

    # build initial db here.
    db = make_db(info_3d, Path(args.bbox_path), Path(args.cropped_path))

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)
  
    crop_body_mocap(args, db, body_mocap, visualizer)


if __name__ == '__main__':
    main()