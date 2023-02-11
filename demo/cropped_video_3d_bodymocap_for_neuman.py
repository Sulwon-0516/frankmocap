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
        parser.add_argument('--out_dir', type=str, default='./mocap_output/extract_neuman', help='Folder of output images.')
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
        parser.add_argument('--render_traj_path', type=str, default='/home/inhee/VCL/insect_recon/frankmocap/demo/camera_path_longer.json', help='Path of rendering camera')
        parser.add_argument('--frame_per_data', type=int, default=4, help='render n consecutive frames using same mesh')
        
        parser.add_argument('--res_video_name', type=str, default='long_novel_view', help='Resultant video name')
        
        parser.add_argument('--left_nerf_images', type=str, default='/home/inhee/VCL/insect_recon/nerfstudio/renders/nerfacto_default_long', help='Resultant video left lower part')
        parser.add_argument('--right_nerf_images', type=str, default='/mnt/hdd/experiments/nerfstudio/render/nerfacto_pifu_bbox_longer_imgs', help='Resultant video right lower part')


        parser.add_argument('--save_mesh_transformed', action='store_true', help='save the converted mesh')
        parser.add_argument('--save_for_neuman', action='store_true', help='extract information for mesh')
        # 3d settings
        # parser.add_argument('--use_trans', action='store_true', help='Consider center shifting')
        
        self.parser = parser

    def parse(self):
        self.opt = self.parser.parse_args()
        '''
        For easy debug
        '''
        self.opt.__setattr__('no_display', True)
        self.opt.__setattr__('save_for_neuman', True)
        return self.opt

def crop_body_mocap(args, db, body_mocap, visualizer, c2ws, seconds):
    #Setup input data to handle different types of inputs
    input_type = 'video'

    #read index txts
    if len(c2ws) > (len(db) * args.frame_per_data):
        args.__setattr__('frame_per_data', ceil(len(c2ws) / len(db)))

    timer = Timer()
    for data in db:     
        timer.tic()
        image_path = str(data['crop_path'])
        img_original_bgr  = cv2.imread(image_path)
        body_bbox_list = [data['bbox']]
        hand_bbox_list = [None, ] * len(body_bbox_list)

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

        timer.toc(bPrint=True,title="Time")
        print(f"Processed : {image_path}")  
        
        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        if False:
            # visualization
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list = pred_mesh_list, 
                body_bbox_list = body_bbox_list)
            img_dir = osp.join(args.out_dir, 'imgs')
            os.makedirs(img_dir, exist_ok=True)
            demo_utils.save_res_img(img_dir, image_path, res_img)
                
        # save information for Neuman
        if args.save_for_neuman:
            neuman_data_path = os.path.join(args.out_dir, 'neuman_data')
            os.makedirs(neuman_data_path, exist_ok=True)
            
            # shift img_joints into org_image.
            # img joints are in (x,y) order
            mesh_ind = data['fid']
            img_joints = pred_output_list[0]['pred_smpl_joints_img']
            crop_shape = data['crop_shape']
            x1_org = crop_shape[0]
            y1_org = crop_shape[1]
            org_img_joints = img_joints[:,0:2] + np.array([[x1_org, y1_org]])
            

            save_data = []
            save_data.append(
                {
                    'verts':pred_output_list[0]['pred_vertices_smpl'],
                    'j3d_all54': pred_output_list[0]['pred_smpl_joints'],
                    'pj2d_org': org_img_joints,
                    'poses': pred_output_list[0]['pred_body_pose'],
                    'betas':pred_output_list[0]['pred_betas']
                }
            )
            np.savez(os.path.join(neuman_data_path, str(mesh_ind).zfill(5)+'.npz'), results=save_data)
            
    demo_utils.gen_video_out(args.out_dir + '/imgs', 'result.mp4', 5)
    cv2.destroyAllWindows()


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.8f %.8f %.8f\n' % (v[0], v[1], v[2]))
    if faces is not None:
        for f in faces:
            if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
                continue
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()




def make_db(bbox_path, cropped_path):
    db = []
    for cur, dirs, files in os.walk(bbox_path):
        for file in sorted(files):
            if file.endswith('npy'):
                fname = osp.basename(file)[:-4]
                fid = int(fname)
                bbox_np = np.load(osp.join(cur, file))[0]
                
                data = {}
                data['fid'] = fid
                data['crop_path'] = osp.join(cropped_path, fname+'.png')

                # absolute position of bbox 
                x1 = int(floor(bbox_np[0]))
                x2 = int(floor(bbox_np[2]))
                y1 = int(ceil(bbox_np[1]))
                y2 = int(ceil(bbox_np[3]))

                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # org_width 
                size = w if w > h else h
                w_org = int(size*1.2)
                x1_org = floor((x1+x2-w_org)/2)
                y1_org = floor((y1+y2-w_org)/2)
                data['bbox'] = np.array((x1-x1_org, y1-y1_org, w, h)).astype(np.int32) 
                data['crop_shape'] = np.array((x1_org, y1_org, w_org, w_org))
                
                db.append(data)

    return db



def get_path_from_json(camera_path_filename):
    """Takes a camera path dictionary and returns a properties of camera.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """
    with open(camera_path_filename, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
    seconds = camera_path["seconds"]

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = np.array(camera["camera_to_world"]).reshape(4, 4)[:3]
        c2ws.append(c2w)
        # field of view
        fov = camera["fov"]
        aspect = camera["aspect"]
        pp_h = image_height / 2.0
        focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
        fxs.append(focal_length)
        fys.append(focal_length)

    #camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    
    return c2ws, fov, aspect, (image_height, image_width), seconds


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
    
    # build initial db here.
    db = make_db(Path(args.bbox_path), Path(args.cropped_path))

    # load rendering traj
    c2ws, fov, aspect, image_size, seconds = get_path_from_json(args.render_traj_path)

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    from renderer.screen_free_visualizer import Visualizer
    visualizer = Visualizer('pytorch3d')
  
    crop_body_mocap(args, db, body_mocap, visualizer, c2ws, seconds)


if __name__ == '__main__':
    main()