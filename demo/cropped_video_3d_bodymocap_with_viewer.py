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
        parser.add_argument('--out_dir', type=str, default='./mocap_output/get_smpl_params', help='Folder of output images.')
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
        if False:   
            # for local
            parser.add_argument('--bbox_path', type=str, default='/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected', help='Path of storing bbox-npy')
            parser.add_argument('--info_3d_path', type=str, default='/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/selected/3D_info.json',  help='Path of storing bbox-npy & 3D positions')
            parser.add_argument('--cropped_path', type=str, default='/mnt/hdd/auto_colmap/20221026_dataset_iphone_statue/kaist_statue_inhee_dynamic/output/segmentations/cropped', help='Path of storing bbox-npy & 3D positions')
            parser.add_argument('--render_traj_path', type=str, default='/home/inhee/VCL/insect_recon/frankmocap/demo/camera_path_longer.json', help='Path of rendering camera')
            parser.add_argument('--frame_per_data', type=int, default=4, help='render n consecutive frames using same mesh')
        else:
            # for server
            
            parser.add_argument('--bbox_path', type=str, default='/home/disk1/inhee/auto_colmap/iphone_inhee_statue/inhee_statue_dynamic/output/segmentations/selected', help='Path of storing bbox-npy')
            parser.add_argument('--info_3d_path', type=str, default='/home/disk1/inhee/auto_colmap/iphone_inhee_statue/inhee_statue_dynamic/output/segmentations/selected/3D_info.json',  help='Path of storing bbox-npy & 3D positions')
            parser.add_argument('--cropped_path', type=str, default='/home/disk1/inhee/auto_colmap/iphone_inhee_statue/inhee_statue_dynamic/output/segmentations/cropped', help='Path of storing bbox-npy & 3D positions')
            parser.add_argument('--render_traj_path', type=str, default='/home/inhee/repos/frankmocap/demo/camera_path_longer.json', help='Path of rendering camera')
            parser.add_argument('--frame_per_data', type=int, default=4, help='render n consecutive frames using same mesh')
            
            
        parser.add_argument('--res_video_name', type=str, default='long_novel_view', help='Resultant video name')
        
        parser.add_argument('--left_nerf_images', type=str, default='/home/inhee/VCL/insect_recon/nerfstudio/renders/nerfacto_default_long', help='Resultant video left lower part')
        parser.add_argument('--right_nerf_images', type=str, default='/mnt/hdd/experiments/nerfstudio/render/nerfacto_pifu_bbox_longer_imgs', help='Resultant video right lower part')


        parser.add_argument('--save_mesh_transformed', action='store_true', help='save converted mesh')
        parser.add_argument('--save_smpl_param', action='store_true', help='save raw smpl params')
        parser.add_argument('--is_render', action='store_true', help='render and save imgs')
        
        # 3d settings
        # parser.add_argument('--use_trans', action='store_true', help='Consider center shifting')
        
        self.parser = parser

    def parse(self):
        self.opt = self.parser.parse_args()
        '''
        For easy debug
        '''
        self.opt.__setattr__('save_smpl_param', True)
        self.opt.__setattr__('no_display', True)
        # I don't need render output anymore
        self.opt.__setattr__('is_render', False)
        return self.opt

def mesh_from_output(pred_output_list, rot, trans, scale, imgSize):
    pred_mesh_list = list()
    for pred_output in pred_output_list:
        if pred_output is not None:
            if 'left_hand' in pred_output: # hand mocap # rotation isn't considered here
                assert 0, "we don't consider hand now (22.12.28)"
            else: # body mocap (includes frank/whole/total mocap)
                vertices = pred_output['pred_vertices_img']
                vertices = vertices.copy()

                # shift center back
                vertices[:, 0:2] -= imgSize/2

                # TODO wouldn't it affect scale?
                mean_d = np.mean(vertices[:, 2])
                vertices[:, 2] -= mean_d

                # rotate to nerf-coordinate
                vertices = np.einsum('ix,xj->ij', rot.T, vertices.T)
                # scale the vertices
                #vertices[:, 2] += mean_d
                vertices = vertices * scale / imgSize.max()
                vertices = vertices.T
                #vertices[:, 0:3] += trans[[0,2,1]]
                vertices[:, 0:3] += trans
                faces = pred_output['faces'].astype(np.int32)

                # also save joints 
                joints = pred_output['pred_joints_img']
                joints = joints.copy()

                # shift center back
                joints[:, 0:2] -= imgSize/2

                # TODO wouldn't it affect scale?
                mean_d = np.mean(joints[:, 2])
                joints[:, 2] -= mean_d

                # rotate to nerf-coordinate
                joints = np.einsum('ix,xj->ij', rot.T, joints.T)
                # scale the joints
                #vertices[:, 2] += mean_d
                joints = joints * scale / imgSize.max()
                joints = joints.T
                #vertices[:, 0:3] += trans[[0,2,1]]
                joints[:, 0:3] += trans
                
                pred_mesh_list.append(dict(
                    vertices = vertices,
                    faces = faces,
                    joints = joints
                ))

    return pred_mesh_list

def crop_body_mocap(args, db, body_mocap, visualizer, c2ws, seconds):
    #Setup input data to handle different types of inputs
    input_type = 'video'

    #read index txts
    if len(c2ws) > (len(db) * args.frame_per_data):
        args.__setattr__('frame_per_data', ceil(len(c2ws) / len(db)))


    #modification to extract mesh properly
    if args.save_mesh_transformed:
        args.frame_per_data = 1
        c2ws = c2ws[0:len(db)]

    fps = len(c2ws) // seconds 
    timer = Timer()
    for i in range(len(c2ws)):
        if i%args.frame_per_data == 0:
            data = db[i//args.frame_per_data]
            
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
            imgSize = np.array([[img_original_bgr.shape[1], img_original_bgr.shape[0]]])
            bbox_size = data['nerf_scale']
            pred_mesh_list = mesh_from_output(pred_output_list, data['rot_mat'], data['trans'], bbox_size, imgSize)

        if args.is_render:
            # visualization
            res_img, render_img, alpha = visualizer.visualize(
                img_original_bgr,
                i,
                pred_mesh_list = pred_mesh_list, 
                body_bbox_list = body_bbox_list)
        
            # load nerf images
            left_nerf = cv2.imread(osp.join(args.left_nerf_images, str(i).zfill(5)+'.png'))
            right_nerf = cv2.imread(osp.join(args.right_nerf_images, str(i).zfill(5)+'.png'))

            # save combined images
            comb_img_l = render_img + (1-alpha) * left_nerf
            os.makedirs(args.out_dir+"/combl", exist_ok=True)
            demo_utils.save_res_img(args.out_dir+"/combl", str(i).zfill(5)+'.png', comb_img_l)
            
            comb_img_r = render_img + (1-alpha) * right_nerf
            os.makedirs(args.out_dir+"/combl", exist_ok=True)
            demo_utils.save_res_img(args.out_dir+"/combr", str(i).zfill(5)+'.png', comb_img_r)

            nerf_img = np.concatenate((comb_img_l, comb_img_r), axis=1)
            # combined images
            res_img = np.concatenate((res_img, nerf_img), axis=0)
            
            # show result in the screen
            if not args.no_display:
                res_img = res_img.astype(np.uint8)
                ImShow(res_img)

            # save result image
            if args.out_dir is not None:
                os.makedirs(args.out_dir + '/images', exist_ok=True)
                demo_utils.save_res_img(args.out_dir + '/images', str(i).zfill(5)+'.png', res_img)

            # save predictions to pkl
            if args.save_pred_pkl:
                demo_type = 'body'
                demo_utils.save_pred_to_pkl(
                    args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        timer.toc(bPrint=True,title="Time")
        print(f"Processed : {image_path}")  

        # save meshes
        if args.save_mesh_transformed:
            # first make dirs
            mesh_path = os.path.join(args.out_dir, 'mesh')
            cmesh_path = os.path.join(args.out_dir, 'canon_mesh')
            joint_path = os.path.join(args.out_dir, 'joint')
            os.makedirs(mesh_path, exist_ok=True)
            os.makedirs(cmesh_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            # save the results.
            # remove duplicate with following setting
            if i%args.frame_per_data == 0:
                mesh_ind = data['fid']
                # save the first one only
                vertices = pred_mesh_list[0]['vertices']
                faces = pred_mesh_list[0]['faces']
                joints = pred_mesh_list[0]['joints']
                betas = pred_output_list[0]['pred_betas']
                img_joints = pred_output_list[0]['pred_joints_img']
                crop_shape = data['crop_shape']
                save_obj_mesh(os.path.join(mesh_path,str(mesh_ind).zfill(5)+'.obj'), vertices, faces)
                save_obj_mesh(os.path.join(cmesh_path,str(mesh_ind).zfill(5)+'.obj'), pred_output_list[0]['canon_verts'], faces)
                save_joints(os.path.join(joint_path, str(mesh_ind).zfill(5)+'.txt'), joints, img_joints, crop_shape, betas)


        if args.save_smpl_param:
            smpl_param_path = os.path.join(args.out_dir, 'smpl_params')
            if i%args.frame_per_data == 0:
                fid = data['fid']
                betas = pred_output_list[0]['pred_betas']
                aas = pred_output_list[0]['pred_body_pose']
                
                beta_fname = os.path.join(smpl_param_path, str(fid).zfill(5)+'_beta.np')
                aa_fname = os.path.join(smpl_param_path, str(fid).zfill(5)+'_aa.np')
                np.save(beta_fname, betas)
                np.save(aa_fname, aas)
                    
                
                
            
        
        
    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir + '/images', args.res_video_name, fps)

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

def save_joints (joints_path, joints, img_joints, crop_shape, betas):
    file = open(joints_path, 'w')

    for v in joints:
        file.write('j %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    for v in img_joints:
        file.write('ij %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    cs = crop_shape
    file.write('crop %d %d %d %d\n' % (cs[0], cs[1], cs[2], cs[3]))

    bs = betas.squeeze()
    file.write('beta')
    for b in bs:
        file.write(' %.4f' % (b))
    file.write('\n')

    file.close()


def get_rotmat(dir1, dir2, dir3):
    # +y : is upper direction here
    if False:
        x_dir = np.array([dir2])[:,[0,2,1]]
        y_dir = np.array([dir1])[:,[0,2,1]]
        z_dir = -np.array([dir3])[:,[0,2,1]]
    else:
        x_dir = np.array([dir2])
        y_dir = np.array([dir1])
        z_dir = -np.array([dir3])
    
    rot_mat = np.concatenate([x_dir, y_dir, z_dir], axis=0)
    #rot_mat = rot_mat.transpose() 
    #because it's inverse calculation here.

    return rot_mat



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

        # get rotation matrix.
        dir1 = data['3d_info']['dir1']
        dir2 = data['3d_info']['dir2']
        dir3 = data['3d_info']['dir3']
        rot_mat = get_rotmat(dir1, dir2, dir3)
        data['rot_mat'] = rot_mat
        data['trans'] = np.array(data['3d_info']['center'])


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
        data['nerf_scale'] = data['3d_info']['size']
        data['img_scale']= max(data['3d_info']['x']) - min(data['3d_info']['x'])

        # bbox location in original image
        data['org_bbox'] = np.array((x1, y1, x2-x1, y2-y1))

        # org_width 
        size = w if w > h else h
        w_org = int(size*1.2)
        x1_org = floor((x1+x2-w_org)/2)
        y1_org = floor((y1+y2-w_org)/2)
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
    
    
    with open(args.info_3d_path, 'r') as f:
        info_3d = json.load(f)

    # build initial db here.
    db = make_db(info_3d, Path(args.bbox_path), Path(args.cropped_path))

    # load rendering traj
    c2ws, fov, aspect, image_size, seconds = get_path_from_json(args.render_traj_path)

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    from renderer.novelviewRenderer import NovelViewVisualizer
    visualizer = NovelViewVisualizer(image_size, fov, aspect, c2ws)
  
    crop_body_mocap(args, db, body_mocap, visualizer, c2ws, seconds)


if __name__ == '__main__':
    main()