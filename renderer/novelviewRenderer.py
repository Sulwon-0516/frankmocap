
import cv2
import os
import sys
import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    BlendParams,
    MeshRasterizer,  
    SoftPhongShader,
)
from .image_utils import draw_body_bbox


class NovelViewRenderer(object):

    def __init__(self, render_size, mesh_color, fov, aspect):
        self.device = torch.device("cuda:0")
        # self.render_size = 1920

        #self.img_size = img_size

        # mesh color
        mesh_color = np.array(mesh_color)[::-1]
        self.mesh_color = torch.from_numpy(
            mesh_color.copy()).view(1, 1, 3).float().to(self.device)

        # renderer for large objects, such as whole body.
        self.render_size_large = render_size
        lights = PointLights(
            ambient_color = [[1.0, 1.0, 1.0],],
            diffuse_color = [[1.0, 1.0, 1.0],],
            device=self.device, location=[[1.0, 1.0, -30]])
        self.renderer_large = self.__get_renderer(self.render_size_large, lights, fov, aspect)



    def __get_renderer(self, render_size, lights, fov=50, aspect=1.):
        cameras = FoVPerspectiveCameras(
            device = self.device,
            znear=0.001,
            zfar=1000.0,
            fov = fov,
            aspect_ratio = aspect
        )

        raster_settings = RasterizationSettings(
            image_size = render_size,
            blur_radius = 0,
            faces_per_pixel = 10,
            bin_size = 0
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer
    

    def render(self, verts, faces, rend_img, c2w):
        # TODO numpy is slower than Tensor in cuda. 
        # consider using self.rendrer(Meshworld = mesh.clone(), R=R, T=T)
        verts = verts.copy()
        faces = faces.copy()

        '''
        R = c2w[0:3, 0:3] # (3,3)
        # get inverted translation
        T = c2w[:,3:4]
        T = T.T

        verts = ( R @ verts.transpose() ).transpose() + T
        # modify verts in camera-coordinate
        '''

        # resolve coordinate difference issues (turn back)
        # verts = verts[:,[0,2,1]]     

        # the coords of pytorch-3d is (1, 1) for upper-left and (-1, -1) for lower-right
        # so need to multiple minus for vertices
        # verts[:, 1] *= -1 

        # resolve coordinate difference issues (go to nerf system)
        # verts = verts[:,[0,2,1]]


        #c2w[0:3,2] *= -1 # flip the y and z axis
        #c2w[0:3,1] *= -1
        #c2w = c2w[[1,0,2],:] # swap y and z
        #c2w[2,:] *= -1 # flip whole world upside down
        # get inverted rotation matrix
        R = c2w[0:3, 0:3] # (3,3)
        # get inverted translation
        T = c2w[:,3:4]
        T = T.T # (1,3)
        
        R_inv = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ])
        R = R @ R_inv
        T = -T
        verts = verts + T
        verts = (R.T @ verts.T ).T

        #R = torch.from_numpy(R).float().unsqueeze(0).cuda()
        #T = torch.from_numpy(T).float().unsqueeze(0).cuda()
        #R = R
        #T = T
        #verts = verts
        
        print('mean:',verts.mean(0))
        print('min:',verts.min(0))
        print('max:',verts.max(0))

      
        renderer = self.renderer_large

        verts_tensor = torch.from_numpy(verts).float().unsqueeze(0).cuda()
        faces_tensor = torch.from_numpy(faces.copy()).long().unsqueeze(0).cuda()

        # set color
        mesh_color = self.mesh_color.repeat(1, verts.shape[0], 1)
        textures = Textures(verts_rgb = mesh_color)
        
        
        mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)
        #rend_img = renderer(mesh, Rs=R, Ts=T)
        rend_img = renderer(mesh)
    

    
        # blending rendered mesh with background image
        
        rend_img = rend_img[0].cpu().numpy()
        # print(rend_img.max())
        alpha = rend_img[:, :, 3:4]
        alpha[alpha>0] = 1.0

        rend_img = rend_img[:, :, :3] 
        maxColor = rend_img.max()
        rend_img *= 255 /maxColor #Make sure <1.0
        rend_img = alpha * rend_img[:, :, ::-1]


        return rend_img, alpha


class NovelViewVisualizer(object):
    def __init__(self, render_size, fov, aspect, c2ws):
        colors = {
            # colorbline/print/copy safe:
            'light_gray':  [0.9, 0.9, 0.9],
            'light_purple':  [0.8, 0.53, 0.53],
            'light_green': [166/255.0, 178/255.0, 30/255.0],
            'light_blue': [0.65098039, 0.74117647, 0.85882353],
        }

        self.render_size = render_size
        self.c2ws = c2ws

        self.renderer = NovelViewRenderer(
            render_size=self.render_size, 
            mesh_color=colors['light_purple'],
            fov=fov,
            aspect=aspect)


    def __render_pred_verts(self, pred_mesh_list, c2w_id):
        assert c2w_id <= len(self.c2ws), \
            f"Currently, no that many camer position exists"
       
        rend_img = np.ones((self.render_size[0], self.render_size[1], 3))

        c2w = self.c2ws[c2w_id]
        for mesh in pred_mesh_list:
            verts = mesh['vertices']
            faces = mesh['faces']
            rend_img, alpha = self.renderer.render(verts, faces, rend_img, c2w)
            
        # when mesh out of scope
        if np.isnan(rend_img).sum() > 0 or alpha.max() == 0:
            print("{}.jpg doesn't have visible mesh.")
            rend_img = np.zeros_like(rend_img, dtype = 'f')
            alpha = np.zeros_like(alpha, dtype = 'f')

        return rend_img[:,:,:3], alpha


    def visualize(self, 
        input_img, 
        c2w_id,
        body_bbox_list = None,
        pred_mesh_list = None
    ):
        # init
        res_img = input_img.copy()

        # draw body bbox
        if body_bbox_list is not None:
            body_bbox_img = draw_body_bbox(input_img, body_bbox_list)
            body_bbox_img = cv2.resize(body_bbox_img, (512,512))
            res_img = np.ones((self.render_size[0], self.render_size[1], 3))
            res_img[self.render_size[0]//2-256:self.render_size[0]//2+256, self.render_size[1]//2-256:self.render_size[1]//2+256] = body_bbox_img
        
        # render predicted meshes
        if pred_mesh_list is not None:
            rend_img, alpha = self.__render_pred_verts(pred_mesh_list, c2w_id)
            res_img = np.concatenate((res_img, rend_img), axis=1)
            # res_img = rend_img
        
        return res_img, rend_img, alpha