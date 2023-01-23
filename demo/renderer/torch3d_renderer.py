import cv2
import os
import sys
import json
import torch
import math
import random
import os.path as osp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# io utils
from pytorch3d.io import load_obj
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
    TexturesVertex
)

DPI = 200
SAMPLE_OBJ = "/home/inhee/VCL/insect_recon/torch3d/simple_stag.obj"
CAM_PATH = "/home/inhee/VCL/insect_recon/frankmocap/demo/renderer/camera_path_longer.json"

'''codes for debug start'''
def make_camera(theta=50, H=1080, W=1920):
    '''
    input:
    - theta : Fov (degree) (camera intrinsic)
    - H,W -> H/W of the model
    return:
    Poly3DCollections    
    '''
    if isinstance(W, type(None)):
        W = H
    CAM_SCALE = 0.001

    longer_side = W if W > H else H

    f = (longer_side / 2.) * (1/math.tan(theta/2*math.pi/180))
    a1 = np.array([W/2., H/2., f])*CAM_SCALE
    a2 = np.array([W/2., -H/2., f])*CAM_SCALE
    a3 = np.array([-W/2., -H/2., f])*CAM_SCALE
    a4 = np.array([-W/2., H/2., f])*CAM_SCALE
    b = np.array([0,0,0])
    
    axis = [
        [a1*2, b],
        [a2*2, b],
        [a3*2, b],
        [a4*2, b]
    ]
    
    verts = [
        [a1,a2,b],
        [a2,a3,b],
        [a3,a4,b],
        [a4,a1,b],
        [a1,a2,a3,a4],
    ]

    return verts, axis

def plot_verts(res_path, verts, i):
    os.makedirs(res_path, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cams, axis = make_camera()

    ax.add_collection3d(Poly3DCollection(
        cams, 
        facecolors = (1., 1., 1., 0.),
        linewidth = 0.1,
        edgecolors='b',
        alpha=.5))
    ax.add_collection3d(Poly3DCollection(
        axis, 
        facecolors = None,
        linewidth = 0.5,
        edgecolors='grey',
        alpha=.25))


    test_sample = verts

    X = test_sample[:,0]
    Y = test_sample[:,1]
    Z = test_sample[:,2]

    mins = np.min(test_sample, axis=0)
    maxs = np.max(test_sample, axis=0)

    rand_ind = random.sample(range(X.shape[0]), 512)
    X = X[rand_ind]
    Y = Y[rand_ind]
    Z = Z[rand_ind]

    ax.scatter(X,Y,Z, marker='.', color='orange')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(10, 10)
    fname = 'trans_volume_' + str(i) + '_0.png'
    fname = os.path.join(res_path, fname)
    plt.savefig(fname, transparent = False, dpi=DPI)

    ax.view_init(10, 100)
    fname = 'trans_volume_' + str(i) + '_90.png'
    fname = os.path.join(res_path, fname)
    plt.savefig(fname, transparent = False, dpi=DPI)


def load_system(obj_path = SAMPLE_OBJ, cam_path=CAM_PATH):
    # load test object
    loaded_obj, verts, faces =  load_test_mesh(obj=obj_path)
    
    # load camera path to test
    c2ws, fov, aspect, image_size, seconds  = get_path_from_json(camera_path_filename=cam_path)

    # define model
    NVR = NovelViewVisualizer(image_size, fov, aspect, c2ws)

    return NVR, c2ws, verts, faces


def load_test_mesh(device=torch.device("cuda"), obj=SAMPLE_OBJ, b_size = 1):
    # Load the obj and ignore the textures and materials.
    # We will focus it to be in (+-1) in all direction
    verts, faces_idx, _ = load_obj(obj)

    min_vert = verts.min()
    max_vert = verts.max()

    scaler = -min_vert if (-min_vert) > max_vert else max_vert
    scaler = scaler * 4

    verts = verts/scaler
    verts = verts - verts.mean(0)
    verts = verts.to(device)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None].repeat(b_size,1,1)  # (b_size, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    test_mesh = Meshes(
        verts=[verts.to(device) for i in range(b_size)],   
        faces=[faces.to(device) for i in range(b_size)], 
        textures=textures
    )

    return test_mesh , verts, faces
'''codes for debug end'''





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
            faces_per_pixel = 20
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
    
    def torch3d_default_render(self, verts, faces):
        renderer = self.renderer_large

        verts[:,2] += 1.5
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


        return rend_img, verts

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
        #verts[:,2] += 4

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


        return rend_img, verts


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
            rend_img, verts = self.renderer.render(verts, faces, rend_img, c2w)
            #rend_img, verts = self.renderer.torch3d_default_render(verts, faces)
            
        plot_verts("/home/inhee/VCL/insect_recon/frankmocap/demo/renderer/result/transed", verts, c2w_id)

        return rend_img[:,:,:3]


    def visualize(self, 
        c2w_id,
        pred_mesh_list = None
    ):
        rend_img = self.__render_pred_verts(pred_mesh_list, c2w_id)

        return rend_img

def save_res_img(out_dir, image_path, res_img):
    out_dir = osp.join(out_dir, "rendered")
    os.makedirs(out_dir, exist_ok=True)
    img_name = osp.basename(image_path)
    img_name = img_name[:-4] + '.jpg'           #Always save as jpg
    res_img_path = osp.join(out_dir, img_name)
    cv2.imwrite(res_img_path, res_img)
    print(f"Visualization saved: {res_img_path}")


if __name__ == '__main__':
    NVR, c2ws, verts, faces = load_system()

    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()

    TEST_DIR = "/home/inhee/VCL/insect_recon/frankmocap/demo/renderer/result"
    os.makedirs(TEST_DIR, exist_ok=True)
    if True:
        for i, c2w in enumerate(c2ws):
            pred_mesh = [dict(
                vertices = verts,
                faces = faces
            )]
            res_img = NVR.visualize(i, pred_mesh)
            save_res_img(TEST_DIR, str(i).zfill(5)+'.png', res_img)

    VIDEO_NAME="/home/inhee/VCL/insect_recon/frankmocap/demo/renderer/"+"res.mp4"
    fps = 24
    ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate {str(int(fps))} -pattern_type glob -i "{TEST_DIR}/rendered/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {VIDEO_NAME}'
    os.system(ffmpeg_cmd)