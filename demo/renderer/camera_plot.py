from logging import root
from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import rand
from tqdm import tqdm
import numpy as np
import math
import random
import os
import imageio
import glob

from torch3d_renderer import get_path_from_json, load_test_mesh
    
COLORS = [
    'b',
    'g',
    'r',
    'c',
    'm',
    'y',
    'k'
]


# default : nerf camera
TORCH3D_CAMERA = True


RES_PATH = "./result"
GIF_PATH = "./gif_result"

USE_GUIDE = False
USE_AXIS = True

N_frame = 10
ang_interv = 180 / N_frame
ANGLES_phi = [a * ang_interv for a in range(N_frame)]
ANGLES_theta = [a * ang_interv for a in range(N_frame)]

SCALER = 1
DPI = 200
#V_r = 0.01 * SCALER
CAMERA_SIZE = 0.00005
line_length = 0.5 # camera axis line length

n_points = 512



class CamerPlot():
    def __init__(self, cpath, EXP_NAME, is_torch3d = TORCH3D_CAMERA,fov = 30):
        # intialize camera shape
        c2ws, fov, aspect, image_size, seconds = get_path_from_json(cpath)

        self.c2ws = c2ws
        self.image_size = image_size
        self.H = image_size[0]
        self.W = image_size[1]
        self.fov = fov
        self.verts, self.axis = self.make_camera(theta = fov, is_torch3d = is_torch3d)
        self.camera_default_color = 'grey'
        self.camera_selected_color = 'r'
        self.camera_axis = 'silver'

        self.min_points = np.array([100., 100., 100.])
        self.max_points = np.array([-100., -100., -100.])

        self.z_c = np.array([0,0,0.])

        self.n_view = len(c2ws)

        self.Rs=[]
        self.Ts=[]
        self.filenames = []

        self.tot_epch = N_frame

        self.EXP_NAME = EXP_NAME

        os.makedirs(RES_PATH, exist_ok= True)
        self.res_path = os.path.join(RES_PATH, self.EXP_NAME)
        os.makedirs(self.res_path, exist_ok=True)
        
        self._load_view()

    def _plot_sample_pc(self, ax):
        _, verts, _ = load_test_mesh()
        verts = verts.cpu().numpy()
        test_sample = verts

        X = test_sample[:,0]
        Y = test_sample[:,1]
        Z = test_sample[:,2]

        mins = np.min(test_sample, axis=0)
        maxs = np.max(test_sample, axis=0)


        rand_ind = random.sample(range(X.shape[0]), n_points)
        X = X[rand_ind]
        Y = Y[rand_ind]
        Z = Z[rand_ind]


        self.min_points = np.array([mins[i] if mins[i] < self.min_points[i] else self.min_points[i] for i in range(3)])
        self.max_points = np.array([maxs[i] if maxs[i] > self.max_points[i] else self.max_points[i] for i in range(3)])

        ax.scatter(X,Y,Z, marker='.', color='orange')



    def make_camera(self, theta=10, is_torch3d = False, H=None, W=None):
        '''
        input:
        - theta : Fov (degree) (camera intrinsic)
        - H,W -> H/W of the model
        return:
        Poly3DCollections    
        '''
        if isinstance(H, type(None)):
            H = self.H * CAMERA_SIZE
        if isinstance(W, type(None)):
            W = self.W * CAMERA_SIZE
        
        longer_side = W if W > H else H
        
        f = (longer_side / 2.) * (1/math.tan(theta/2*math.pi/180))
        if is_torch3d:
            f = f
        else:
            f = -f
        a1 = np.array([W/2., H/2., f])
        a2 = np.array([W/2., -H/2., f])
        a3 = np.array([-W/2., -H/2., f])
        a4 = np.array([-W/2., H/2., f])
        b = np.array([0,0,0])
        
        if is_torch3d:
            axis = [np.array([0,0,line_length]), b]
        else:
            axis = [np.array([0,0,-line_length]), b]
        
        verts = [
            [a1,a2,b],
            [a2,a3,b],
            [a3,a4,b],
            [a4,a1,b],
            [a1,a2,a3,a4],
        ]

        return verts, axis


    def plot_trains(self, use_axis = False):

        for epoch in tqdm(range(self.tot_epch)):
            self.plot_n_camera(epoch, use_axis)

        os.makedirs(GIF_PATH, exist_ok=True)
        with imageio.get_writer(GIF_PATH+'/'+self.EXP_NAME+'_res.gif', mode='I') as writer:
            for filename in tqdm(self.filenames):
                image = imageio.imread(filename)
                writer.append_data(image)



    def plot_n_camera(self, i, use_axis = False):
        '''
        inds :list of index of the camera, we want to plot
        '''
        Rs = self.Rs
        Ts = self.Ts

        for ind in range(len(self.Rs)):
            R = Rs[ind]
            T = Ts[ind]

            if ind == 0:
                ax = self.plot_single_camera(
                    R = R,
                    T = T,
                    use_axis = use_axis,
                    edgecolors=COLORS[ind]
                )
            else:
                self.plot_single_camera(
                    ax = ax,
                    R = R,
                    T = T,
                    use_axis = use_axis,
                    edgecolors=COLORS[ind%7]
                )
        
        #print("adding single original pc")
        self._plot_sample_pc(ax)



        diff = (self.max_points - self.min_points)
        Tmin = self.min_points - diff * 0.3
        Tmax = self.max_points + diff * 0.3


        xymin = np.min(Tmin[0:2])
        xymax = np.max(Tmax[0:2])

        ax.set_xlim([xymin, xymax])
        ax.set_ylim([xymin, xymax])
        ax.set_zlim([Tmin[2], Tmax[2]])

        #ax.axes.xaxis.set_ticks([])
        #ax.axes.yaxis.set_ticks([])
        #ax.axes.zaxis.set_ticks([])
        #ax.axes.xaxis.set_ticklabels([])
        #ax.axes.yaxis.set_ticklabels([])
        #ax.axes.zaxis.set_ticklabels([])


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        

        
        ax.view_init(10, ANGLES_theta[i])
        fname = self.EXP_NAME + '_1_' + str(i) + '.png'
        fname = os.path.join(self.res_path, fname)
        plt.savefig(fname, transparent = False, dpi=DPI)
        self.filenames.append(fname)

        


        
    def plot_single_camera(self, ax=None, R=None, T=None, edgecolors=None, axis_color=None, use_axis=False):
        '''
        return plot
        '''

        if isinstance(ax, type(None)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        if isinstance(R, type(None)):
            R = np.eye(3, dtype=np.float32)
            
        if isinstance(T, type(None)):
            T = np.zeros(3, dtype=np.float32)

        if isinstance(edgecolors, type(None)):
            edgecolors = self.camera_selected_color

        if isinstance(axis_color, type(None)):
            axis_color = self.camera_axis

        modified_verts = []
        for inst in self.verts:
            new_inst = [(np.matmul(R, point) + T + self.z_c)*SCALER for point in inst]
            modified_verts.append(new_inst)

        if use_axis:
            line = [(np.matmul(R, point) + T + self.z_c)*SCALER for point in self.axis]

        v = np.matmul(R,self.axis[1]) + T + self.z_c
        vx = v[0]
        vy = v[1]
        vz = v[2]

        if vx > self.max_points[0]:
            self.max_points[0] = vx
        elif vx < self.min_points[0]:
            self.min_points[0] = vx

        if vy > self.max_points[1]:
            self.max_points[1] = vy
        elif vy < self.min_points[1]:
            self.min_points[1] = vy

        if vz > self.max_points[2]:
            self.max_points[2] = vz
        elif vz < self.min_points[2]:
            self.min_points[2] = vz

        ax.add_collection3d(Poly3DCollection(
            modified_verts, 
            facecolors = (1., 1., 1., 0.),
            linewidth = 0.1,
            edgecolors=edgecolors,
            alpha=.25))
        '''
        ax.add_collection3d(Poly3DCollection(
            modified_verts[-1:], 
            facecolors = None,
            linewidth = 0.1,
            edgecolors=edgecolors,
            alpha=.25))
        '''

        if use_axis:
            ax.add_collection3d(Poly3DCollection(
                [line], 
                facecolors = None,
                linewidth = 0.5,
                edgecolors=axis_color,
                alpha=.25))

        return ax


    def _load_view(self):
        for j, c2w in enumerate(self.c2ws):
            R = c2w[0:3, 0:3]
            T = c2w[:, 3]

            R=-R

            self.Rs.append(R)
            self.Ts.append(T)


def plot_verts(self, res_path, verts, i):
    os.makedirs(res_path, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _, verts, _ = load_test_mesh()
    verts = verts.cpu().numpy()
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

    ax.view_init(10, 10)
    fname = 'trans_volume_' + str(i) + '.png'
    fname = os.path.join(res_path, fname)
    plt.savefig(fname, transparent = False, dpi=DPI)

def get_rotation_matrix(tx, ty, tz):
    m_x = np.zeros((len(tx), 3, 3))
    m_y = np.zeros((len(tx), 3, 3))
    m_z = np.zeros((len(tx), 3, 3))

    m_x[:, 1, 1], m_x[:, 1, 2] = np.cos(tx), np.sin(-tx)
    m_x[:, 2, 1], m_x[:, 2, 2] = np.sin(tx), np.cos(tx)
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = np.cos(ty), np.sin(ty)
    m_y[:, 2, 0], m_y[:, 2, 2] = np.sin(-ty), np.cos(ty)
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = np.cos(tz), np.sin(-tz)
    m_z[:, 1, 0], m_z[:, 1, 1] = np.sin(tz), np.cos(tz)
    m_z[:, 2, 2] = 1

    return np.matmul(m_z, np.matmul(m_y, m_x))



if __name__ == '__main__':
    EXP_NAME = 'test_plot'
    if not USE_AXIS:
        EXP_NAME = EXP_NAME + 'wo_line'

    if TORCH3D_CAMERA:
        EXP_NAME = EXP_NAME + '_torch3d'

    if USE_GUIDE:
        EXP_NAME = EXP_NAME + 'w_guide'

    CP = CamerPlot(cpath = '/home/inhee/VCL/insect_recon/frankmocap/demo/renderer/camera_path_longer.json', EXP_NAME=EXP_NAME)
    CP.plot_trains(use_axis=USE_AXIS)