from bigfile import BigFile
import numpy as np
from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera
from scipy.spatial import cKDTree as KDTree
import os, sys

fac = 121.14740013761634  # GADGET unit Protonmass / Bolztman const
GAMMA = 5 / 3.
Xh=0.76

cx,cy,cz = sys.argv[1:]
center = np.array([np.float32(cx),np.float32(cy),np.float32(cz)])

prefix = '/home1/06431/yueyingn/scratch/BT3/work/2-cut-more/'
fname = 'CUT-%06.0f-%06.0f-%06.0f-40MPCh' % tuple([float(x) for x in center])
file = prefix + fname
part = BigFile(file)

Lbox = 400000.
lgt = 2000
slab_width=1000.

def smooth(pos):
    tree = KDTree(pos)
    d, i = tree.query(pos,k=60)
    return d[:,-1].copy()

def cut(part,lgt,cpos,boxsize,slab_width):
    ppos = np.float32(part.open('1/Position')[:])
    ppos -= cpos
    ppos[ppos < -boxsize/2] += boxsize
    ppos[ppos > boxsize/2] -= boxsize

    mask = np.abs(ppos[:,0]) < lgt
    mask &= np.abs(ppos[:,1]) < lgt
    mask &= np.abs(ppos[:,2]) < slab_width

    return mask,ppos[mask]

mask,dmpos= cut(part,lgt,center,Lbox,slab_width)
del mask
print ("load done")

imsize = 5000
resolution = 5
ct = lgt
#sml = np.ones(len(dmpos))*resolution
sml = smooth(dmpos)
sml *= 0.6
sml[sml>50] = 50
weight = np.ones(len(dmpos))

mpers = camera.ortho(-slab_width,slab_width,(-ct,ct,-ct,ct))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix
mmv = camera.lookat((0,0,slab_width),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
dm2d = camera.apply(camera.matrix(mpers,mmv),dmpos) # apply a camera matrix to data coordinates, return position in clip coordinate
dmdev = camera.todevice(dm2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate
channels = painter.paint(dmdev, sml, [weight], (imsize, imsize), np=8)

ofile_dir = prefix.replace('2-cut-more','3-paint/dm-channel-4Mpc/')
if not os.path.exists(ofile_dir):
    os.makedirs(ofile_dir)
ofilename = ofile_dir + fname
np.save(ofilename,channels)




