from bigfile import BigFile
import numpy as np
from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera
import os, sys

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

# -------------------------------------------------------------------

snap = sys.argv[1]
snap = str(snap)

time = np.loadtxt('/home1/06431/yueyingn/scratch/asterix/cut-batch/Snapshots.txt')
sn = np.float32(snap)
aa = time[:,1][np.where(sn==time[:,0])[0][0]]
zz = (1/aa)-1

info1 = "a = %.2f; "%aa
info2 = "z = %.2f"%zz

center = np.float32(np.array([11492,152985,228942]))
info = info1 + info2

outprefix = '/home1/06431/yueyingn/scratch/asterix/plot-batch/gas-40Mpc/CUT-'+snap
ofilename = outprefix + '-40MPCh'

data_root = '/home1/06431/yueyingn/scratch/asterix/cut-batch/'
prefix = 'CUT-PART-' + snap
filename = prefix + '-%06.0f-%06.0f-%06.0f-40MPCh' % tuple([float(x) for x in center])
file = data_root + filename

# -------------------------------------------------------------------
part = BigFile(file)

Lbox = 250000.
lgt = 20000
slab_width=10000.

fac = 121.14740013761634  # GADGET unit Protonmass / Bolztman const
GAMMA = 5 / 3.
Xh=0.76

def cut(part,lgt,cpos,boxsize,slab_width):
    ppos = np.float32(part.open('0/Position')[:])
    print ("pos ave:",np.average(ppos,axis=0))
    ppos -= cpos
    ppos[ppos < -boxsize/2] += boxsize
    ppos[ppos > boxsize/2] -= boxsize

    mask = np.abs(ppos[:,0]) < lgt
    mask &= np.abs(ppos[:,1]) < lgt
    mask &= np.abs(ppos[:,2]) < slab_width

    return mask,ppos[mask]

mask,gaspos= cut(part,lgt,center,Lbox,slab_width)
gasm = part.open('0/Mass')[:][mask]
print (len(gasm))
gasen = part.open('0/InternalEnergy')[:][mask]
ye = part.open('0/ElectronAbundance')[:][mask]
gastem = (GAMMA-1)*fac*gasen/(ye*Xh+(1-Xh)*0.25+Xh)
del gasen, ye
del mask
# print ("load done")

imsize = 5000
resolution = 2
ct = lgt

mpers = camera.ortho(-slab_width,slab_width,(-ct,ct,-ct,ct))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix
mmv = camera.lookat((0,0,slab_width),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
gas2d = camera.apply(camera.matrix(mpers,mmv),gaspos) # apply a camera matrix to data coordinates, return position in clip coordinate
gasdev = camera.todevice(gas2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate
gassl = np.ones(len(gasm))*resolution
channels = painter.paint(gasdev, gassl, [gasm, gasm*gastem], (imsize, imsize), np=8)
channels[1] /= channels[0]

vmin = np.log10(np.percentile(channels[0],1))
vmax = np.log10(np.percentile(channels[0],99))

#----------------------------------------------------------
Lb = 40000

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12,12))
ax = axes
img = color.CoolWarm(color.NL(channels[1], range=(4,6.8)), color.NL(channels[0], range=(vmin,vmax)))
ax.imshow(img, extent=[0,Lb,0,Lb])

xst = Lb*0.0
yst = Lb*0.975
lt = Lb*1
ax.annotate(s='',xy=(xst+lt,yst),xytext=(xst,yst),\
            arrowprops=dict(arrowstyle='<->,head_width=1',color='white',lw=3))

ax.text(Lb*0.5-Lb*0.08,yst-Lb*0.03,'40 cMpc/h',dict(color='white'),size=20)
ax.text(Lb*0.02,Lb*0.09,info,dict(color='white'),size=18)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(ofilename+'.png',dpi=80,bbox_inches="tight")
plt.close()

