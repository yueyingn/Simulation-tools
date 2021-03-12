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

theta = sys.argv[2]
theta = np.float32(theta)

time = np.loadtxt('/home1/06431/yueyingn/scratch/asterix/cut-batch/Snapshots.txt')
sn = np.float32(snap)
aa = time[:,1][np.where(sn==time[:,0])[0][0]]
zz = (1/aa)-1

info1 = "a = %.2f; "%aa
info2 = "z = %.2f"%zz

center = np.float32(np.array([11492,152985,228942]))
info = info2

outprefix = '/home1/06431/yueyingn/scratch/asterix/plot-batch/gas-4Mpc-3D/CUT-'+snap
lbl = '-4MPCh-theta=%.2f'%theta
ofilename = outprefix + lbl

data_root = '/home1/06431/yueyingn/scratch/asterix/cut-batch/'
prefix = 'CUT-PART-' + snap
filename = prefix + '-%06.0f-%06.0f-%06.0f-40MPCh' % tuple([float(x) for x in center])
file = data_root + filename

# -------------------------------------------------------------------
part = BigFile(file)

Lbox = 250000.
lgt = 10000

fac = 121.14740013761634  # GADGET unit Protonmass / Bolztman const
GAMMA = 5 / 3.
Xh=0.76

def cut(part,lgt,cpos,boxsize):
    ppos = np.float32(part.open('0/Position')[:])
    ppos -= cpos
    ppos[ppos < -boxsize/2] += boxsize
    ppos[ppos > boxsize/2] -= boxsize

    mask = np.abs(ppos[:,0]) < lgt
    mask &= np.abs(ppos[:,1]) < lgt
    mask &= np.abs(ppos[:,2]) < lgt

    return mask,ppos[mask]

mask,gaspos= cut(part,lgt,center,Lbox)
gasm = part.open('0/Mass')[:][mask]
# gassl = part.open('0/SmoothingLength')[:][mask]
# gassl /= 1.7
gasen = part.open('0/InternalEnergy')[:][mask]
ye = part.open('0/ElectronAbundance')[:][mask]
gastem = (GAMMA-1)*fac*gasen/(ye*Xh+(1-Xh)*0.25+Xh)
del gasen, ye
del mask
# print ("load done")

#----------------------------------------------------------
r=2
x,y = r*np.cos(theta),r*np.sin(theta)
z=0.5
ct_scale=1.6

imsize = 5000
resolution = 2
gassl = np.ones(len(gasm))*resolution

ct = lgt*ct_scale
mpers = camera.ortho(-ct,ct,(-ct,ct,-ct,ct))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix
mmv = camera.lookat((x*lgt,y*lgt,z*lgt),(0,0,0),(0,0,1)) # (position of camera, focal point, up direction)
gas2d = camera.apply(camera.matrix(mpers,mmv),gaspos) # apply a camera matrix to data coordinates, return position in clip coordinate
gasdev = camera.todevice(gas2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate
channels = painter.paint(gasdev, gassl, [gasm, gasm*gastem], (imsize, imsize), np=8)
channels[1] /= channels[0]

channels[0] = np.transpose(channels[0])
channels[1] = np.transpose(channels[1])

mask = channels[0]>0
vmin = np.log10(np.percentile(channels[0][mask],8))
vmax = np.log10(np.percentile(channels[0][mask],99))
print (vmin,vmax)
#----------------------------------------------------------

img = color.CoolWarm(color.NL(channels[1], range=(4,7)), color.NL(channels[0], range=(vmin,vmax)))

plt.figure(figsize=(12,12))
plt.imshow(img,origin='lower')

lgt = lgt*1.0
corner = np.array([[-lgt,-lgt,-lgt],
                   [-lgt,-lgt,+lgt],
                   [-lgt,+lgt,-lgt],
                   [-lgt,+lgt,+lgt],
                   [+lgt,-lgt,-lgt],
                   [+lgt,-lgt,+lgt],
                   [+lgt,+lgt,-lgt],
                   [+lgt,+lgt,+lgt]])

corner2d = camera.apply(camera.matrix(mpers,mmv),corner)
cornerdev = camera.todevice(corner2d, extent=(imsize, imsize)) 

if (0 <= theta < np.pi/2):
    dash_corner = cornerdev[0]
elif (np.pi/2 <= theta < np.pi):
    dash_corner = cornerdev[4]
elif (np.pi <= theta < 1.5*np.pi):
    dash_corner = cornerdev[6]
elif (1.5*np.pi <= theta < 2*np.pi):
    dash_corner = cornerdev[2]

for i in range(0,len(cornerdev)):
    p = cornerdev[i]
    rr = np.linalg.norm(corner-corner[i],axis=-1)
    msk = np.abs(rr-2*lgt)<0.01*lgt
    for o in cornerdev[msk]:
        x,y = np.array([p[0],o[0]]),np.array([p[1],o[1]])
        r2 = np.linalg.norm(o-dash_corner,axis=0)
        r3 = np.linalg.norm(p-dash_corner,axis=0)
        if (r2<1 or r3<1):
            plt.plot(x,y,c='grey',alpha=0.5,lw=1,linestyle='--')
        else:
            plt.plot(x,y,c='red',alpha=0.5,lw=2)

plt.xlim(0,imsize)
plt.ylim(0,imsize)
plt.text(0.03*imsize,0.05*imsize,info,dict(color='white'),size=18)
plt.xticks([])
plt.yticks([])

plt.savefig(ofilename+'.png',dpi=80,bbox_inches="tight")
plt.close()




