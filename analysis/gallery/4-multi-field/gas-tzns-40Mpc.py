from bigfile import BigFile
import numpy as np
from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera
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
lgt = 20000
slab_width=10000.


def cut(part,lgt,cpos,boxsize,slab_width):
    ppos = np.float32(part.open('0/Position')[:])
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
# gassl = part.open('0/SmoothingLength')[:][mask]
gasen = part.open('0/InternalEnergy')[:][mask]
ye = part.open('0/ElectronAbundance')[:][mask]
gastem = (GAMMA-1)*fac*gasen/(ye*Xh+(1-Xh)*0.25+Xh)
del gasen, ye

gassfr = part.open('0/StarFormationRate')[:][mask]
gasnf = part.open('0/H2Fraction')[:][mask]
gasz = part.open('0/Metallicity')[:][mask]
del mask
print ("load done")

imsize = 5000
resolution = 5
ct = lgt

mpers = camera.ortho(-slab_width,slab_width,(-ct,ct,-ct,ct))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix
mmv = camera.lookat((0,0,slab_width),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
gas2d = camera.apply(camera.matrix(mpers,mmv),gaspos) # apply a camera matrix to data coordinates, return position in clip coordinate
gasdev = camera.todevice(gas2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate
gassl = np.ones(len(gasm))*resolution

# temperature
channels = painter.paint(gasdev, gassl, [gasm, gasm*gastem], (imsize, imsize), np=8)
channels[1] /= channels[0]
ofile_dir = prefix.replace('2-cut-more','3-paint/temp-channel-40Mpc/')
if not os.path.exists(ofile_dir):
    os.makedirs(ofile_dir)
ofilename = ofile_dir + fname
np.save(ofilename,channels)

# metal
channels = painter.paint(gasdev, gassl, [gasm, gasm*gasz], (imsize, imsize), np=8)
channels[1] /= channels[0]
ofile_dir = prefix.replace('2-cut-more','3-paint/metal-channel-40Mpc/')
if not os.path.exists(ofile_dir):
    os.makedirs(ofile_dir)
ofilename = ofile_dir + fname
np.save(ofilename,channels)

# sfr
channels = painter.paint(gasdev, gassl, [gasm, gasm*gassfr], (imsize, imsize), np=8)
channels[1] /= channels[0]
ofile_dir = prefix.replace('2-cut-more','3-paint/sfr-channel-40Mpc/')
if not os.path.exists(ofile_dir):
    os.makedirs(ofile_dir)
ofilename = ofile_dir + fname
np.save(ofilename,channels)

# nh2
channels = painter.paint(gasdev, gassl, [gasm, gasm*gasnf], (imsize, imsize), np=8)
channels[1] /= channels[0]
ofile_dir = prefix.replace('2-cut-more','3-paint/nh2-channel-40Mpc/')
if not os.path.exists(ofile_dir):
    os.makedirs(ofile_dir)
ofilename = ofile_dir + fname
np.save(ofilename,channels)



