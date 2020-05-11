import numpy as np
from bigfile import BigFile
import os, sys


def getCubeRange(cpos,crop=10000):
    """cut the ID range within 10 Mpc from the cpos
    """
    xlim = np.array([cpos[0]-crop,cpos[0]+crop])
    ylim = np.array([cpos[1]-crop,cpos[1]+crop])
    zlim = np.array([cpos[2]-crop,cpos[2]+crop])
    
    r = (np.array([xlim,ylim,zlim]))
    return r

def overlap1D(x1,x2):
    """determine whether x1 [amin,amax] has overlap with x2 [bmin,bmax]
    """
    overlap = False
    if (x1[0]>x2[0]) & (x1[0]<x2[1]):
        overlap = True
    if (x1[1]>x2[0]) & (x1[1]<x2[1]):
        overlap = True
    return overlap

def overlap3D(r1,r2):
    """determine cube r1 has overlap with cube r2, 
       r in shape ([xmin,xmax],[ymin,ymax],[zmin,zmax])
    """
    
    f1 = overlap1D(r1[0],r2[0])
    f2 = overlap1D(r1[1],r2[1])
    f3 = overlap1D(r1[2],r2[2])
    f = f1&f2
    f &= f3
    
    return f


arg = int(sys.argv[1])

posrange = np.load('ID_range_800.npy')

dmcpos = np.load('BH500-pre-cpos.npy')
center = dmcpos[arg]
cube = getCubeRange(center)

print ("center = ",center)
print ("crop cube:",cube)

prefix = 'CUT-BH%d-'%arg
ofilename = prefix + '%06.0f-%06.0f-%06.0f-20MPCh' % tuple([float(x) for x in center])
print ("output file name:",ofilename,flush=True)


pig = BigFile('/home1/06431/yueyingn/scratch/BT1/PART_005_recover')

Nchunk  = len(posrange)
chunksize = int((7040**3)/Nchunk)

dmpos = []
dmID = []

crop = 10000
for i in range(0,Nchunk):   
    
    print (i,flush=True)
    f = overlap3D(cube,posrange[i])   
    
    if f == True:
        print ("cube overlapped",flush=True)
        start = int(i*chunksize)
        end = int((i+1)*chunksize)
        pos = np.float32(pig['1/Position'][start:end])
        pos -= center

        pos[pos < - 200000] += 400000
        pos[pos > 200000] -= 400000

        mask = np.abs(pos[:,0])  < crop
        mask &= np.abs(pos[:,1]) < crop
        mask &= np.abs(pos[:,2]) < crop
#        print (np.min(np.abs(pos[:,0])))

        ld = len(pos[mask])
        if ld>0:
            print ("collected %d dm particles"%ld,flush=True)            
            dmpos.append(pos[mask])
            
            d = pig.open('1/ID')[start:end]
            dmID.append(d[mask])


dmpos = np.concatenate(dmpos)
dmID = np.concatenate(dmID)

path = '/home1/06431/yueyingn/scratch/BT1/extract-BT/CUT-005/'
os.makedirs(path, exist_ok=True)

rkstr=BigFile(path + ofilename + '/1',create=1)

blockname='Position'
rkstr.create_from_array(blockname,dmpos)

blockname='ID'
rkstr.create_from_array(blockname,dmID)

