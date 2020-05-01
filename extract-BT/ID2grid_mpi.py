import numpy as np
from bigfile import BigFile
import os, sys
from mpi4py import MPI

# This file use ID to recover the grid-position of the particle, split the particles into Nchunks of groups, and return the 
# x,y,z range of the Nchunk groups. This boost the speed of SAMPLE-CUT.

def id2pos(pid,boxsize=400000,Ng=7040):
    """Assume `pid` starts from 0 and aligns with the Lagrangian lattice.
    """
    cellsize = boxsize / Ng
    z = pid % Ng
    y = pid // Ng % Ng
    x = pid // (Ng * Ng)
    grid = np.stack([x, y, z], axis=-1)
    
    return grid*cellsize

def getposrange(file,sl):    
    """
    Calculate the range of grid-position from a chunk of dmID
    sl is the slice(group_start,group_end)
    """
    pid = file['1/ID'][sl]-7040**3-1
    worksize = 1024*1024
    pmin = []
    pmax = []
    for i in range(0,len(pid),worksize):
        s = slice(i, i + worksize)
        prepos = id2pos(pid[s])
        pmin.append(np.min(prepos,axis=0))
        pmax.append(np.max(prepos,axis=0))
        
    pmin = np.min(np.array(pmin),axis=0)
    pmax = np.max(np.array(pmax),axis=0)
    return np.transpose(np.array([pmin,pmax]))


def main():
    
    comm = MPI.COMM_WORLD 
    
    pig = BigFile('/home1/06431/yueyingn/scratch/BT1/PART_005_recover')       
    
    Ntot = 7040**3
    
    start = Ntot * comm.rank // comm.size
    end = Ntot * (comm.rank  + 1) // comm.size    
    
    Nchunks = 800 # split the particles into Nchunks of groups     
    chunksize = int(Ntot/Nchunks) # each group has chunksize of particles
    
    pos_range = []
    
    for i in range(start,end,chunksize):
        sl = slice(i,i+chunksize)
        pos_range.append(getposrange(pig,sl))
        
    pos_range = np.array(pos_range)
    reps = np.shape(pos_range)
    
    sendbuf = pos_range
    recvbuf = None
    
    if comm.rank == 0:
        recvbuf = np.empty([comm.size,reps[0],reps[1],reps[2]])
    comm.Gather(sendbuf, recvbuf, root=0)
    
#     comm.barrier()
    
    if comm.rank == 0:
        recvbuf = np.concatenate(recvbuf,axis=0)
        np.save('ID_range_800',recvbuf)

    
if __name__ == "__main__":
    main()  
