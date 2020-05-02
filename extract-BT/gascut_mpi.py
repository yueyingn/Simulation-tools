import numpy as np
from bigfile import BigFile
import os, sys
from mpi4py import MPI


def main():
    
    part = BigFile('/home1/06431/yueyingn/scratch/BT3/PART_271')
    center = np.array([278083,27826,249177])
    crop = 20000 
    
    comm = MPI.COMM_WORLD
    
    prefix = 'CUT-'
    ofilename = prefix + '%06.0f-%06.0f-%06.0f-40MPCh' % tuple([float(x) for x in center])
    
    if comm.rank == 0:
        ofile = BigFile(ofilename, create=True)
        print('writing to ', ofilename)
        
    comm.barrier()
    
    ofile = BigFile(ofilename, create=True)
    
    #######################################################
    Ntot = 7040**3
    
    start = Ntot * comm.rank // comm.size
    end = Ntot * (comm.rank  + 1) // comm.size     
    
    Nchunks = 800 # split the particles into Nchunks of groups for reading position    
    chunksize = int(Ntot/Nchunks) # each group has chunksize of particles
    
    gaspos = []
    gasm = []
    u = []
    ne = []
    sml = []    
    
    for i in range(start,end,chunksize):
        sl = slice(i,i+chunksize)
        
        pos = np.float32(part['0/Position'][sl])
        pos -= center

        pos[pos < - 200000] += 400000
        pos[pos > 200000] -= 400000
        
        mask = np.abs(pos[:,0])  < crop
        mask &= np.abs(pos[:,1]) < crop
        mask &= np.abs(pos[:,2]) < crop
        
        ld = mask.sum()
        
        if ld>0:
            gaspos.append(part['0/Position'][sl][mask])
            u.append(part['0/InternalEnergy'][sl][mask])
            ne.append(part['0/ElectronAbundance'][sl][mask])
            sml.append(part['0/SmoothingLength'][sl][mask])
            gasm.append(part['0/Mass'][sl][mask])
    
    if len(gasm)>0:
        gaspos = np.array(np.concatenate(gaspos))
        gasm = np.array(np.concatenate(gasm))
        u = np.array(np.concatenate(u))
        ne = np.array(np.concatenate(ne))
        sml = np.array(np.concatenate(sml))    
    
    comm.barrier()
    
    ###############################################
    olength = len(gasm)
    ooffset = sum(comm.allgather(olength)[:comm.rank])
    oN = comm.allreduce(olength)
    
    blocks = ['0/Position', '0/Mass', '0/InternalEnergy', '0/ElectronAbundance','0/SmoothingLength']
    data = [gaspos,gasm,u,ne,sml]
    
    for i,block in enumerate(blocks):
        iblock = part[block]
        if comm.rank == 0:
            print(block, 'size', oN)
            ofile.create(block, size=oN, dtype=iblock.dtype, Nfile=4)
        comm.barrier()
        oblock = ofile[block]
        oblock.write(ooffset, data[i])
        

if __name__ == "__main__":
    main()
