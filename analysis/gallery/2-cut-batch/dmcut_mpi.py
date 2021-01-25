import numpy as np
from bigfile import BigFile
import os, sys
from mpi4py import MPI

snap = sys.argv[1]
snap = str(snap)
filename = "/scratch3/04808/tg841079/large-run/frontera5500/output/PART_"+snap
BoxSize = 250000 # kpc/h
center = np.float32(np.array([11492,152985,228942]))

def main():
    
    part = BigFile(filename)
    crop = 20000

    comm = MPI.COMM_WORLD

    prefix = 'CUT-PART-'+snap
    ofilename = prefix + '-%06.0f-%06.0f-%06.0f-40MPCh' % tuple([float(x) for x in center])

    if comm.rank == 0:
        ofile = BigFile(ofilename, create=True)
        print('writing to ', ofilename)

    comm.barrier()

    ofile = BigFile(ofilename, create=True)
    
    #######################################################
    Ntot = 5500**3
    
    start = Ntot * comm.rank // comm.size
    end = Ntot * (comm.rank  + 1) // comm.size     
    
    Nchunks = 800 # split the particles into Nchunks of groups for reading position    
    chunksize = int(Ntot/Nchunks) # each group has chunksize of particles
    
    dmpos = []
    
    for i in range(start,end,chunksize):
        sl = slice(i,i+chunksize)
        
        pos = np.float32(part['1/Position'][sl])
        pos -= center

        pos[pos < - BoxSize*0.5] += BoxSize
        pos[pos > BoxSize*0.5] -= BoxSize
        
        mask = np.abs(pos[:,0])  < crop
        mask &= np.abs(pos[:,1]) < crop
        mask &= np.abs(pos[:,2]) < crop
        
        ld = mask.sum()
        
        if ld>0:
            dmpos.append(part['1/Position'][sl][mask])
    
    if len(dmpos)>0:
        dmpos = np.array(np.concatenate(dmpos))
    
    comm.barrier()
    
    ###############################################
    olength = len(dmpos)
    ooffset = sum(comm.allgather(olength)[:comm.rank])
    oN = comm.allreduce(olength)
    
    blocks = ['1/Position']
    data = [dmpos]
    
    for i,block in enumerate(blocks):
        iblock = part[block]
        if comm.rank == 0:
            print(block, 'size', oN)
            ofile.create(block, size=oN, dtype=iblock.dtype, Nfile=16)
        comm.barrier()
        oblock = ofile[block]
        oblock.write(ooffset, np.array(data[i]))
        

if __name__ == "__main__":
    main()
