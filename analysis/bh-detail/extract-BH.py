import numpy as np
from bigfile import BigFile
import struct
import glob
import os, sys

#--------------------------------
snap = sys.argv[1]
snap = str(snap)

searchID = 173623152589
#searchID = 174288691088
bhdir = '/home1/06431/yueyingn/frontera5500/output/BlackholeDetails-R'+snap
bhdir = bhdir + '/*'
print ("Reading in ",bhdir)
ofilename = 'bhdetail-'+str(searchID)+'-'+snap
print ("ofilename:",ofilename)

hh = 0.6774

def getinfo_fulldf_Mdyn(BHdir,searchID,countall=False):  # need searchID 
    """
    collect all BH info around position criteria
    """
    BHIDs = []
    BHMass = []
    Mdots = []
    Densitys = []
    timebin = []
    encounter = []
    MinPos = []
    MinPot = []
    Entropys = []
    Gasvel = []
    ac_momentum = []
    acMass = []
    acBHMass = []
    
    Fdbk = []
    SPHID = []
    CountProgs = []
    SwallowID = []
    Swallowed = []
    
    BHpos = []
    BH_SurroundingDensity = []
    BH_SurroundingParticles = []
    BH_SurroundingVel = []
    BH_SurroundingRmsVel = []
    
    DFAccel = []
    DragAccel = []
    GravAccel = []   
    BHvel = []
    time = []
    
    Mtrack = []
    Mdyn = []

    def split(chunk):              
        BHIDs.append(struct.unpack("q",chunk[4:12]))
        BHMass.append(struct.unpack("d",chunk[12:20]))
        Mdots.append(struct.unpack("d",chunk[20:28]))
        Densitys.append(struct.unpack("d",chunk[28:36]))
        timebin.append(struct.unpack("i",chunk[36:40]))
        encounter.append(struct.unpack("i",chunk[40:44]))
        
        MinPos.append(struct.unpack("3d",chunk[44:68]))
        MinPot.append(struct.unpack("d",chunk[68:76]))
        Entropys.append(struct.unpack("d",chunk[76:84]))
        Gasvel.append(struct.unpack("3d",chunk[84:108]))
        
        ac_momentum.append(struct.unpack("3d",chunk[108:132]))
        
        acMass.append(struct.unpack("d",chunk[132:140]))
        acBHMass.append(struct.unpack("d",chunk[140:148]))    
        Fdbk.append(struct.unpack("d",chunk[148:156]))
        
        SPHID.append(struct.unpack("q",chunk[156:164]))
        SwallowID.append(struct.unpack("q",chunk[164:172]))
        
        CountProgs.append(struct.unpack("i",chunk[172:176]))
        Swallowed.append(struct.unpack("i",chunk[176:180]))
        
        #--------------------------------------------------
        BHpos.append(struct.unpack("3d",chunk[180:204]))
        BH_SurroundingDensity.append(struct.unpack("d",chunk[204:212]))
        BH_SurroundingParticles.append(struct.unpack("d",chunk[212:220]))
        BH_SurroundingVel.append(struct.unpack("3d",chunk[220:244]))
        BH_SurroundingRmsVel.append(struct.unpack("d",chunk[244:252]))
        
        DFAccel.append(struct.unpack("3d",chunk[252:276]))
        DragAccel.append(struct.unpack("3d",chunk[276:300]))
        GravAccel.append(struct.unpack("3d",chunk[300:324]))
        
        BHvel.append(struct.unpack("3d",chunk[324:348]))
        Mtrack.append(struct.unpack("d",chunk[348:356]))
        Mdyn.append(struct.unpack("d",chunk[356:364]))
        
        time.append(struct.unpack("d",chunk[364:372]))     
        
    chunk_size = 376
    
    for filename in sorted(glob.glob(BHdir)):
        f = open(filename,'rb')
        while True:
            buf = f.read(chunk_size)            
            if not buf:
                f.close()
                break
                
            if countall==False:    
                bid = struct.unpack("q",buf[4:12])
                if bid[0] == searchID:
                    split(buf) 
            else:
                sw = struct.unpack("i",buf[176:180])  
                if sw[0] == 0:
                    split(buf)            
       
    data = np.zeros(len(BHIDs), dtype={'names':('BHIDs','SwallowID','SPHID','BHMass','Mtrack','Mdyn','Mdot','Fdbk','Density',\
                'Entropy','acMass','acBHMass','MinPot','MinPos','CountProgs','timebin','encounter','Swallowed','z',\
                'BHpos','DFAccel','DragAccel','GravAccel','BHvel','srVel','srParticles','srDensity'),\
                'formats':('q','q','q','d','d','d','d','d','d','d','d','d','d','3d',\
                           'i','i','i','i','d','3d','3d','3d','3d','3d','3d','d','d')})
   
    if len(BHIDs)==0:
        print ("BH not found!")
        return
   
    data['BHIDs'] = np.concatenate(np.array(BHIDs))
    data['SPHID'] = np.concatenate(np.array(SPHID))
    data['SwallowID'] = np.concatenate(np.array(SwallowID))
    data['BHMass'] = np.concatenate(np.array(BHMass))*10**10/hh
    data['Mtrack'] = np.concatenate(np.array(Mtrack))*10**10/hh
    data['Mdyn'] = np.concatenate(np.array(Mdyn))*10**10/hh
    
    data['Mdot'] = np.concatenate(np.array(Mdots))
    data['Fdbk'] = np.concatenate(np.array(Fdbk))
    data['Density'] = np.concatenate(np.array(Densitys))
    data['Entropy'] = np.concatenate(np.array(Entropys))
    data['acMass'] = np.concatenate(np.array(acMass))
    data['acBHMass'] = np.concatenate(np.array(acBHMass))
    data['MinPot'] = np.concatenate(np.array(MinPot))
    data['MinPos'] = (np.array(MinPos))
    data['CountProgs'] = np.concatenate(np.array(CountProgs))
    data['timebin'] = np.concatenate(np.array(timebin))
    data['encounter'] = np.concatenate(np.array(encounter))
    data['Swallowed'] = np.concatenate(np.array(Swallowed))
    data['BHpos'] = (np.array(BHpos))
    data['DFAccel'] = np.array(DFAccel)
    data['DragAccel'] = (np.array(DragAccel))
    data['GravAccel'] = (np.array(GravAccel))
    data['srVel'] = np.array(BH_SurroundingVel)
    data['srParticles'] = np.concatenate(np.array(BH_SurroundingParticles))
    data['srDensity'] = np.concatenate(np.array(BH_SurroundingDensity))
    data['BHvel'] = (np.array(BHvel))
   
    time = np.concatenate(np.array(time))
    data['z'] = 1/time - 1
    
    return data

bhdata = getinfo_fulldf_Mdyn(bhdir,searchID,countall=False)
np.save(ofilename,bhdata)

