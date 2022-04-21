"""
This script plot BH host galaxies, and BH trajectories for some randomly selected massive BHs from pig_file

Example usage:

pig_file = '/home1/06431/yueyingn/scratch3/ASTRID-II/output/PIG_284'
bhdetail_file='/home1/06431/yueyingn/scratch3/BH-detail-reduce/BH-Details-R282'
output_dir = 'R282-plot'

python plot-stellar-bhs-traj.py --pig-file "$pig_file" --bhdetail-file "$bhdetail_file" --output-dir "$output_dir"
"""


import numpy as np
from bigfile import BigFile
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os,sys
from plotgalaxy import *
import argparse
import matplotlib
matplotlib.use('agg')

plt.rcParams['axes.linewidth'] = 1.8 #set the value globally
plt.rcParams["font.family"] = "serif"

#--------------------------
parser = argparse.ArgumentParser(description='BH-host-galaxy-plot')
parser.add_argument('--output-dir',required=True,type=str,help='path of the output dir')
parser.add_argument('--bhdetail-file',required=True,type=str,help='path of BHdetail directory')
parser.add_argument('--pig-file',required=True,type=str,help='PIG file to select BHs')
args = parser.parse_args()


def select_bhs(file):
    """
    randomly select 10 massive BHs
    return array of BHID
    """
    searchID = []
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    bhm = (BigFile(file).open('5/BlackholeMass')[:])*10**10/hh
    bhid = BigFile(file).open('5/ID')[:]
    bhix = np.argsort(-bhm)
    bhm = bhm[bhix]
    bhid = bhid[bhix]

    searchID += list(bhid[0:2])
    msk = bhm>1e9
    searchID += list(np.random.choice(bhid[msk],7))
    msk = (bhm>1e8) & (bhm<1e9)
    searchID += list(np.random.choice(bhid[msk],4))
    return np.unique(np.array(searchID))[0:10]

def get_ngb_bhs(file,searchID,crop):
    """
    return ngb BH information within crop regions around BH searchID
    """
    pig = BigFile(file)
    hh = pig.open('Header').attrs['HubbleParam'][0]
    bhIDs = pig.open('5/ID')[0:1000000]
    bhidx = np.where(searchID==bhIDs)[0][0]
    grpidx = pig.open('5/GroupID')[bhidx]-1
    print ("groupID:",grpidx)
    center = pig.open('5/Position')[bhidx]
    
    Length = pig.open('FOFGroups/LengthByType')[0:1000000]
    OffsetByType = np.cumsum(Length,axis=0)
    a1 = np.array([[0,0,0,0,0,0]],dtype=np.uint64)
    OffsetByType = np.append(a1,OffsetByType,axis=0)
    
    bhoff = OffsetByType[grpidx:grpidx+2][:,5]
    pos5 =  pig.open('5/Position')[bhoff[0]:bhoff[1]]

    ppos = pos5 - center
    mask = np.abs(ppos[:,0]) < crop
    mask &= np.abs(ppos[:,1]) < crop
    mask &= np.abs(ppos[:,2]) < crop
    ngbID = pig.open('5/ID')[bhoff[0]:bhoff[1]][mask]
    ngbbhm = pig.open('5/BlackholeMass')[bhoff[0]:bhoff[1]][mask]*10**10/hh
    ngbpos = pos5[mask]
    return ngbID,ngbbhm,ngbpos,center

def get_bh_trajs(bhdetail_files,searchIDs):
    """
    get BH trajectory from BHdetail files
    searchIDs: array of query BHs
    """
    searchIDs = np.array(searchIDs)
    
    fields = ['z','BHID','BHpos']
    formats = ['d','q','3d']

    bhd = BigFile(bhdetail_files)
    BHID = bhd.open('BHID')[:]
    idxs = np.isin(BHID,searchIDs)
    BHID = BHID[idxs]
    data = np.zeros(len(BHID),dtype={'names':tuple(fields),'formats':tuple(formats)})
    if len(BHID)<1:
        print ("Did not find any BHs in searchIDs")
    else:
        for block in fields:
            data[block] = bhd[block][:][idxs]
    return data
            
def plot_host_galaxy_bh_traj(ax,file,searchID,bhdetail_files,orientation='xy',plotBH='False',
                             Lb=200,vmin0=None,vmax0=None):
    """
    plot in background the stellar field centered at BH = searchID
    file: pig file
    plotBH true: plot BHs and their traj found in bhdetail files
    """
    crop = Lb*0.5
    channels,mstar = extract_host_star(file,searchID,crop=crop,orientation=orientation)
    if vmax0 is None:
        vmax0 = np.log10(np.percentile(channels[0],99))
        print ("vmax0:",vmax0)
    if vmin0 is None:
        vmin0 = np.log10(np.percentile(channels[0],30))
        print("vmin0:",vmin0)
    img = starmap(color.NL(channels[1], range=(1.6,3.5)), color.NL(channels[0], range=(vmin0,vmax0)))
    ax.imshow(img,extent=[0,Lb,0,Lb],origin='upper')
    
    if plotBH == 'True':
        # ------- get ngb BHs ----------
        ngbID,ngbbhm,ngbpos,center = get_ngb_bhs(file,searchID,crop=crop)
        msk = ngbbhm>1e5
        bhdata = get_bh_trajs(bhdetail_files,ngbID[msk])
        ngbpos = ngbpos[msk]-center
        ngbbhm = ngbbhm[msk]

        t_color = 'yellow'
        # -------- plot BH and their traj ----------
        for sID in np.unique(bhdata['BHID']):
            d = bhdata[bhdata['BHID']==sID]
            d = d[np.argsort(d['z'])]
            dr = d['BHpos']-center
            if orientation is 'xy':
                x,y = Lb/2+ngbpos[:,0],Lb/2+ngbpos[:,1]
                ax.plot(Lb/2+dr[:,0],Lb/2+dr[:,1],lw=0.7,c=t_color,alpha=0.8)
            if orientation is 'xz':
                x,y = Lb/2+ngbpos[:,0],Lb/2-ngbpos[:,2]
                ax.plot(Lb/2+dr[:,0],Lb/2-dr[:,2],lw=0.7,c=t_color,alpha=0.8)
            if orientation is 'yz':
                x,y = Lb/2+ngbpos[:,1],Lb/2+ngbpos[:,2]
                ax.plot(Lb/2+dr[:,1],Lb/2+dr[:,2],lw=0.7,c=t_color,alpha=0.8)

        ax.scatter(x, y, marker='x',c='red',s=0.5*ngbbhm**(0.25),alpha=0.8,lw=0.7)
    
    # ------ scale and legend -------
    tcolor="white"
    info = r'$M_{*}$ = %.1e$M_{\odot}$'%mstar
    ax.text(0.05,0.05,info,dict(color=tcolor),size=15,transform=ax.transAxes)
    ax.text(0.05,0.9,'Lbox = %d ckpc/h'%Lb,dict(color='white'),size=15,transform=ax.transAxes)
    if plotBH == 'False':
        ax.text(0.05,0.12,'BHID='+str(searchID),dict(color=tcolor),size=13,transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0,Lb)
    ax.set_ylim(0,Lb)


# main
#-----------------------------    
searchIDs = select_bhs(args.pig_file)
file = args.pig_file
bhdetail_files = args.bhdetail_file

# plot 40ckpc view of galaxy
f,axes = plt.subplots(2,5,figsize=(20,8))
f.subplots_adjust(hspace=0.05,wspace=0.05)
for i in range(0,10):
    ax = axes.flat[i]
    plot_host_galaxy_bh_traj(ax,file,searchIDs[i],bhdetail_files,Lb=40,vmin0=-5,vmax0=-3)
    
ofilename=args.output_dir+'/galaxy-40ckpc.png'
plt.savefig(ofilename,bbox_inches="tight",dpi=150)  


# plot 400ckpc view of stellar field and BH trajs
f,axes = plt.subplots(2,5,figsize=(20,8))
f.subplots_adjust(hspace=0.05,wspace=0.05)
for i in range(0,10):
    ax = axes.flat[i]
    plot_host_galaxy_bh_traj(ax,file,searchIDs[i],bhdetail_files,plotBH='True',Lb=400,vmin0=-6.8,vmax0=-4)

ofilename=args.output_dir+'/galaxy-BH-400ckpc.png'
plt.savefig(ofilename,bbox_inches="tight",dpi=150)  


    
    
