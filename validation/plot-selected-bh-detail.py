"""
This script plot bh detail history in bhdetail_file for some randomly selected BHs from pig_file

Example usage:

bhdetail_file='/home1/06431/yueyingn/scratch3/BH-detail-reduce/BH-Details-R282'
pig_file = '/home1/06431/yueyingn/scratch3/ASTRID-II/output/PIG_284'
output_dir = 'R282-plot'

python plot-selected-bh-detail.py --bhdetail-file "$bhdetail_file" --pig-file "$pig_file" --output-dir "$output_dir"


"""
import numpy as np
from bigfile import BigFile
import os, sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io

matplotlib.use('agg')

#--------------------------
parser = argparse.ArgumentParser(description='bhdetail-plot')
parser.add_argument('--bhdetail-file',required=True,type=str,help='path of BHdetail directory')
parser.add_argument('--output-dir',required=True,type=str,help='path of the output dir')
parser.add_argument('--pig-file',required=True,type=str,help='PIG file to select BHs')
args = parser.parse_args()


# selected fields to plpt
# --------------
fields = ['z','BHID','Mdot','BHMass','Density','Entropy','KineticFdbkEnergy','KEflag']
formats = ['d','q','d','d','d','d','d','i']


# Lx
# --------------
Msolar = 1.99*10**30           # unit: kg
c = 3e8                        # in m/s
eta = 0.1
yr = 365*24*3600               # in second
acrtolum = eta*c*c*(1e10/(980*10**6))*(Msolar/yr)*10**7
## X-ray luminosity for AGN
Lsolar = 3.839e33 # erg/s
L10 = 1e10*Lsolar

hh = BigFile(args.pig_file).open('Header').attrs['HubbleParam'][0]

def Mdot_to_Lx(bhmdot):    # from erg/s to Lx [2-10]keV band
    L = bhmdot*acrtolum
    k = 10.83*(L/L10)**0.28+6.08*(L/L10)**(-0.02)
    return (L/k)

def select_bhs(file):
    """
    randomly select 12 BHs in different mass bins
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
    searchID += list(np.random.choice(bhid[msk],3))
    msk = (bhm>1e8) & (bhm<1e9)
    searchID += list(np.random.choice(bhid[msk],3))
    msk = (bhm>1e7) & (bhm<1e8)
    searchID += list(np.random.choice(bhid[msk],2))
    msk = (bhm>1e6) & (bhm<1e7)
    searchID += list(np.random.choice(bhid[msk],3))
    searchID = np.unique(np.array(searchID))
    return searchID[:12]

def get_bh_info(bhdetail_files,searchIDs):
    searchIDs = np.array(searchIDs)
    
    bhd = BigFile(bhdetail_files)
    BHID = bhd.open('BHID')[:]
    idxs = np.isin(BHID,searchIDs)
    BHID = BHID[idxs]
    data = np.zeros(len(BHID),dtype={'names':tuple(fields),'formats':tuple(formats)})
    if len(BHID)<1:
        print ("Did not find any BHs in searchIDs")
    else:
        for block in fields:
            print ("loading ",block,flush=True)
            data[block] = bhd[block][:][idxs]
    return data

def plot_bhdetail(d):
    """
    d is structured array for one specific BH
    """
    f,axes = plt.subplots(4,1,figsize=(6,6),sharex=True)
    f.subplots_adjust(hspace=0,wspace=0.1)
    
    xlim=(np.min(d['z']),np.max(d['z']))
    
    ax = axes.flat[0]
    text = 'BHID = ' + str(d['BHID'][0])
    ax.set_title(text,fontsize=15)
    ax.scatter(d['z'],d['BHMass']*10**10/hh,s=0.1)
    ax.set_yscale('log')
    ax.set_ylabel(r'$M_{BH}$',fontsize=15)
    
    ax = axes.flat[1]
    yy = Mdot_to_Lx(d['Mdot'])
    ax.scatter(d['z'],yy,s=0.1)
    msk1 = d['KEflag']>0
    if len(d[msk1])>0:
        ax.scatter(d['z'][msk1],yy[msk1],s=0.1,c='red')
    msk2 = d['KEflag']==2
    if len(d[msk2])>0:
        ax.scatter(d['z'][msk2],yy[msk2],s=10,marker='*',c='orange')
    ax.set_yscale('log')
    ax.set_ylabel(r'Lx',fontsize=15)
    yup = np.max(yy)*2
    ylow = np.min(yy)*0.5
    ax.set_ylim(ylow,yup)
    
    ax = axes.flat[2]
    yy = d['Density']
    ax.scatter(d['z'],yy,s=0.1)
    ax.set_yscale('log')
    ax.set_ylabel(r'rho',fontsize=15)
    yup = np.max(yy)*2
    ylow = np.min(yy)*0.5
    ax.set_ylim(ylow,yup)
    
    ax = axes.flat[3]
    yy = d['Entropy']
    ax.scatter(d['z'],yy,s=0.1)
    ax.set_yscale('log')
    ax.set_ylabel(r'Entropy',fontsize=15)
    yup = np.max(yy)*2
    ylow = np.min(yy)*0.5
    ax.set_ylim(ylow,yup)
    ax.set_xlabel('z',fontsize=17)
    
    return f
    
def image_grid(imgs, rows, cols):
    """
    imgs: list of PIL Image objects
    """
    assert len(imgs) == rows*cols
    w, h = imgs[0].size 
    w += 5
    h += 5
    grid = Image.new('RGBA',(cols*w, rows*h),(255, 255, 255, 255))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# main
#-----------------------------    
searchIDs = select_bhs(args.pig_file)
datas = get_bh_info(args.bhdetail_file,searchIDs)

frames = []
for sID in np.unique(datas['BHID']):
    d = datas[datas['BHID']==sID]
    fig = plot_bhdetail(d)
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png',bbox_inches="tight",dpi=80)
    new_frame = Image.open(img_buf)
    frames.append(new_frame)
    
grid = image_grid(frames,rows=3,cols=4)
ofilename = args.output_dir + '/BH-details.png'
grid.save(ofilename)
#--------------------------------------------------
        
