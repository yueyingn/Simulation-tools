"""
This script plot BHMF, BHXLF, GSMF from z=5 to the latest redshift 

Example usage:

pig_file = '/home1/06431/yueyingn/scratch3/ASTRID-II/output/PIG_284'
output_dir = 'R282-plot'

python plot-stats.py --pig-file "$pig_file" --output-dir "$output_dir"

"""

import numpy as np
from bigfile import BigFile
import argparse
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io
matplotlib.use('agg')

#------------
plt.rcParams['axes.linewidth'] = 1.8 #set the value globally
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 18
plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
plt.rc('ytick', labelsize=17) 

parser = argparse.ArgumentParser(description='bhdetail-plot')
parser.add_argument('--pig-file',required=True,type=str,help='PIG file to plot summary stats')
parser.add_argument('--output-dir',required=True,type=str,help='path of the output dir')
args = parser.parse_args()

## Mass function
## --------------------------------------------------------

def gethm(file):
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    return (BigFile(file).open('FOFGroups/Mass')[:])*10**10/hh

def getstr(file):
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    return np.transpose(BigFile(file).open('FOFGroups/MassByType')[:])[4]*10**10/hh

def getbhm(file):
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    bhm = (BigFile(file).open('5/BlackholeMass')[:])*10**10/hh
    return bhm

def mf(m,Lbox,hh,nbins=30):
    mbin = np.logspace(3.5,14,nbins)
    binmid = np.log10(mbin)[:-1]+np.diff(np.log10(mbin))/2
    mhis = np.histogram(m,mbin)
    mask = mhis[0]>0
    Volumndlog = np.diff(np.log10(mbin))*(Lbox/hh)**3
    yy = mhis[0]/Volumndlog
    err = yy[mask]/np.sqrt(mhis[0][mask])
    y1 = np.log10(yy[mask]+err)
    y2 = yy[mask]-err
    y2[y2<=0] = 1e-50
    return (binmid[mask]),np.log10(yy[mask]), y1, np.log10(y2)

def getbmf(file,nbins=30):
    bhm = getbhm(file)
    Lbox = BigFile(file).open('Header').attrs['BoxSize'][0]/1000 # to Mpc/h
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    bmf = mf(bhm,Lbox,hh,nbins)
    return bmf

def getgsmf(file,nbins=30):
    sm = getstr(file) 
    Lbox = BigFile(file).open('Header').attrs['BoxSize'][0]/1000 # to Mpc/h
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    rsl = 2*min(sm[sm>0])
    smf = mf(sm[sm>rsl],Lbox,hh,nbins)
    return smf


## BH Luminosity
## --------------------------------------------------------
## convert BH accretion rate from 1e10Msun/980Myr to ergs/s

Msolar = 1.99*10**30           # unit: kg
c = 3e8                        # in m/s
eta = 0.1
yr = 365*24*3600               # in second
acrtolum = eta*c*c*(1e10/(980*10**6))*(Msolar/yr)*10**7

## UV magnitude for AGN
fb = 10.2
deltab = -0.48
mub = 6.74*10**14

def Lbolo_to_Muv(L):   # from erg/s to Muv
    Muv = -2.5 * np.log10(L/(1e7*fb*mub))+34.1+deltab 
    return Muv

def Muv_to_Lbolo(Muv):
    ep = (Muv-deltab-34.1)/(-2.5)
    L = (1e7*fb*mub)*10**ep
    return L

## X-ray luminosity for AGN
Lsolar = 3.839e33 # erg/s
L10 = 1e10*Lsolar

def Lbolo_to_Lx(L):    # from erg/s to Lx [2-10]keV band
    k = 10.83*(L/L10)**0.28+6.08*(L/L10)**(-0.02)
    return (L/k)

def getbhlum(pig):
    bhmdot = BigFile(pig).open('5/BlackholeAccretionRate')[:]
    agn_lum = bhmdot*acrtolum 
    muv_agn = Lbolo_to_Muv(agn_lum)
    Lx = Lbolo_to_Lx(agn_lum)
    return muv_agn,Lx,agn_lum 

def qlf(Lbol,Lbox,hh,nn=18):  
    Lbin = np.linspace(40,48,nn)
    binmid = Lbin[:-1]+np.diff(Lbin)/2
    hist = np.histogram(np.log10(Lbol),Lbin)
    mask = hist[0]>0
    volume = np.diff(Lbin)*(Lbox/hh)**3
    yy = hist[0]/volume
    err = yy[mask]/np.sqrt(hist[0][mask])
    y1 = yy[mask]+err
    y2 = yy[mask]-err
    return binmid[mask],yy[mask],y1,y2,mask

def getxlf(file,nbins):
    Lbox = BigFile(file).open('Header').attrs['BoxSize'][0]/1000 # to Mpc/h
    hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
    muv_agn,Lx,Lbol = getbhlum(file)
    lfm = qlf(Lx,Lbox,hh,nbins)
    return lfm

## --------------------------------------------------------


pig_files = ['/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_107', # z = 5
             '/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_147', # z = 4
             '/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_214', # z = 3 
             '/home1/06431/yueyingn/scratch3/asterix/PIG/PIG_272', # z = 2.5
            ]

file = args.pig_file
pig_files.append(file)
redshift = 1./BigFile(file).open('Header').attrs['Time']-1
hh = BigFile(file).open('Header').attrs['HubbleParam'][0]
print ("new pig file at z=%.2f"%redshift[0])
label_new = 'z = %.2f'%redshift[0]

f = pig_files[0] # z = 5
bmf_z5 = getbmf(f,nbins=40)
xlf_z5 = getxlf(f,nbins=20)
gsmf_z5 = getgsmf(f,nbins=30)

f = pig_files[1] # z = 4
bmf_z4 = getbmf(f,nbins=40)
xlf_z4 = getxlf(f,nbins=20)
gsmf_z4 = getgsmf(f,nbins=30)

f = pig_files[2] # z = 3
bmf_z3 = getbmf(f,nbins=40)
xlf_z3 = getxlf(f,nbins=20)
gsmf_z3 = getgsmf(f,nbins=30)

f = pig_files[3] # z = 2.5
bmf_z25 = getbmf(f,nbins=40)
xlf_z25 = getxlf(f,nbins=20)
gsmf_z25 = getgsmf(f,nbins=30)

f = pig_files[4] # new file
bmf_z = getbmf(f,nbins=40)
xlf_z = getxlf(f,nbins=20)
gsmf_z = getgsmf(f,nbins=30)

labels = ['z=5','z=4','z=3','z=2.5', label_new]
colors = ['#45bdad', '#3497a9', '#3671a0', '#40498e', '#382a54']

# plot bhmf
# -------------------------------
def plot_bhmf():
    f,ax = plt.subplots(1,1,figsize=(6,5))

    xlim = (4.6,10.5)
    ylim = (-7.5,0)

    mbh_min = np.log10(3e4/hh)
    mbh_max = np.log10(3e5/hh)

    ax.axvline(x=mbh_min,c='black',linestyle='--',lw=1,alpha=0.8)
    ax.axvline(x=mbh_max,c='black',linestyle='--',lw=1,alpha=0.8)

    lfms = [bmf_z5,bmf_z4,bmf_z3,bmf_z25,bmf_z]
    for i in range(0,5):
        lfm = lfms[i]
        d,=ax.plot(lfm[0],lfm[1],lw=2,c=colors[i],label=labels[i])
        ax.fill_between(lfm[0],lfm[2],lfm[3],facecolor=colors[i],alpha=0.1)

    ax.legend(fontsize=15)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid('True',linestyle=':', linewidth=0.7)
    ax.set_xticks([5,6,7,8,9,10])
    ax.set_yticks([-7,-6,-5,-4,-3,-2,-1,0])
    ax.set_xlabel(r'$\mathrm{log}_{10} [M_{\rm BH}/M_{\odot}]$')
    ax.set_ylabel(r'$\mathrm{log}_{10} \phi/[\mathrm{dex}^{-1} \mathrm{Mpc}^{-3}]$')
    ax.set_title('BHMF',fontsize=17)
    
    img_buf = io.BytesIO()
    f.savefig(img_buf, format='png',bbox_inches='tight',dpi=80)
    img = Image.open(img_buf)
    return img
    
    
# plot bh xlf
# -------------------------------
def plot_bhxlf():    
    f,ax = plt.subplots(1,1,figsize=(6,5))

    xlim = (42.6,45.6)
    ylim = (2e-8,4e-3)

    ax.set_yscale('log')   
    lfms = [xlf_z5,xlf_z4,xlf_z3,xlf_z25,xlf_z]
    for i in range(0,5):
        lfm = lfms[i]
        d,=ax.plot(lfm[0],lfm[1],lw=2,c=colors[i],label=labels[i])
        ax.fill_between(lfm[0],lfm[2],lfm[3],facecolor=colors[i],alpha=0.1)

    ax.legend(fontsize=15)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$\log_{10}$($L_X$ [ergs/s])') 
    ax.set_ylabel(r'$\mathrm{log}_{10} \phi/[\mathrm{dex}^{-1} \mathrm{Mpc}^{-3}]$')
    ax.set_title('XLF',fontsize=17)
    ax.grid('True',linestyle=':', linewidth=0.7)
    
    img_buf = io.BytesIO()
    f.savefig(img_buf, format='png',bbox_inches='tight',dpi=80)
    img = Image.open(img_buf)
    return img
    
    
# plot GSMF
# -------------------------------
def plot_gsmf():
    f,ax = plt.subplots(1,1,figsize=(6,5))
    
    xlim = (7,13)
    ylim = (-8,0)

    lfms = [gsmf_z5,gsmf_z4,gsmf_z3,gsmf_z25,gsmf_z]
    for i in range(0,5):
        lfm = lfms[i]
        d,=ax.plot(lfm[0],lfm[1],lw=2,c=colors[i],label=labels[i])
        ax.fill_between(lfm[0],lfm[2],lfm[3],facecolor=colors[i],alpha=0.1)
    
    ax.legend(fontsize=15)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$\mathrm{log}_{10} [M_{*}/M_{\odot}]$')
    ax.set_ylabel(r'$\mathrm{log}_{10} \phi/[\mathrm{dex}^{-1} \mathrm{Mpc}^{-3}]$')
    ax.set_title('GSMF (FOF)',fontsize=17)
    ax.grid('True',linestyle=':', linewidth=0.7)
    
    img_buf = io.BytesIO()
    f.savefig(img_buf, format='png',bbox_inches='tight',dpi=80)
    img = Image.open(img_buf)
    return img

# ------------------------------- 
def image_grid(imgs, rows, cols):
    """
    imgs: list of PIL Image objects
    """
    assert len(imgs) == rows*cols
    w, h = imgs[0].size 
    w += 20
    h += 20
    grid = Image.new('RGBA',(cols*w, rows*h),(255, 255, 255, 255))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
  
frames = []
frames.append(plot_bhmf())
frames.append(plot_bhxlf())
frames.append(plot_gsmf())

ofilename = args.output_dir + '/global-stats.png'
grid = image_grid(frames,rows=1,cols=3)
grid.save(ofilename)

