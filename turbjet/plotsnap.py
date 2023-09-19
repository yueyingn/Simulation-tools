import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,colorConverter
from matplotlib.collections import LineCollection
import h5py    # hdf5 format
from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera


""" customize color maps """

class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)
    
def make_colormap(cmap_name):
    y = np.linspace(0,1,100)
    COL = MplColorHelper(cmap_name, 0, 1)
    x = COL.get_rgb(y)
    return color.Colormap(x[:,0:3])


jetmap = make_colormap('cubehelix')
sfrmap = make_colormap('rainbow')


""" global units """
FloatType = np.float64
IntType = np.int32

PROTONMASS = FloatType(1.67262178e-24)
BOLTZMANN = FloatType(1.38065e-16)
GRAVITY = FloatType(6.6738e-8) # G in cgs

MYR_IN_S = (3.15576e13)
PC = FloatType(3.085678e+18) 
MSOLAR = FloatType(1.989e+33) 

mu = FloatType(0.6165) # mean molecular weight 
gamma = FloatType(5./3.) # single-atom


""" snippets """

def overlap(img1,img2,thr=0):
    """
    overlap img2 onto img1, img2 is transparent in the region of black
    """
    assert (img1.shape == img2.shape)
    
    c2 = np.sum(img2[:,:,0:3],axis=-1)
    img1_ = img1.copy()
    img2_ = img2.copy()
    img1_[c2>thr] = np.array([0,0,0,0])
    img2_[c2<=thr] = np.array([0,0,0,0])
    
    return (img1_ + img2_)

def blend_channel(colorRGBA1, colorRGBA2):
    """
    blend RGB channels by weight of alpha
    """
    assert (colorRGBA1.shape == colorRGBA2.shape)
    
    result = np.zeros_like(colorRGBA1)  
    result[:,:,3] = 255 - ((255 - colorRGBA1[:,:,3]) * (255 - colorRGBA2[:,:,3]) / 255) # alpha
#     print ("alpha:",np.average(result[:,:,3]))
    
    w1 = (255 - colorRGBA2[:,:,3])/255
    w2 = (colorRGBA2[:,:,3])/255
    result[:,:,0] = np.multiply(colorRGBA1[:,:,0],w1)  + np.multiply(colorRGBA2[:,:,0],w2)
    result[:,:,1] = np.multiply(colorRGBA1[:,:,1],w1)  + np.multiply(colorRGBA2[:,:,1],w2)
    result[:,:,2] = np.multiply(colorRGBA1[:,:,2],w1)  + np.multiply(colorRGBA2[:,:,2],w2)
    
    return result

def overlay(img1,img2,x):
    """
    x is the weight of the img2 pixel, will squash x by sigmoid, as the alpha of img2
    """
    c = np.uint(255*(1/(1+np.exp(-1*x))))
    img1_ = img1.copy()
    img2_ = img2.copy()

    img2_[:,:,3] = 0
    img2_[:,:,3] += np.uint8(c)

    img3 = blend_channel(img1_,img2_)
    
    return img3

def cut(part,center,lbox,slab_width,orientation):
    """
    take the entire snapshot and cut a slab from specified projection
    
    Parameters
    ----------
    part: h5py.File(file,'r')['PartType0']
    center: numpy array, center of the chunk
    lbox: box length
    slab_width: projection depth
    orientation: projection of 'xy','yz','xz'
    
    Returns
    -------
    mask and relative position to the center
    """
    ppos = np.float32(part['Coordinates'][:]) # non-periodic
    ppos -= center
    
    if orientation == 'xy':
        d1,d2,d3 = 0,1,2
    if orientation == 'yz':
        d1,d2,d3 = 1,2,0
    if orientation == 'xz':
        d1,d2,d3 = 0,2,1
    
    mask = np.abs(ppos[:,d1]) < 0.5*lbox
    mask &= np.abs(ppos[:,d2]) < 0.5*lbox
    mask &= np.abs(ppos[:,d3]) < 0.5*slab_width

    return mask,ppos[mask]

def pos2device(ppos,lbox,slab_width,imsize,orientation):
    """
    transfer cartesian position to device coordinates
    
    Parameters
    ----------
    ppos: (N,3), particle position centered at [0,0,0]
    lbox: box length
    slab_width: projection depth
    orientation: projection of 'xy','yz','xz'
    imsize: Npixel of the image
    
    Returns
    particle position in device coordinates
    """    
    lgt = 0.5*lbox
    mpers = camera.ortho(-slab_width,slab_width,(-lgt,lgt,-lgt,lgt))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix
    
    if orientation == 'xy':
        mmv = camera.lookat((0,0,-slab_width),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
    if orientation == 'yz':
        mmv = camera.lookat((slab_width,0,0),(0,0,0),(0,1,0)) # (position of camera, focal point, up direction)
    if orientation == 'xz':
        mmv = camera.lookat((0,slab_width,0),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
        
    pos2d = camera.apply(camera.matrix(mpers,mmv),ppos) # apply a camera matrix to data coordinates, return position in clip coordinate
    posdev = camera.todevice(pos2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate
    
    return posdev

def get_channel_gas_rhotemp(file,center,lbox,slab_width=None,
                    imsize=2000,smlfac=1.0,orientation='xy'):
    """
    get channels of the gas density and mass weighted temperature 
    """
    if slab_width is None:
        slab_width = lbox
    
    try:
        UnitVelocity = h5py.File(file,'r')['Header'].attrs['UnitVelocity_in_cm_per_s']
    except:
        UnitVelocity = 100000
    temp_to_u = (BOLTZMANN/PROTONMASS)/mu/(gamma-1)/UnitVelocity/UnitVelocity
    
    part = h5py.File(file,'r')['PartType0']
    mask,gaspos= cut(part,center,lbox,slab_width,orientation)
    print ("num gas:",len(gaspos))
    
    gasm = part['Masses'][:][mask]
    gast = part['InternalEnergy'][:][mask]/temp_to_u
    gasvol = gasm/part['Density'][:][mask]
    gassl = smlfac*(gasvol**(1/3))
        
    p_resolution = lbox/imsize
    psml = gassl/p_resolution
    psml[psml<1]=1
    
    #---------------------------------------    
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    channels = painter.paint(gasdev, psml, [gasm, gasm*gast], (imsize, imsize), np=8)
    channels[1] /= channels[0]
    
    return channels

def get_channel_jet_tracer(file,center,lbox,slab_width=None,jetcolumn=-1,
                           imsize=2000,smlfac=1.0,orientation='xy',weight='tracer'):
    """
    get channels of the jet tracer, if weight='mass', return mass weighted jet tracer
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask,gaspos= cut(part,center,lbox,slab_width,orientation)
    
    gasm = part['Masses'][:][mask]
    gasvol = gasm/part['Density'][:][mask]
    gassl = smlfac*(gasvol**(1/3))
    
    gasjet = part['Jet_Tracer'][:][mask]
    if len(gasjet.shape)>1:
        gasjet = gasjet[:,jetcolumn]    
    gasjet[gasjet<1e-20] = 1e-20
        
    p_resolution = lbox/imsize
    psml = gassl/p_resolution
    psml[psml<1]=1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    if weight=='mass':
        channels = painter.paint(gasdev, psml, [gasm, gasm*gasjet], (imsize, imsize), np=8)    
        channels[1] /= channels[0]
    elif weight=='tracer':
        channels = painter.paint(gasdev, psml, [gasjet, gasjet], (imsize, imsize), np=8)
    else:
        raise NotImplementedError
    return channels




class snapshot:
    """
    plot snapshot
    """
    def __init__(self,fn):
        part = h5py.File(fn,'r')
        self.fn = fn
        try:
            self.UnitLength = part['Header'].attrs['UnitLength_in_cm']
            self.UnitVelocity = part['Header'].attrs['UnitVelocity_in_cm_per_s']
            self.UnitMass = part['Header'].attrs['UnitMass_in_g']
        except:
            self.UnitLength = 3.08568e+21 # kpc
            self.UnitVelocity = 100000 # km/s
            self.UnitMass = 1.989e+43 # 1e10Msun

        self.UnitTime = self.UnitLength/self.UnitVelocity
        self.UnitDensity = self.UnitMass / (self.UnitLength**3)
        self.UnitEnergy = self.UnitMass * (self.UnitLength**2) / (self.UnitTime**2)
        self.temp_to_u = (BOLTZMANN/PROTONMASS)/mu/(gamma-1)/self.UnitVelocity/self.UnitVelocity
        self.rho_to_numdensity = 1.*self.UnitDensity/(mu*PROTONMASS)
        self.time = part['Header'].attrs['Time']*(self.UnitTime/MYR_IN_S)
        
        BoxSize = part['Header'].attrs['BoxSize']
        self.BoxSize = BoxSize
        self.center = np.array([0.5*BoxSize,0.5*BoxSize,0.5*BoxSize])
    
    def project_field(self,lbox,slab_width=None,imsize=2000,orientation='xy'):
        """
        plot projection plot of gas rho, temp, jet tracer and rho colored by T

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image

        Returns
        -------
        mask and relative position to the center
        """
        BoxSize = self.BoxSize
        center = self.center

        cn0 = np.ones((imsize,imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K
        
        #---------
        f,axes = plt.subplots(1,4,figsize=(20,5),sharex=True,sharey=True)
        f.subplots_adjust(hspace=0.05,wspace=0.05)
        
        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        x = channels[0][channels[0]>0]
        vmin0,vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        img_dens = color.CoolWarm(color.NL(cn1,range=(t_low,t_up)),color.NL(channels[0], range=(vmin0,vmax0)))
        img_temp = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(cn0, range=(-7,-6)))
        img_2field = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(channels[0], range=(vmin0,vmax0)))
        
        # jet tracer
        channels = get_channel_jet_tracer(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                          imsize=imsize,jetcolumn=-1,orientation=orientation)
        img_jet = jetmap(color.NL(channels[1],range=(-5,0)),color.NL(cn0, range=(-7,-6)))
        
        #---------
        fields = [img_dens,img_temp,img_jet,img_2field]
        labels = ['Density',r'Temperature',r'JetTracer',r'Dens-T']
        for i in range(0,len(fields)):
            ax = axes.flat[i]
            ax.imshow(fields[i],extent=[0,lbox,0,lbox],origin='lower')
            ax.text(0.05,0.9,labels[i],color='white',fontsize=18,transform=ax.transAxes)
            ax.set_xlabel('pc')
            if i==0:
                ax.set_title('t=%.2f Myr'%(self.time))
                
    def overlap_jet(self,lbox,slab_width=None,imsize=2000,orientation='xy'):
        """
        plot jet map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        """
        BoxSize = self.BoxSize
        center = self.center

        cn0 = np.ones((imsize,imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K
        
        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        x = channels[0][channels[0]>0]
        vmin0,vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        img1 = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(channels[0], range=(vmin0,vmax0)))
        
        # jet tracer
        channels = get_channel_jet_tracer(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                          imsize=imsize,jetcolumn=-1,orientation=orientation)
        img2 = jetmap(color.NL(channels[1],range=(-5,0)),color.NL(cn0, range=(-7,-6)))
        
        #---------
        f,ax = plt.subplots(1,1,figsize=(5,5))
        scale = color.NL(channels[1],range=(-5,0))
        img = overlay(img1,img2,2*np.nan_to_num(scale))
        
        ax.imshow(img,extent=[0,lbox,0,lbox],origin='lower')
        ax.set_xlabel('pc')
        ax.set_title('t=%.2f Myr'%(self.time))
