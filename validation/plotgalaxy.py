"""
Some snippets to process the star/galaxy image
"""

import numpy as np
from bigfile import BigFile
from scipy import integrate
import scipy.interpolate as interpolate
from scipy.spatial import cKDTree as KDTree
from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,colorConverter
from matplotlib.collections import LineCollection


# customize the colormap for stellar field
# ------------------------------------
class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

y = np.linspace(0,1,100)
COL = MplColorHelper('pink_r', 0, 1)
x1 = COL.get_rgb(y)

COL = MplColorHelper('gray', 0, 1)
x2 = COL.get_rgb(y)

x = np.concatenate([np.repeat(x2[95:],6,axis=0),x1[0:70]],axis=0)
starmap = color.Colormap(x[:,0:3])

# ---------------------
def age(a0,a1,omm=0.3089,oml=0.6911,hh=0.6774):
    """
    return age between a0 and a1 in yrs
    """
    Mpc_to_km = 3.086e+19
    sec_to_megayr = 3.17098e-8 
    f = lambda a : 1.0/(a*(np.sqrt(omm*(a)**(-3)+oml)))
    t = 1./(hh*100.0)*integrate.quad(f,a0,a1)[0]
    t *= Mpc_to_km*sec_to_megayr
    return t

def smooth(pos):
    """
    pos in shape of (N,3)
    return smoothing length calculated by the k-nearest neighbor
    """
    tree = KDTree(pos)
    d, i = tree.query(pos,k=60)
    return d[:,-1].copy()


def inertia(pos,mass):  
    """
    pos in shape [[x1,x2..],[y1,y2..],[z1,z2..]]
    return Imom eigenvector with eigenvalue from smallest to largest
    """
    g11=np.sum((pos[1]*pos[1]+pos[2]*pos[2])*mass)
    g22=np.sum((pos[0]*pos[0]+pos[2]*pos[2])*mass)
    g33=np.sum((pos[0]*pos[0]+pos[1]*pos[1])*mass)
    g12=-np.sum(pos[0]*pos[1]*mass)
    g13=-np.sum(pos[0]*pos[2]*mass)
    g23=-np.sum(pos[1]*pos[2]*mass)
    g21=g12
    g31=g13
    g32=g23
    mx = np.array([[g11,g12,g13],[g21,g22,g23],[g31,g32,g33]])/np.sum(mass)
    w, v = np.linalg.eig(mx)
    v = v[:, np.abs(w).argsort()] # column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    return v       

def face_on_3rotate(dr,mass,th=-1): 
    """
    dr in shape (N,3)
    if th>0, apply radius cut |dr|<th
    """
    rr = np.linalg.norm(dr,axis=1)
    if th>0:
        mask = rr<th
    else:
        mask = rr>0
    v = inertia(np.transpose(dr[mask]),mass[mask])
    xnew = np.einsum('ij,j->i', dr, v[:,1])
    ynew = np.einsum('ij,j->i', dr, v[:,0])
    znew = np.einsum('ij,j->i', dr, v[:,2])
    newpos = np.transpose(np.array([xnew,ynew,znew]))
    return newpos

def edge_on_3rotate(dr,mass,th=-1):
    """
    dr in shape (N,3)
    """
    rr = np.linalg.norm(dr,axis=1)
    if th>0:
        mask = rr<th
    else:
        mask = rr>0
    v = inertia(np.transpose(dr[mask]),mass[mask])
#     print ("rotate axis:",v)
    xnew = np.einsum('ij,j->i', dr, v[:,1])
    ynew = np.einsum('ij,j->i', dr, v[:,2])
    znew = np.einsum('ij,j->i', dr, v[:,0])
    newpos = np.transpose(np.array([xnew,ynew,znew]))
    return newpos

def place(number, offarray):   # offarray is sorted from smallest to largest 
    dummy = np.where(offarray <= number)[0]
    return dummy[-1]

def extract_host_star(file,searchID,crop=20,orientation='xz',ort_radius=-1):
    """
    return 2D RGBA channel for stellar density-age field
    """
    pig = BigFile(file)
    redshift = 1./pig.open('Header').attrs['Time'][0]-1
    hh = pig.open('Header').attrs['HubbleParam'][0]

    a_cur=1./(1.+redshift)
    SFT_space=np.linspace(0,a_cur,500)
    age_space=[age(SFT,1./(1.+redshift)) for SFT in SFT_space]  # age in year
    gen_SFT_to_age=interpolate.interp1d(SFT_space,age_space,fill_value='extrapolate')

    bhIDs = pig.open('5/ID')[0:1000000]
    
    bhidx = np.where(searchID==bhIDs)[0][0]
    bhpos = pig.open('5/Position')[bhidx]
    center = bhpos

    Length = pig.open('FOFGroups/LengthByType')[0:1000000]
    OffsetByType = np.cumsum(Length,axis=0)
    a1 = np.array([[0,0,0,0,0,0]],dtype=np.uint64)
    OffsetByType = np.append(a1,OffsetByType,axis=0)
    bhoff = OffsetByType[:,5]

    groupindex = place(bhidx,bhoff)
    staroff = np.transpose(OffsetByType[groupindex:groupindex+2])[4]
    
    #---------------------------------------------------------
    pos4 = pig.open('4/Position')[staroff[0]:staroff[1]]
    ppos = pos4 - center

    mask = np.abs(ppos[:,0]) < crop
    mask &= np.abs(ppos[:,1]) < crop
    mask &= np.abs(ppos[:,2]) < crop

    ppos = ppos[mask]

    m4 = pig.open('4/Mass')[staroff[0]:staroff[1]][mask]
    st = pig.open('4/StarFormationTime')[staroff[0]:staroff[1]][mask]
    star_age = gen_SFT_to_age(st)/1e6 # to Myr
    mstar = np.sum(m4)*10**10/hh
    print ("mstar = %.2e"%mstar)
    
    if orientation is 'face_on':
        ppos = face_on_3rotate(ppos,m4,th=ort_radius)

    if orientation is 'edge_on':
        ppos = edge_on_3rotate(ppos,m4,th=ort_radius)

    sml = smooth(ppos)
    sml *= 1.2
    sml[sml>2] = 2

    lbox = 2*crop
    resolution = 0.1
    imsize = int(lbox/resolution)
    print ("imsize = ",imsize)

    if imsize>2000:
        imsize=2000

    ct = lbox/2

    mmv = camera.lookat((0,lbox,0),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
    if orientation is 'xy':
        mmv = camera.lookat((0,0,lbox),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
    if orientation is 'yz':
        mmv = camera.lookat((lbox,0,0),(0,0,0),(0,1,0)) # (position of camera, focal point, up direction)
    if orientation is 'xz':
        mmv = camera.lookat((0,lbox,0),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)

    mpers = camera.ortho(-ct,ct,(-ct,ct,-ct,ct))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix

    star2d = camera.apply(camera.matrix(mpers,mmv),ppos) # apply a camera matrix to data coordinates, return position in clip coordinate
    stardev = camera.todevice(star2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate

    channels = painter.paint(stardev, sml/resolution, [m4,m4*star_age], (imsize, imsize), np=8)
    channels[1] /= channels[0]

    return channels,mstar


def SpikeCollection(x, y, radius, linewidth=1, color=(1, 1, 0.9), alpha=0.9, gamma=3.33):
    """ 
    returns a LineCollection of spikes.
    example:
    ax = gca()
    c = SpikeCollection(x=[0, 5, -3, 3], y=[0, 2, -1, -5], radius=[10, 5, 5, 8], 
                  color=[(1, 1, 0.9), (0, 1, 0)], alpha=0.9)
    ax.add_collection(c)
    x, y, radius needs to be of the same length, otherwise the length of x is taken as the number of points
    color is repeated if it is shorter than the list of x, y radius
    alpha is the maximum alpha in the spike. gamma is used to correct for the buggy matplotlib transparency code
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    radius = np.asarray(radius).reshape(-1)

    Npoints = x.shape[0]

    alpha0 = 0.05
    Nseg = int(alpha / alpha0)
    alpha0 = alpha / Nseg

    l = np.linspace(0, 1, Nseg, endpoint=False)
    l **= 1 / gamma

    lines = np.zeros((Npoints, Nseg * 2, 2, 2))
    lines[:, :Nseg, 0, 1] = (-1 + l)
    lines[:, :Nseg, 1, 1] = (1. - l)

    lines[:, Nseg:, 0, 0] = (-1 + l)
    lines[:, Nseg:, 1, 0] = (1. - l)

    lines[:, :, :, :] *= radius[:, np.newaxis, np.newaxis, np.newaxis]
    lines[:, :, :, 0] += x[:, np.newaxis, np.newaxis]
    lines[:, :, :, 1] += y[:, np.newaxis, np.newaxis]

    lines.shape = -1, 2, 2

    colors = colorConverter.to_rgba_array(color).repeat(Nseg * 2, axis=0).reshape(-1, Nseg * 2, 4)
    # this formular is math trick:
    # it ensures the alpha value of the line segment N, when overplot on the previous line segment 0..N-1,
    # gives the correct alpha_N = N * alpha0.
    # solve (1 - x_N) (N-1) alpha_0 + x_N = N alpha_0
    colors[:, :Nseg, 3] = 1.0 / ( 1. / alpha0 - np.arange(Nseg))[np.newaxis, :]
    colors[:, Nseg:, 3] = colors[:, :Nseg, 3]
    colors.shape = -1, 4

    c = LineCollection(lines, linewidth, colors=colors, antialiased=1)
    return c






