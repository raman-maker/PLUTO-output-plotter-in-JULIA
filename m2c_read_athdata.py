"""
 #file
 #brief Python script for plotting Athena++ MAD&SANE disc data.
  #             Reads Feng Yuan's group (SHAO) Athena++
  #            results and uses my CAMK Pluto plotting.
  #            First do side-view plot, to read the radius and
  #             theta angle from output parameters, and only then  
  #             the top-view plot.
               
 #author Miljenko Cemeljic (miki@camk.edu.pl)
 #date July, 2020 
 #modified May 2024 to work with python2.7
"""
import os, sys
import numpy as np
import re  # for input colormap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import math
from mpl_toolkits import mplot3d
from scipy import integrate
from scipy.interpolate import interp2d
from scipy.interpolate import bisplrep
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck
import warnings
import matplotlib.path as mpath
from scipy.ndimage import map_coordinates
from pyfiles import read_athinput 
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objs as go
import plotly.io as pio


#specify fonts used in plots-some are overwritten later, but not all
plt.rcParams.update({'font.size': 45, 'font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix'})
axis_font = {'fontname':'DejaVu Sans', 'size':'40'}

# Define by hand here along which coordinate lines will quantities be calculated
#R[194]=139.9,R[191]=130.6,R[149]=50,R[127]=30,R[197]=149.89,R[175]=90.4,R[170]=80.6
#R[80]=9.95,R[139]=39.5,R[210]=202.09,
linr=287#155#139#number of grid cells in radial direction,Rmax=287
 
iznad_key=0#0 default sideview, 1 for view from above

# Choose the min and max Bphi
vvminb=0.1
vvmaxb=0.01 #set this for Bphi

vphimax=0.5

phic=0#default
lina=0#default
linr1=0#default
linr2as=0
linr2bs=0
#ccm Input by hand in the lines below!
interac_key = 0 #default=0 for input by hand; 1 for interactive
if interac_key == 1:
# Query for input of Athena++ data file
    print('Enter Athena++ *.athdf file number (without 0s):')
    athfile=eval(input())
    print(('The input string is:', athfile))
    tim=athfile
    
#    '''#uncomment if wish changing phi by hand

    if iznad_key == 0: #set iznad_key above!
        print('Enter index of azimuthal angle phi')
        azind=eval(input())
        print(('The index of azimuthal angle is:', azind))
        phic=azind
    else:
        print('Enter index of angle for radius of middle of plasmoid')
        rmid=eval(input())
        print(('The index of angle for radius of middle of plasmoid is:', rmid))
        print('Enter index of angle for height above equatorial plane')
        lina=eval(input())
        print(('The index of angle for height above equatorial plane is:', lina))

#    '''
if interac_key == 0: 
    tim=1800 # If not interactively, insert here by hand, without 0s:
# Choose the plane of cut in phi=phic:
    phic=0
#
timm=str(tim)
#
# Load data
#ccm--010220--part from FYuan group for loading the data

from pyfiles import * 

runname='./' #define start dir and file, uncomment below

sane = 1#0 for MAD98, 1 for SANE00
if sane == 1:
    inputfile=runname+'sane00.athinput'

datafile='sane00.prim.01800.athdf'
data=read_data(datafile,athinput=inputfile)
d       = data.rho # density
prs     =data.prs  #pressure

r=data.metric.r[0,0,:] #
theta=data.metric.theta[0,:,0]#
phi=data.metric.phi[:,0,0]#

nr=len(r)
ntheta=len(theta)
nphi=len(phi)
# velocity of  physical quantities
uc=data.ucon()
ecov=data.metric.ecov()
u=TensorDotProduct(ecov,uc)
vr=u[1,0]/u[0,0]
vtheta=u[2,0]/u[0,0]
vphi=u[3,0]/u[0,0]
# magnetic field of physical quantities
b=data.bcon()
#ecov=data.metric.ecov()
B=TensorDotProduct(ecov,b)
br=B[1,0]
btheta=B[2,0]
bphi=B[3,0]
beta=data.PlasmaBeta()
#--end of part from FYuan gruop
#ccm210720-Bernoulli condition
#
#ccm020820 added for psi plot
rh=data.metric.r
thetah=data.metric.theta
rh=rh[0]
thetah=thetah[0]
brh=br[0]#sum(br,axis=0)/64.
where_are_nan = np.isnan(brh)
brh[where_are_nan] = 0
nthetah,nrh= shape(rh)
beta=beta[0]

bernouli=(1.+4.*prs/d)*u[0,0]

#
# ============================= Main =================================
def main():
    global lina, linas, rmid, linr, linr2as, linr2bs

# Directory with this script:
    directory='/home/raman/Pictures/MikiFor2to3/'
    dir=directory
    N_files=1
    N_r=nr
    N_theta=ntheta
    N_phi=nphi

    R = data.metric.r[0,0,:]#np.zeros((N_r))
    T = data.metric.theta[0,:,0]#np.zeros((N_theta))
    P = data.metric.phi[:,0,0]#np.zeros((N_phi))

    print(('linr, R[linr], R[nr-1]=',linr, R[linr], R[nr-1]))
    print(('lina, T[lina], T[ntheta-1]=',lina, T[lina], T[ntheta-1]))
    print(('phic, P[phic], P[nphi-1]=',phic, P[phic], P[nphi-1]))

# Reshape 3D arrays into 2D, choose azimuthal plane:
    br2 = np.zeros((N_theta, N_r))
    btheta2 = np.zeros((N_theta, N_r))
    bphi2 = np.zeros((N_theta, N_r))

    for i in range(0, nr-1):
        for j in range(0,ntheta-1):
            br2[j,i]=br[phic,j,i]
            btheta2[j,i]=btheta[phic,j,i]
            bphi2[j,i]=bphi[phic,j,i]
# For streamlines color:
            if bphi2[j,i] < 0:
                bphi2[j,i]=-0.1
            else:
                bphi2[j,i]=0.1
#           
#=============================================
#ccm--Inputs by hand, to customize the result
#=============================================    
# Uncomment the wanted quantit(ies) for plotting by setting
# their key name to =1 (default is 0)
    density_key=density3_key=pressure_key=veloc_key=Bpfield_key=0
    Bpfieldvec_key=Bphifield_key=psicontour_key=psicontourb_key=0
    curcont1_key=curcont2_key=curdens_key=sigmap_key=sigmat_key=0
    vphifieldvec_key=line_key=Bphicont_key=Bphicont3_key=key_key=0
    Bphi3_key=filtline=Bernouli_key=Bernoulicont_key=Bpstream_key=0
    bottom_key=plasbeta_key=0
        
# The rest is side view:    
    density_key=1
    Bpstream_key=1
    line_key=1 #line along which we compute velocities, forces
# Set by hand here, initial guesses, later improved by filters:    
    linr1=127#143#135#97#109#120#number of grid cells in R to point in the middle of plasmoid
    lina=89#95# number of grid cells in theta direction to the middle of plasmoid
#
# If second pair of coordinate lines along which to compute is set:
    linrs=114#157#140
    linas=43#36#34
    irm=linr1#defaults needed if filtline =0
    jrm=lina
        
    irms=linrs
    jrms=linas
# Define the domain-here one can zoom the view    
    x_lim=55#50#275#150#55#40#25#30#2#1225
    y_lim=55#50#800#155#100#35#2#1225

# Choose number of files: (0 is default for 1 file)
    t=0

# Use this to get velocity vectors of same length or multiply by a factor
    factv=200.#30
    factB=1.
    facvu=1
    drv1=0#1shift in start point of plotted vectors
    drvv=70#index of position in R for which you set condition
    dv=2#plot every #th vector, 1 is for a vector/cell.
    dvb=1
    dvc=dv
    
    min = 0
    max = x_lim#phys. distance, not grid,enables zooming into the same box
    maxy = y_lim
#    maxr = linr#max nr. of grid cells for given zoom 

#ccm310720 shao--perform interpolation in spherical coords, to obtain
#  more points for computing smoother streamlines later. Following the
# method from python routine supplied with Athena++

# Read data
#    level = None
    r_max = max
#    r_max = np.log10(r_max)# Account for logarithmic radial coordinate
#    r_a = np.log10(r)# Account for logarithmic radial coordinate
    r_a = r
    r_grid, theta_grid = np.meshgrid(r_a, theta)
    x_grid = r_grid * np.sin(theta_grid)
    y_grid = r_grid * np.cos(theta_grid)

# Create streamline grid
    x_stream = np.linspace(0,r_max,1000)#-r_max, r_max)
    z_stream = np.linspace(-maxy, maxy)#(-r_max, r_max,1000)#-r_max, r_max)
    x_grid_stream, z_grid_stream = np.meshgrid(x_stream, z_stream)
    r_grid_stream_coord = (x_grid_stream.T**2 + z_grid_stream.T**2) ** 0.5
    theta_grid_stream_coord = np.pi - \
       np.arctan2(x_grid_stream.T, -z_grid_stream.T)
    theta_grid_stream_pix = ((theta_grid_stream_coord + theta[0])
           / (2.0*np.pi + 2.0 * theta[0])) * (2 * ntheta + 1)

    r_grid_stream_pix = np.empty_like(r_grid_stream_coord)
    for (i, j), r_val in np.ndenumerate(r_grid_stream_coord):
        index = sum(r < r_val) - 1
        if index < 0:
            r_grid_stream_pix[i, j] = -1
        elif index < nr - 1:
            r_grid_stream_pix[i, j] = index + \
                (r_val - r[index]) / (r[index + 1] - r[index])
        else:
            r_grid_stream_pix[i, j] = nr

#ccm--150820--without unifying in theta (above), to obtain actual position 
    vals_r_right = br[phic-1, :, :].T
    vals_r_left = br[phic, :, :].T
    vals_theta_right = btheta[phic-1, :, :].T
    vals_theta_left = -btheta[phic, :, :].T
    vals_phi = bphi[:, linas, :].T

# Join vector data through boundaries
    vals_r = np.hstack((vals_r_left[:, :1], vals_r_right, vals_r_left[:, ::-1],
            vals_r_right[:, :1]))
    vals_r = map_coordinates(vals_r, (r_grid_stream_pix, theta_grid_stream_pix),
            order=1, cval=np.nan)
    vals_theta = np.hstack((vals_theta_left[:, :1], vals_theta_right,
            vals_theta_left[:, ::-1], vals_theta_right[:, :1]))
    vals_theta = map_coordinates(vals_theta,
           (r_grid_stream_pix, theta_grid_stream_pix),
               order=1, cval=np.nan)

# Transform vector data to Cartesian components
    r_vals = r_grid_stream_coord
    sin_theta = np.sin(theta_grid_stream_coord)
    cos_theta = np.cos(theta_grid_stream_coord)
    dx_dr = sin_theta
    dz_dr = cos_theta
    dx_dtheta = r_vals * cos_theta
    dz_dtheta = -r_vals * sin_theta
    dx_dtheta /= r_vals
    dz_dtheta /= r_vals
    vals_x = dx_dr * vals_r + dx_dtheta * vals_theta
    vals_z = dz_dr * vals_r + dz_dtheta * vals_theta

    print(('shape(x_stream, z_stream)=', shape(x_stream), shape(z_stream)))    
    print(('shape(vals_r, vals_theta, vals_phi)=',shape(vals_r), shape(vals_theta), shape(vals_phi)))

# Interpolate var_phi to the same shape in (x_stream, z_stream)
    b_phi = interp2d(theta, r, bphi2.T)
    b2_phi = b_phi(x_stream, z_stream)
    print(('shape(b2_phi)=',shape(b2_phi)))

#ccm240620shao--
# Project spherical data to equatorial plane and parallel to it (polar)
# create polar angles shifted for half angular width of a cell, to close
# the circle, otherwise half grid cell will be missing from each side
    phi2 = np.zeros((nphi-1))#new angle, same indices
    phi2 = np.linspace(0,2.*np.pi,nphi+1)#now angle starts from 0,not Dphi/2
    
#    print('phi2,phi=',phi2,phi)
    x3, y3 = pol2cart(lina, phi2, r)
    print(('shapes of rr,x3,y3=',shape(r),shape(x3),shape(y3)))
#ccm240620shao--
#
#ccm021120CAMK            
    vminbpfi=7.5e-5#1.e-5
    vmaxbpfi=2.e-2
    rphi=linrs#183
#    print('r[rphi]=',r[rphi])

# Project spherical data to the cartesian plane with phi=phic where needed
    x, z = spherical_to_cartesian(r, theta)
    print(('shape x,z=',shape(x),shape(z)))

    v_x, v_z = spherical_to_cartesian_field(N_phi, N_theta, N_r, phic, theta, vr, vtheta)
    
    B_x, B_z = spherical_to_cartesian_bfield(N_phi, N_theta, N_r, phic, theta, br, btheta)

# Filter-out unwanted part of the velocity vector field (in plots)
    with np.errstate(invalid='ignore'):  # ignore division by zero
        N = np.sqrt(v_x**2 + v_z**2)
        v_x, v_z = np.divide(v_x, N), np.divide(v_z, N)#normalize

    with np.errstate(invalid='ignore'):  # ignore division by zero
        Nb = np.sqrt(B_x**2 + B_z**2)
        B_x, B_z = np.divide(B_x, Nb), np.divide(B_z, Nb)
        B_x, B_z = factB*np.multiply(B_x, Nb), factB*np.multiply(B_z, Nb)#return to nonnormalized
        B_x, B_z = np.multiply(factB*B_x, Nb), np.multiply(factB*B_z, Nb)

# Calculate poloidal velocity field magnitude
    v_p = np.zeros((N_phi, N_theta, N_r))
    v_p = np.sqrt(vr**2 + vtheta**2)

# Calculate poloidal magnetic field magnitude
    B_p = np.zeros((N_phi, N_theta, N_r))
    B_p = np.sqrt(br**2 + btheta**2)#defined at each phi
    
# Toroidal field
    B_phi = bphi
#    print('B_phi=',B_phi[phic,49,103])

# Toroidal velocity
    v_phi = vphi

# Magnetization sigma
# Poloidal sigma:
    sigmap=np.zeros((N_phi, N_theta, N_r))
    sigmap=B_p*B_p/d
# Total sigma:   
    sigmat=np.zeros((N_phi, N_theta, N_r)) 
    sigmat=(B_p*B_p+B_phi*B_phi)/d

# Calculate the Stokes stream function (psi) of the poloidal magnetic field    
    order=1 #0,1,-1; default 1

    psi = psi_func(nrh, nthetah, rh, thetah, brh)

    psib = np.zeros((N_r, N_theta))
    psib = stream_functionb(r, theta, br[phic,:,:].T, btheta[phic,:,:].T)
    
# Contour_levels 
    psicontourb_levels = np.linspace(-100,100,30)
    
#    psimax,psimin=psi.max(),psi.min()
#    print('psimin,psimax=',psi.min(),psi.max())
    psicont0_levels = np.linspace(-27.57,27.72,12) 
#    
    psicont1_levels = np.linspace(4.e0,10,10)#.001, 10., 50)
    psicont2_levels = np.linspace(-10,-4.e0,10)#-10., -.001, 50)    
#
    curcont1_levels= np.linspace(-10.,-0.001,40)#red
    curcont2_levels= np.linspace(0.001,10.,40)#green

    Bphicont_levels = [-0.02]
    sigmap1_levs = [1.0]

    Bern_levels = [1.0]
#    colormap =  custom_colormap('cmap.csv', reverse=False)

    colormap='jet'#gnuplot2'#jet,jet_r,nipy_spectral,seismic,inferno

# Define plot(s), 
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
   
    if not os.path.isdir(dir): os.makedirs(dir)
    for t in range(N_files):
        ax.cla()
        xlabel = r"$\mathbf{r}$"
        ylabel = r"$\mathbf{z}$"

        title = []
        title.append(r"$\mathbf{t=%.0f}$"% float(tim*10.))

# Info written above the figure and in terminal
        if ((density3_key == 0) and (Bphi3_key == 0)):
            if bottom_key == 0:#input by hand at _keys

                phica=phi2[phic]*180./np.pi#+(phi[2]-phi[1])/2.)*180./np.pi
                plt.text(x_lim+1,y_lim+1.,r'$\mathbf{\varphi(i=%.0f) =%.1f^o}$'% (float(phic), float(phica)), #halfpi

#                    plt.text(x_lim*180./np.pi/r[rphi],y_lim+5.,r'$\mathbf{r_\varphi=%.0f}$'% (float(r[rphi])),
#                plt.text(x_lim+7,y_lim+2.2,r'$\mathbf{\varphi(i=%.0f)}$'% float(phic),#fullpi
                     horizontalalignment='right',fontsize=35)#in both cases

#MC140420shao--for plot of spiral path
#                plt.text(x_lim-14,y_lim-6.,r'$\mathbf{T=%.0f,\ \varphi =%.1f^o}$'% (float(tim*10.), float(phica)), #halfpi
#                plt.text(x_lim-35, y_lim-55.,r'$\mathbf{T=%.0f,\ \varphi =%.1f^o}$'% (float(tim*10.), float(phica)), #halfpi
#                 horizontalalignment='right',fontsize=70,color='k')#in spiral plots only

        if ((density3_key == 1) or (Bphi3_key == 1)):
            ylabel = xlabel
            linad=lina#hand input, number of grid cells in theta to the middle of plasmoid
            plt.text(x_lim+4,x_lim+1.3,r'$\mathbf{Z(i=%.0f)}$'% float(linad),
              horizontalalignment='right',fontsize=50)
            
# General part on axes and labels in all cases
        ax.set_xlabel(xlabel,**axis_font)
        ax.set_ylabel(ylabel,**axis_font)
        ax.set_xlabel(xlabel,fontsize=50)
        ax.set_ylabel(ylabel,fontsize=50)

#--Set the wanted visualization, theta [0,pi/ or [0,pi/2]:
#        ax.set_ylim(-x_lim, x_lim)#view from above,phi=[0,2pi] case
        ax.set_xlim(0, x_lim)#default for all side-view cases
#        ax.set_xlim(0, x_lim*180./np.pi/r[rphi])#for fi-z plot only        

# Uncomment wanted combination:
        if bottom_key == 0:
            ax.set_ylim(-y_lim, y_lim)#for theta=[0,pi] 
#            ax.set_ylim(0, y_lim)#for theta=[0,pi/2] and fi-z           
        else:
            ax.set_ylim(-y_lim, 0)#for theta=[pi/2,pi]  
#----
        ax.set_title(title[t],x=0.5,y=1.03,**axis_font)#y sets title height
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        ax.xaxis.set_tick_params(width=3,length=5,labelsize=40)
        ax.yaxis.set_tick_params(width=3,length=5,labelsize=40)
        ax.set_aspect('equal',adjustable='box')

        vvmin=1.e-6#1.e-2#ccm--here choose by hand min and max when density,
        vvmax=10.#0.9971 # it sets both plot and colorbar

# Default for all side-view (r-z plane) plots:
        if ((density3_key == 0) and (Bphi3_key == 0)):
            density = ax.pcolormesh(x, z, transpose(d[phic,:,:]),cmap=colormap,
             norm=plt.Normalize(vmin=np.log10(vvmin),vmax=np.log10(vvmax)))

            vminpres=1.e-5#enter by hand here for pressure (temperature)
            vmaxpres=1.e-2#9999
            pressuref = ax.pcolormesh(x, z, transpose(prs[phic,:,:]),#/d[phic,:,:]),
                cmap=colormap, norm=plt.Normalize(vmin=np.log10(vminpres),
                vmax=np.log10(vmaxpres)))

            vminbp=7.5e-5#1.e-5
            vmaxbp=2.e-2
            Bpfieldf = ax.pcolormesh(x, z, transpose(B_p[phic,:,:]),
              cmap=colormap,norm=plt.Normalize(vmin=np.log10(vminbp), 
             vmax=np.log10(vmaxbp)))

            sigmapf = ax.pcolormesh(x, z, transpose(sigmap[phic,:,:])        
            ,cmap=colormap,norm=plt.Normalize(vmin=np.log10(0.01), 
             vmax=np.log10(10)))

            sigmatf = ax.pcolormesh(x, z, transpose(sigmat[phic,:,:])
             ,cmap=colormap,norm=plt.Normalize(vmin=np.log10(0.01),
               vmax=np.log10(10)))

            aspect = 40 
            pad_fraction = 0.5
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            cbar = plt.colorbar(density, cax=cax)
            cbar.set_label(r"$\mathbf{\rm{log}_{10}(\rho)}$")
            cbar.ax.tick_params(labelsize=40)
            
# Density:
        if density_key == 1:
            density = ax.pcolormesh(x, z, transpose(d[phic,:,:]),
            cmap=colormap, norm=LogNorm(vmin=vvmin, vmax=vvmax))             

# Bp streamplot      
        if Bpstream_key == 1:
            ax.streamplot(x_stream, z_stream, vals_x.T, vals_z.T,
                density=3, linewidth=2, arrowsize=3,
                color='k', 
#                color=b2_phi, cmap='seismic'
                 )                

        if line_key == 1:
#--Set linr, definition of lina is at the beginning of the script
# or if you wish to find it by some condition, set filtline=1 (default =0):

# Define the approximate R of a point in middle of the plasmoid by hand,
# then find exact point by some other criteria.

            filtline = filtline#0#1#0 default, set 1 for finding plasmoid center 
            wd=5#wd#5#10set how wide is search (in both directions)
            plusmi=0 #0 for Bp(hi)>=0, 1 for Bp(hi)<0 case
            if filtline == 1:
                linr2a=linr1-wd
                linr2b=linr1+wd
                lina2a=lina-wd
                lina2b=lina+wd
                if Bphicont_key == 1:
                    minBphi=-1.e-9#default
                    maxBphi=1.e-9#default
                else:#for min |B_p|
                    minB_p=100.#default
                irm=jrm=1#defaults
                
                for ir in range (linr2a,linr2b):
                    for jr in range (lina2a,lina2b):
                        if plusmi == 1:
                            minBphi1=B_phi[phic,jr,ir]
                            if minBphi1 < minBphi:
                                minBphi=minBphi1
                                jrm=jr
                                irm=ir
#                            print("minBphi=",minBphi,irm,jrm)

                        if plusmi == 0:
                            if Bphicont_key == 1:
                                maxBphi1=B_phi[phic,jr,ir]
                                if maxBphi1 > maxBphi:
                                    maxBphi=maxBphi1
                                    jrm=jr
                                    irm=ir
#                                    print("maxBphi=",maxBphi,irm,jrm)
                            else:#for min |B_p|
                                minB_p1=B_p[phic,jr,ir]
                                if minB_p1 < minB_p:
                                    minB_p=minB_p1
                                    jrm=jr
                                    irm=ir
#                                print("minB_p(i,j)=",minB_p,irm,jrm)

#Second pair of spherical coordinate lines along which we plot results 
                for ir in range (linr2as,linr2bs):
                    for jr in range (lina2as,lina2bs):
                                minB_p1=B_p[phic,jr,ir]
                                if minB_p1 < minB_p:
                                    minB_p=minB_p1
                                    jrms=jr
                                    irms=ir

            linr=irm
            lina=jrm
            coll='black'
            if sigmap_key == 1:
                coll='red'

            ax.plot(x[linr,:], z[linr,:],coll,linestyle='--',linewidth=6)

            lineang = np.arctan(x[0:linr,lina:lina+1]/z[0:linr,lina:lina+1])
            lineangrad = lineang
            lineang = lineang*180./np.pi
            if lineang[0] < 0:
                lineang = 180.+lineang#when lineang<0 deg
#
            linrs=irms
            linas=jrms
            coll='black'
            coll2='red'
            if sigmap_key == 1:
                coll='red'
            if bottom_key == 0:
#uncomm for 2nd pair
                ax.plot(x[0:nr,linas:linas+1], z[0:nr,linas:linas+1],coll,linestyle='--'
                   ,linewidth=6)
            else:
                ax.plot(x[0:nr,lina:lina+1], z[0:nr,lina:lina+1],coll,linestyle='--'
                   ,linewidth=6)
            ax.plot(x[0:nr,lina-1:lina], z[0:nr,lina-1:lina],coll2,linestyle='--'
            ,linewidth=6)
#uncomm for 2nd pair
#            ax.plot(x[linr,lina-lina:64-lina+lina], 
#            z[linr,lina-lina:64-lina+lina],coll2,linestyle='--'
#            ,linewidth=6)
            ax.plot(x[linr,:], 
            z[linr,:],coll2,linestyle='--'
            ,linewidth=6)

            ax.plot(x[linrs,:], z[linrs,:],coll,linestyle='--',linewidth=6)
#            lineang = np.arctan(x[0:linrs,linas-1:linas]/z[0:linrs,linas-1:linas])
            lineangs = np.arctan(x[0:linrs,linas:linas+1]/z[0:linrs,linas:linas+1])
            lineangrads = lineangs
            lineangs = lineangs*180./np.pi
            if lineangs[0] < 0:
                lineangs = 180.+lineangs#when lineang<0 deg
#
            print('====================================')
            print('++++++++++++++++++++++++++++++++++++')
            print(('t,rsph,ryl,zcyl,phic(rad)',tim*10.,r[linr],x[linr,lina],z[linr,lina],phi[phic]))
            print(('i,j,phic,theta(rad),line_angle(r,theta)[deg]=,phic(deg)',
            linr,lina,phic,lineangrad[0],lineang[0],phi2[phic]*180./np.pi))
            print(('vr,vtheta,vphi_plasmoidcenter=',vr[phic,lina,linr],
            vtheta[phic,lina,linr],vphi[phic,lina,linr]))
            print('++++++++++++++++++++++++++++++++++++')
            print('++++++++++++++++++++++++++++++++++++')
            print(('t,rsph,rcyl,zcyl,phic(rad)',tim*10.,r[linrs], x[linrs,linas],z[linrs,linas],phi2[phic]))
            print(('i,j,phic,theta(rad),line_angle(r,theta)[deg]=,phic(deg)',
            linrs,linas,phic,lineangrads[0],lineangs[0],phi2[phic]*180./np.pi))
            print(('vr,vtheta,vphi_plasmoidcenter=',vr[phic,linas,linrs],
            vtheta[phic,linas,linrs],vphi[phic,linas,linrs]))
            print('++++++++++++++++++++++++++++++++++++')
            print('====================================')
#---------------
# Choose here if doing separate plots or multiple panels plot:
        multip=0#0 default for separate plots, if 1 then multiple panels plot

# Make, name and save plot
        fname = 'image.%04d.png' % t
        plot_path = os.path.join(dir, fname)
        print('Saving frame', plot_path)
        plt.savefig(plot_path, bbox_inches='tight')
        
# Define next plot(s), 
        fig = plt.figure(figsize=(20, 7))
#    ax = plt.subplot(111)

# Plot B_p or B_phi at chosen R, Theta in Phi direction
        bporbphi=0#0 for Bp, 1 for B_phi

        plt.clf()
        x1 = [i for i in range (0,nphi)]
        plt.xticks(size=45)#it is fontsize
        plt.yticks(size=45)
        plt.tick_params(width=2, which='both', size=15)
        plt.tick_params(width=4, which='major', size=22)
        plt.minorticks_on()
        plt.xlabel(r"$i(\varphi)$", size=60)

        if bporbphi == 0:
            plt.ylabel(r"B", size=60)
        else:
            plt.ylabel(r"B$_\varphi$", size=60)

        if bporbphi == 0:
            plt.plot(x1, B_p[:,jrms,irms], color='black', ls='dashdot',label=r"B$_{p}$",linewidth=6)
            plt.plot(x1, br[:,jrms,irms], color='r', ls='solid',label=r"B$_{r}$",linewidth=2)
            plt.plot(x1, btheta[:,jrms,irms], color='r', ls='dashed',label=r"B$_{\theta}$",linewidth=3)
            plt.plot(x1, bphi[:,jrms,irms], color='r', ls='dotted',label=r"B$_{\varphi}$",linewidth=3)

# Plot a horizontal black line at y=0        
            x1 = np.linspace(0,70)
            y1 = np.zeros(len(x1))
            plt.plot(x1, y1, color='black', linewidth=1.5)
            plt.legend(loc='best', fontsize='30',ncol=1)

# Make, name and save plot
        fname = 'balongphi.%04d.png' % t
        plot_path = os.path.join(dir, fname)
        print('Saving frame', plot_path)
        plt.savefig(plot_path, bbox_inches='tight')
        print("leng d=", shape(d))
        print("leng B_p=", shape(B_p))
        print("leng r=", shape(r))
        # plotting
        ax.set_xlabel('B_p', fontsize=14, color='blue')
        ax.set_ylabel('d', fontsize=14, color='blue')
        ax.set_zlabel('r', fontsize=14, color='blue')
        ax.tick_params(axis='both', which='major', labelsize=12)  # X and Y axis
        ax.tick_params(axis='z', which='major', labelsize=12)     # Z axis
        # Define the original grid
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(B_p[1,1,:], d[1,1,:], r, linewidth=0.1)

        # Set ticks size to small
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        # Set labels (optional)
        ax.set_xlabel('B_p_new', fontsize=10, color='black')
        ax.set_ylabel('d_new', fontsize=10, color='black')
        ax.set_zlabel('vals_r', fontsize=10, color='black')

        # Save the figure
        plt.savefig('3d_line_plot.png', dpi=1000, bbox_inches='tight')

        # Show the plot (optional)
        plt.show()

# Plot a two-panels plot :
        if multip == 1:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 3))
            axes[0].ax.fig1
            axes[1].ax.fig2
            fname = 'figspanels'
            print(('Saving panels',fname))
            plt.savefig(dir + fname)
            fig.tight_layout()
            
#---end of main

# ============================= Functions =================================

def spherical_to_cartesian(r, theta):
#    Given 1D arrays for r and theta, the function makes a spherical (r,theta)
#    grid and then transforms it to cartesian coordinates. It outputs two 2D
#    arrays, one for x and one for z. Angle theta is a co-latitudinal coordinate.
    theta_matrix, radius_matrix = np.meshgrid(theta, r)
    x = radius_matrix*np.sin(theta_matrix)
    z = radius_matrix*np.cos(theta_matrix)
    return x,z

def pol2cart(linax, phi2, r):#notice new set of angles phi2 here
    rr=np.zeros((nr))#polar radii
    for i in range (0, nr):
        rr[i] = r[i]*np.sin(theta[linax])                
    phi_matrix, radius_matrix = np.meshgrid(phi2, rr)
    x3 = radius_matrix*np.cos(phi_matrix)
    y3 = radius_matrix*np.sin(phi_matrix)
    return x3, y3

def spherical_to_cartesian_field(N_phi, N_theta, N_r, phic, theta, v_r, v_theta):
#    Given a vector field (v_r,v_theta) in spherical coordinates, the function
#    outputs a field (v_x, v_z) in Cartesian coordinates.
    v_x = np.zeros((N_phi, N_theta, N_r))
    v_z = np.zeros((N_phi, N_theta, N_r))

    for i in range(0, N_theta):
        v_x[phic, i, :] = v_r[phic, i, :]*np.sin(theta[i]) + v_theta[phic, i, :]*np.cos(theta[i])
        v_z[phic, i, :] = v_r[phic, i, :]*np.cos(theta[i]) - v_theta[phic, i, :]*np.sin(theta[i])
    return v_x, v_z

def spherical_to_cartesian_fieldb(Nth, Nr, theta, v_rb, v_thetab):
#    Given a vector field (v_rb,v_thetab) in spherical coordinates, the function
#    outputs a field (v_xb, v_zb) in Cartesian coordinates.
    v_xb = np.zeros((Nth, Nr))
    v_zb = np.zeros((Nth, Nr))

    for i in range(0, Nth):
        v_xb[i, :] = v_rb[i, :]*np.sin(theta[i]) + v_thetab[i, :]*np.cos(theta[i])
        v_zb[i, :] = v_rb[i, :]*np.cos(theta[i]) - v_thetab[i, :]*np.sin(theta[i])
    return v_xb, v_zb

def spherical_to_cartesian_bfield(N_phi, N_theta, N_r, phic, theta, B_r, B_theta):
#    Given a vector field (B_r,B_theta) in spherical coordinates, the function
#    outputs a field (B_x, B_z) in Cartesian coordinates.
    B_x = np.zeros((N_phi, N_theta, N_r))
    B_z = np.zeros((N_phi, N_theta, N_r))

    for i in range(0, N_theta):
        B_x[phic, i, :] = B_r[phic, i, :]*np.sin(theta[i]) + B_theta[phic, i, :]*np.cos(theta[i])
        B_z[phic, i, :] = B_r[phic, i, :]*np.cos(theta[i]) - B_theta[phic, i, :]*np.sin(theta[i])
    return B_x, B_z

def spherical_to_cartesian_bfieldb(Nth, Nr, theta, B_rb, B_thetab):
#    Given a vector field (B_rb,B_thetab) in spherical coordinates, the function
#    outputs a field (B_xb, B_zb) in Cartesian coordinates.
    B_xb = np.zeros((Nth, Nr))
    B_zb = np.zeros((Nth, Nr))

    for i in range(0, Nth):
        B_xb[i, :] = B_rb[i, :]*np.sin(theta[i]) + B_thetab[i, :]*np.cos(theta[i])
        B_zb[i, :] = B_rb[i, :]*np.cos(theta[i]) - B_thetab[i, :]*np.sin(theta[i])
    return B_xb, B_zb

def psi_func(nrh, nthetah, rh, thetah, brh):
#ccm--020820, Miki, SHAO, modification of psi function by HYang's computation
#    print('shape(rh),rh=',shape(rh),rh)
    ntheta,nr= shape(rh)#is ok as both are same shape
    psif = np.zeros([nthetah, nrh])
    for j in range(nrh):
        for i in range(nthetah)[1:]:
            psif[i,j]=psif[i-1,j]+brh[i,j]*rh[i,j]**2*sin(thetah[i,j])*(thetah[i,j]-thetah[i-1,j])
    psi=psif.T
    
    return psi
#    
def stream_function(phic, r, theta, B_r, B_theta, order):
    N_r = len(r)
    N_theta = len(theta)
    
    #calculate derivatives of the stream function
    dpsi_dr = np.zeros((N_r,N_theta))
    dpsi_dtheta = np.zeros((N_r,N_theta))

    for i in range(0, N_theta-1):
        dpsi_dr[:, i] = -r * np.sin(theta[i]) * B_theta[phic, i, :]
        dpsi_dtheta[:, i] = r * r * np.sin(theta[i]) * B_r[phic, i, :]

    # start at integrating at pole at inner radius, do all along pole, then do
    # for each theta
    psi = np.zeros((N_r,N_theta))
    psi_2 = np.zeros((N_r, N_theta))
    dtheta = np.zeros(N_theta)
    dr = np.zeros(N_r)
    if order >= 0:
        dr[1:] = r[1:] - r[:-1]
        dtheta[1:] = theta[1:] - theta[:-1]

        psi[1:, 0] = psi[:-1, 0] + dpsi_dr[1:, 0]*dr[1:]

        for i in range(1, N_theta):
             psi[:, i] = psi[:, i-1] + dpsi_dtheta[:, i] * dtheta[i]

    if order <= 0:
        dr[:-1] = r[:-1] - r[1:]
        dtheta[:-1] = theta[:-1] - theta[1:]

        for i in range(N_r-2, -1, -1):
            psi_2[i, N_theta - 1] = psi_2[i + 1, N_theta - 1] +\
                                dpsi_dr[i, N_theta - 1]*dr[i]
        for i in range(N_theta-2, -1, -1):
            psi_2[:, i] = psi_2[:, i + 1] + dpsi_dtheta[:, i]*dtheta[i]

        if order < 0:
            return psi_2
        else:
            psi = 0.5 * (psi + psi_2)  # Avg of the two

    return psi

def stream_functionc(phic, r, theta, B_r, B_theta, order):
#
    N_r = len(r)
    N_theta = len(theta)

    #calculate derivatives of the stream function
    dpsi_dr = np.zeros((N_r,N_theta))
    dpsi_dtheta = np.zeros((N_r,N_theta))

    for i in range(0, N_theta-1):
        dpsi_dr[:, i] = -r * np.sin(theta[i]) * B_theta[phic, i, :]
        dpsi_dtheta[:, i] = r * r * np.sin(theta[i]) * B_r[phic, i, :]

    # start at integrating at pole at inner radius, do all along pole, then do
    # for each theta
    psi = np.zeros((N_r,N_theta))
    psi_2 = np.zeros((N_r, N_theta))
    dtheta = np.zeros(N_theta)
    dr = np.zeros(N_r)
    if order >= 0:
        dr[1:] = r[1:] - r[:-1]
        dtheta[1:] = theta[1:] - theta[:-1]

        psi[1:, 0] = psi[:-1, 0] + dpsi_dr[1:, 0]*dr[1:]

        for i in range(1, N_theta):
             psi[:, i] = psi[:, i-1] + dpsi_dtheta[:, i] * dtheta[i]

    if order <= 0:
        dr[:-1] = r[:-1] - r[1:]
        dtheta[:-1] = theta[:-1] - theta[1:]

        for i in range(N_r-2, -1, -1):
            psi_2[i, N_theta - 1] = psi_2[i + 1, N_theta - 1] +\
                                dpsi_dr[i, N_theta - 1]*dr[i]
        for i in range(N_theta-2, -1, -1):
            psi_2[:, i] = psi_2[:, i + 1] + dpsi_dtheta[:, i]*dtheta[i]

        if order < 0:
            return psi_2
        else:
            psi = 0.5 * (psi + psi_2)  # Avg of the two

    return psi

def stream_functionb(r, theta, B_rb, B_thetab):
    N_r = len(r)
    N_theta = len(theta)
#    print('r & th=',N_r,N_theta)
    gdet_grid = np.zeros((N_r,N_theta))

    br_grid = (B_rb[:,:])
    th_grid = (theta[:])
#    print('br_grid=',shape(br_grid),br_grid)
#    print('th_grid=',shape(th_grid),th_grid)    
    psib = integrate.cumtrapz(np.sqrt(-gdet_grid)*br_grid*2*np.pi, 
            x=th_grid, axis = 1, initial = 0)   
#    psib = integrate.romb(np.sqrt(-gdet_grid)*br_grid*2*np.pi, 
#            x=th_grid, axis = 1)#, initial = 0)   
    
    return psib

#Creating a vector field inside the disk
def zerowanie(v, phic, N_r, drvv):
    for r in range(0,drvv):
        for z in range(0,drvv):
            if r < drvv:
                v[phic,r,z] = 0.
	       
    return v

main()

