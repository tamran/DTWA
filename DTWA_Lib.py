import matplotlib.pyplot as plt
from TamLib import *
from matplotlib import cm, colors
from scipy.integrate import odeint
import numpy as np
import math as mth
import scipy
import scipy.sparse as sparse
from qutip import *
from scipy import sparse, linalg
from scipy.interpolate import interp1d
import itertools as it
from itertools import *
from collections import Counter 
from timeit import default_timer as timer
import time
from mpl_toolkits.mplot3d import Axes3D
import imageio

# DTWA GENERAL HELPERS ##############################################################################################

# generates nt initial configurations = # of trajectories for the DTWA with all spins pointing in the direction given by axis
# returns a matrix of dimension nt x N x 3 where the rows = number of init conditions/rajectories, col = each spin, page = vals of spin component
def genInitConfigs(N,nt,axis="x"):    
    if axis=="x":
        return np.array([np.concatenate(((1/2)*np.ones((N,1)),np.random.randint(0,2,size=(N,2))-(1/2)),axis=1) for i in range(nt)])
    elif axis=="y":
        return np.array([np.concatenate((np.random.randint(0,2,size=(N,1))-(1/2),(1/2)*np.ones((N,1)),np.random.randint(0,2,size=(N,1))-(1/2)),axis=1) for i in range(nt)])
    else:
        return np.array([np.concatenate((np.random.randint(0,2,size=(N,2))-(1/2),(1/2)*np.ones((N,1))),axis=1) for i in range(nt)])

# converts spin components from x,y,z basis to +/-/z basis
def cart2ladder(config):
    if len(config.shape)==3:
        x = config[:,:,0]
        y = config[:,:,1]
        p = (x + 1j*y)/2
        m = (x - 1j*y)/2
        return np.stack((p,m,config[:,:,2]),axis=2)
    else:
        x = config[:,0]
        y = config[:,1]
        p = (x + 1j*y)/2
        m = (x - 1j*y)/2
        return np.stack((p,m,config[:,2]),axis=1)

# converts spin components from +/-/z basis to x,y,z basis
def ladder2cart(config):
    if len(config.shape)==3:
        p = config[:,:,0]
        m = config[:,:,1]
        x = p + m
        y = (p - m)/1j
        return np.stack((x,y,config[:,:,2]),axis=2)
    else:
        p = config[:,0]
        m = config[:,1]
        x = p + m
        y = (p - m)/1j
        return np.stack((x,y,config[:,2]),axis=1)

# DTWA ISING DYNAMICS ##############################################################################################

#### Generates general all-to-all Ising model, with generic power law interactions given by Jfunc
# alpha = exponent of decay for interaction strength  J_ij ~ 1/|r_ij|^α
# coord = tuple of (x,y,z) coordinates of each spin
# configs = initial configs in cartesian basis
# returns ALL trajectories as well as the mean config at each time (i.e. mean trajectory) 
# trajectory is a matrix of dimensions: num(times) x nt x N x 3 components where the nt is the number of trajectories each computed for num(times) time points
def IsingEvolve(configs,tvec,Jz,coord=[],Jfunc=[],alpha=[]):
    lconfigs = cart2ladder(configs)
    z = lconfigs[:,:,2]
    
    if coord!=[] and (Jfunc!=[] or alpha!=[]):         # ANISOTROPIC POWER LAW INTERACTIONS
        # get J_ij INTERACTING MATRIX
        Jz_ij = getJij(coord,Jz,alpha,Jfunc=Jfunc)
    else:                                             # UNFIORM INTERACTIONS
        Jz_ij = Jz*(np.ones((configs.shape[1],configs.shape[1])) - np.eye(configs.shape[1]))
    
    config_evol = []
    meanConfig_evol = []
    for t in tvec:
        tdist_ladder = []
        for i in range(0,2):
            s0 = lconfigs[:,:,i].transpose()  # s_i^z = col
            st = s0*np.exp(((-1)**i)*1j*t*(Jz_ij@(lconfigs[:,:,2].transpose()))) 

            tdist_ladder.append(st.transpose())

        tdist_ladder.append(lconfigs[:,:,2])
        tdist_ladder = np.rollaxis(np.array(tdist_ladder),0,3)
        
        tdist = ladder2cart(tdist_ladder)
        config_evol.append(tdist)
        # meanConfig_evol.append(np.mean(tdist,axis=0))  

    config_evol = np.array(config_evol)
    meanConfig_evol = np.mean(config_evol,axis=1)   # take average across trajectories/trials

    return config_evol, meanConfig_evol

# DTWA XY HAMILT DYNAMICS ##############################################################################################

# unravel spin system configuration into vector with spin components of each spin [X1,X2,..XN,Y1,..YN,Z1...ZN]
def config2vec(config):
    return np.ravel(config,order='F')

# get back spin config as an N x 3 matrix
def vec2config(vconfig):
    return np.reshape(b,(int(len(vconfig)/3),3),order='F')

def XYderiv(vconfig, t, J_ij):
    N = int(len(vconfig)/3)
    x = vconfig[0:N]
    y = vconfig[N:2*N]
    z = vconfig[2*N:3*N]
    betay = (J_ij@y)
    betax = (J_ij@x)
    dxdt = z*betay
    dydt = -z*betax
    dzdt = (y*betax) - (x*betay)
    
    return np.concatenate((dxdt,dydt,dzdt),0)

#### Generates evolution under XY Hamiltonian sum_ij (Jperp_ij/2)*(X_i X_j + Y_i Y_j) with generic power law interactions
# Solution is computed by a numerical ODE solver
# alpha = exponent of decay for interaction strength  J_ij ~ 1/|r_ij|^α
# coord = tuple of (x,y,z) coordinates of each spin
# initConfigs = initial configs in cartesian basis
def XYEvolve(initConfigs,tvec,Jperp,coord=[],Jfunc=[],alpha=[]):
    trajectories = []
    config_evol = []
    meanConfig_evol = []

    if coord!=[] and (Jfunc!=[] or alpha!=[]):  # ANISOTROPIC POWER LAW INTERACTIONS
        # get J_ij INTERACTING MATRIX
        J_ij = getJij(coord,Jperp,alpha,Jfunc=Jfunc)
    else:                                             # UNFIORM INTERACTIONS
        J_ij = Jperp*(np.ones((initConfigs.shape[1],initConfigs.shape[1])) - np.eye(initConfigs.shape[1]))

    for i in range(initConfigs.shape[0]):
    #     print("i: %d" %i)
        currtraj = odeint(XYderiv,config2vec(initConfigs[i,:,:]),tvec,args=(J_ij,))
        trajectories.append(np.reshape(currtraj,(len(tvec),int(currtraj.shape[1]/3),3),order='F'))

    config_evol = np.array(trajectories)  # nt x times x N x 3components
    meanConfig_evol = np.mean(config_evol,axis=0)

    return config_evol, meanConfig_evol


# DTWA TF HAMILT DYNAMICS ##############################################################################################

#### Generates evolution under Transverse field Hamiltonian sum_i h_i X_i with 
# Solution is computed exactly as a rotation of the spin components
# h = vector representing sitewide transverse field
# initConfigs = initial configs in cartesian basis
def TFEvolve(initConfigs,tvec,h):
    config_evol = []
    for t in tvec:
        y0 = initConfigs[:,:,1]
        z0 = initConfigs[:,:,2]
        zt = z0*np.cos(h*t) + y0*np.sin(h*t)
        yt = -z0*np.sin(h*t) + y0*np.cos(h*t)
        tconfig = np.stack((initConfigs[:,:,0],yt,zt),axis=2)
        config_evol.append(tconfig)

    config_evol = np.array(config_evol)
    meanConfig_evol = np.mean(config_evol,axis=1)   # take average across trajectories/trials

    return config_evol, meanConfig_evol


# DTWA SQUEEZING ##############################################################################################

# returns expectation value of anticommutator {S_y,S_z} given some distribution of individual spins
def getW(dist):
    return np.mean((2*np.sum(dist[:,:,1],axis=1)*np.sum(dist[:,:,2],axis=1)))

# <S_y^2> +/- <S_z^2>
def getVpm(dist,vtype='p'):
    if vtype=='p':
        return np.mean(np.sum(dist[:,:,1],axis=1)**2) + np.mean(np.sum(dist[:,:,2],axis=1)**2)
    else:
        return np.mean(np.sum(dist[:,:,1],axis=1)**2) - np.mean(np.sum(dist[:,:,2],axis=1)**2)

# returns variances of each spin component <S_a^2> given some distribution of individual spins
def getSsqExpect(dist,axis='z'):
    if axis=='x':
        return np.mean(np.sum(dist[:,:,0],axis=1)**2, axis=0)
    if axis=='y':
        return np.mean(np.sum(dist[:,:,1],axis=1)**2, axis=0)
    if axis=='z':
        return np.mean(np.sum(dist[:,:,2],axis=1)**2, axis=0)
    
# search for minimum squeezing parameter, aka best squeezing, within tvec given some distribution of individual spins
def getDTWASqueezingParam(N,nt,tvec,Jz,coord=[],Jfunc=[],alpha=[]):
    initConfigs = genInitConfigs(N,nt,axis='x')
    tdist, meanConfig_evol = IsingEvolve(initConfigs,tvec,Jz,coord=coord,Jfunc=Jfunc,alpha=alpha)  # tdist is the distribution at time t

    W_dtwa = np.array([getW(dist) for dist in tdist])
    Vp_dtwa = np.array([getVpm(dist,'p') for dist in tdist])
    Vm_dtwa = np.array([getVpm(dist,'m') for dist in tdist])
    squeezing_dtwa_vec = ((Vp_dtwa - np.sqrt(W_dtwa**2 + Vm_dtwa**2))/(N/2)).real
    if not np.allclose(squeezing_dtwa_vec,squeezing_dtwa_vec.real):
        raise Exception("Imag squeezing parameters in vec!")
    return np.min(squeezing_dtwa_vec),squeezing_dtwa_vec         
    

# VISUALIZATION ##############################################################################################

# for each time in tvec, make a subplot with a still frame of the dist within the Bloch sphere
# sphere = set of coordinates of the Bloch sphere if computed already
def visualizeBlochEvol(tvec,dist_evol,sphere=[],axesLabels=False,showAxes=True):
    if sphere==[]:
        N = dist_evol[0].shape[1]
        phi = np.linspace(0, np.pi, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        R = N*(np.sqrt(3)/2)
        x = R*np.sin(phi) * np.cos(theta)
        y = R*np.sin(phi) * np.sin(theta)
        z = R*np.cos(phi)
    else:
        x,y,z = sphere

    fig = plt.figure(figsize=plt.figaspect(1/len(tvec)))
    for i,t in enumerate(tvec):
        ax = fig.add_subplot(1, len(tvec), i+1, projection='3d')
        ax.plot_surface(x, y, z,  rstride=1,color="lightsteelblue",alpha=0.25)
        ax.set_title("t = %.3g" %t)
        ax.dist = 11
        
        xx,yy,zz = np.hsplit(np.sum(dist_evol[i],axis=1).real,3)
        ax.scatter(xx,yy,zz,s=2,alpha=0.25)
        if not showAxes:
            ax.set_axis_off()
        else:
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            ax.set_zticks([]) 
        if axesLabels:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

# view single still frame of distribution on Bloch sphere at some time t
def visualizeBloch(dist,t=-1,sphere=[],viewAngle = [],alpha=0.05,showProjection=False,axesLabels=False,showAxes=True):
    if sphere==[]:
        N = dist.shape[1]
        phi = np.linspace(0, np.pi, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        R = N*(np.sqrt(3)/2)
        x = R*np.sin(phi) * np.cos(theta)
        y = R*np.sin(phi) * np.sin(theta)
        z = R*np.cos(phi)
    else:
        x,y,z = sphere

    fig = plt.figure(figsize=(5,5))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, rstride=1,color="lightsteelblue",alpha=0.3)
    if viewAngle!=[]:
        ax.view_init(viewAngle[0], viewAngle[1])
    xx,yy,zz = np.hsplit(np.sum(dist,axis=1).real,3)
    
    ax.scatter(xx,yy,zz,s=5,alpha=alpha)
    if showProjection:
        ax.scatter(xx,yy,zdir='z',zs = -1.75*R,s=5,alpha=alpha)
        ax.scatter(yy,zz,zdir='x',zs = -1.75*R,s=5,alpha=alpha)
        ax.scatter(xx,zz,zdir='y',zs = 1.75*R,s=5,alpha=alpha)
        
        lims = [-1.25*R,1.25*R]
        ax.auto_scale_xyz(lims,lims,lims)

    
    if not t==-1:
        ax.set_title("t = %.3g" %t)
    if not showAxes:
        ax.set_axis_off()
    else:
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([]) 
    if axesLabels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    return (x,y,z), plt.gca()


# creates a frame for animation
def getBlochFrame(t,tdist,sphere=[],viewAngle=[],alpha=0.05,showProjection=False,axesLabels=True,showAxes=True):
    if sphere==[]:
        N = tdist.shape[1]
        phi = np.linspace(0, np.pi, 100)
        theta = np.linspace(0, 2*np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        R = N*(np.sqrt(3)/2)
        x = R*np.sin(phi) * np.cos(theta)
        y = R*np.sin(phi) * np.sin(theta)
        z = R*np.cos(phi)
    else:
        x,y,z = sphere
        R = np.sqrt(np.amax(x)**2 + np.amax(y)**2 + np.amax(z)**2)

    fig, ax = plt.subplots(figsize=(5,5))
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z,  rstride=1,color="lightsteelblue",alpha=0.3)
    
    if not viewAngle==[]:
        ax.view_init(elev=viewAngle[0],azim=viewAngle[1])
    
    xx,yy,zz = np.hsplit(np.sum(tdist,axis=1).real,3)
    ax.scatter(xx,yy,zz,s=3,alpha=alpha)
    
    if showProjection:
        ax.scatter(xx,yy,zdir='z',zs = -1.5*R,s=3,alpha=alpha)
        ax.scatter(yy,zz,zdir='x',zs = -1.5*R,s=3,alpha=alpha)
        ax.scatter(xx,zz,zdir='y',zs = 1.5*R,s=3,alpha=alpha)
        lims = [-1.3*R,1.3*R]
        ax.auto_scale_xyz(lims,lims,lims)
    
    if not showAxes:
        ax.set_axis_off()
    else:
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([]) 
    if axesLabels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
    ax.set_title("t = %.3g" %t,x=0.5,y=1,pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

# generates animation of evolution of 3D probability distribution in Bloch sphere
# showProjection also shows projection of marginal probability distribution on each plane xy, xz, yz
def animateBloch(tvec,tdist,sphere=[],viewAngle=[],alpha=0.05,showProjection=False,axesLabels=True,showAxes=True,filename=[],saveBool=False,fps=2):
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    plt.ioff()
    ImList = [getBlochFrame(tvec[i],tdist[i],sphere=sphere,viewAngle=viewAngle,alpha=alpha,showProjection=showProjection,axesLabels=axesLabels,showAxes=showAxes) for i in range(len(tvec))]
    plt.ion()
    if saveBool:
        if filename==[]:
            filename = "./animateBloch_N_%d_nt_%.2g_tmax_%0.3g.gif" % (tdist[0].shape[1],tdist[0].shape[0],tvec[-1])
        imageio.mimsave(filename,ImList,fps=fps)
    return ImList
    