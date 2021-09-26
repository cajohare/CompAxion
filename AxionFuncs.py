#================================PlotFuncs.py==================================#
# Created by Ciaran O'Hare 2021

#==============================================================================#

from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from scipy.stats import norm
import PlotFuncs

#K_QCD = 6.743e-5 # GeV^4
K_QCD = 1.69e-5
def m2m1_ratio_hierarchical(f,N0,N1,N2):
    eps = 1e-6
    m1 = 1e9*sqrt(K_QCD/f**2*((N0+N1*eps**2)+(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5))
    m2 = 1e9*sqrt(K_QCD/f**2*((N0+N1*eps**2)-(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5))
    return (1/eps)*m2/m1

def m1m2_ratio_hierarchical(f,N0,N1,N2):
    eps = 1e6
    m1 = 1e9*sqrt(K_QCD/f**2*((N0+N1*eps**2)+(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5))
    m2 = 1e9*sqrt(K_QCD/f**2*((N0+N1*eps**2)-(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5))
    return (1/eps)*m1/m2

def Parameters(f,eps,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    '''
    Input:
    f and f' in GeV, must be the same size
    N's are O(1) numbers

    Output:
    dm_sq = squared mass difference [eV^2]
    m1 = heavier mass [eV]
    m2 = lighter mass [eV]
    tan_2alpha = mixing angle
    '''
    fp = f/eps

    N,Np,Ng,Ngp = AnomalyCoefficients[:]
    N0 = N**2+k*Ng**2
    N1 = Np**2 + k*Ngp**2
    N2 = N*Np + k*Ng*Ngp


    dm_sq = (1e9**2)*(2*K_QCD/f**2)*(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5
    m1 = 1e9*sqrt(K_QCD/f**2*((N0+N1*eps**2)+(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5))
    m2 = 1e9*sqrt(K_QCD/f**2*((N0+N1*eps**2)-(4*N2**2*eps**2 + (N0-N1*eps**2)**2)**0.5))

    m2_small1 = m1*eps*m2m1_ratio_hierarchical(f,N0,N1,N2)
    m2_small2 = (1/eps)*m1/(m1m2_ratio_hierarchical(f,N0,N1,N2))

    m2[f/fp<=1e-6] = m2_small1[f/fp<=1e-6]
    m2[fp/f<=1e-6] = m2_small2[fp/f<=1e-6]

    tan_2alpha = 2*eps*(N*Np+k*Ng*Ngp)/((N**2+k*Ng**2)-eps**2*(Np**2+k*Ngp**2))
    return dm_sq,m1,m2,tan_2alpha


def Couplings(f,eps,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    N,Np,Ng,Ngp = AnomalyCoefficients[:]

    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    alph = arctan(tan_2alpha)/2
    fp = f/eps
    g1 = (1/137)*(1/(2*pi))*1.92*(N*cos(alph)/f-Np*sin(alph)/fp)
    g2 = ((1/137)*(1/(2*pi))*1.92*(N*sin(alph)/f+Np*cos(alph)/fp))
    return m1,m2,g1,g2

def Mixing1(f,eps,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    alph = arctan(tan_2alpha)/2
    Mix1 = 4*cos(alph)**4*((-tan(alph)+eps*(1-tan(alph)**2)+eps**2*tan(alph))**2)/(1+eps**2)**2
    return Mix1

def Mixing2(f,eps,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2],L=1.496e11):
    dat = loadtxt('data/PrimakoffFlux_PlasmonCorrected.txt')
    w = dat[:,0]
    Phi = dat[:,1]
    Phi = Phi/trapz(Phi,w)

    L_eV = L/1.97e-7 # eV^-1
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    n = shape(f)[0]
    Mix2 = zeros_like(f)
    for i in range(0,n):
        for j in range(0,n):
            if dm_sq[i,j]*L_eV/(4*(10.0*1e3))>2*pi:
                Mix2[i,j] = 0.5
            else:
                Mix2[i,j] = trapz(sin(dm_sq[i,j]*L_eV/(4*(w*1e3)))**2*Phi,w)
    return Mix2

def MapLimit(file,Prob,m,g):
    m_lim,g_lim = loadtxt('limit_data/AxionPhoton/'+file+'.txt',unpack=True)

    ni = shape(m)[0]
    nj = shape(m)[1]
    g_lim_interp = zeros_like(m)
    for i in range(0,ni):
        for j in range(0,nj):
            g_lim_interp[i,j] = 10.0**interp(log10(m[i,j]),log10(m_lim),log10(g_lim))

    constrained = ((g*Prob)>g_lim_interp)
    constrained[m>amax(m_lim)] = False
    constrained[m<amin(m_lim)] = False
    return constrained


def MapHaloscope_m1(file,fvals,epsvals,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    f,eps = meshgrid(fvals,epsvals)
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    m1,m2,g1,g2 = Couplings(fvals,eps,k,AnomalyCoefficients)
    Omega_a1 = 1/(1+k**0.41*eps**(-7/6))
    lim_m1 = MapLimit(file,sqrt(Omega_a1),m1,g1)
    return lim_m1

def MapHaloscope_m2(file,fvals,epsvals,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    f,eps = meshgrid(fvals,epsvals)
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    m1,m2,g1,g2 = Couplings(fvals,eps,k,AnomalyCoefficients)
    Omega_a2 = 1/(1+k**-0.41*eps**(7/6))
    lim_m2 = MapLimit(file,sqrt(Omega_a2),m2,g2)
    return lim_m2


def MapHelioscope_m1(file,fvals,epsvals,n=100,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    f,eps = meshgrid(fvals,epsvals)
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    m1,m2,g1,g2 = Couplings(fvals,eps,k,AnomalyCoefficients)

    Mix2 = Mixing2(f,eps,k,AnomalyCoefficients)
    Mix1 = Mixing1(f,eps,k,AnomalyCoefficients)
    th = arcsin(sqrt(Mix1))/2
    SurvivalProb_active = 1-Mix1*Mix2
    Prob_m1 = SurvivalProb_active*cos(th)
    lim_m1 = MapLimit(file,(Prob_m1)**(1/4),m1,g1)
    return lim_m1


def MapHelioscope_m2(file,fvals,epsvals,n=100,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    f,eps = meshgrid(fvals,epsvals)
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    m1,m2,g1,g2 = Couplings(fvals,eps,k,AnomalyCoefficients)

    Mix2 = Mixing2(f,eps,k,AnomalyCoefficients)
    Mix1 = Mixing1(f,eps,k,AnomalyCoefficients)
    th = arcsin(sqrt(Mix1))/2
    SurvivalProb_active = 1-Mix1*Mix2
    Prob_m2 = SurvivalProb_active*sin(th)
    lim_m2 = MapLimit(file,(Prob_m2)**(1/4),m2,g2)
    return lim_m2

def Superradiance(file,fvals,epsvals,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2]):
    f,eps = meshgrid(fvals,epsvals)
    fp = f/eps
    dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
    md,fd = loadtxt('limit_data/fa/BlackHoleSpins_'+file+'.txt',unpack=True)
    ni = shape(f)[0]
    nj = shape(f)[1]
    constrained = zeros((ni,nj))
    for i in range(0,ni):
        for j in range(0,nj):
            m1_ij = m1[i,j]
            m2_ij = m2[i,j]
            if (m1_ij<amax(md)) and (m1_ij>amin(md)):
                g = interp(m1_ij,md,fd)
                constrained[i,j] = ((1/f[i,j]<g) and (1/fp[i,j]<g))
            if (m2_ij<amax(md)) and (m2_ij>amin(md)):
                g = interp(m2_ij,md,fd)
                constrained[i,j] += ((1/f[i,j]<g) and (1/fp[i,j]<g))
    constrained = constrained>0
    return constrained


def StellarCooling(ax,fvals,epsvals,text_pos=[5e6,1e-7],facecolor=PlotFuncs.HB_col,edgecolor='k',text_col='w',fs=30,rotation=90,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2],\
    edge_on=True,linestyle='-'):
    f,eps = meshgrid(fvals,epsvals)
    m1,m2,g1,g2 = Couplings(fvals,eps,k,AnomalyCoefficients)
    g_active = sqrt(g1**2+g2**2)
    HB = (g_active>6.6e-11)
    PlotFuncs.PlotContour(ax,fvals,epsvals,HB,zorder=0,alpha=1.0,lw=5,facecolor=facecolor,edgecolor=edgecolor,linestyle=linestyle,edge_on=edge_on)
    ax.text(text_pos[0],text_pos[1],r'{\bf Stellar cooling}',rotation=rotation,fontsize=fs)
    return

# def Superradiance(ax,fvals,epsvals,k=0.04,AnomalyCoefficients=[3,0.5,13/2,3/2],text_shift=[1,1],fs=25,\
#                             whichfile='Mehta',facecolor='gray',edgecolor='k',text_col='k',text_rot=51):
#     f,eps = meshgrid(fvals,epsvals)
#     fp = f/eps
#     dm_sq,m1,m2,tan_2alpha = Parameters(f,eps,k,AnomalyCoefficients)
#     md,fd = loadtxt('limit_data/fa/BlackHoleSpins_'+whichfile+'.txt',unpack=True)
#     ni = shape(f)[0]
#     nj = shape(f)[1]
#     constrained = zeros((ni,nj))
#     for i in range(0,ni):
#         for j in range(0,nj):
#             m1_ij = m1[i,j]
#             m2_ij = m2[i,j]
#             if (m1_ij<amax(md)) and (m1_ij>amin(md)):
#                 g = interp(m1_ij,md,fd)
#                 constrained[i,j] = ((1/f[i,j]<g) and (1/fp[i,j]<g))
#             if (m2_ij<amax(md)) and (m2_ij>amin(md)):
#                 g = interp(m2_ij,md,fd)
#                 constrained[i,j] += ((1/f[i,j]<g) and (1/fp[i,j]<g))
#
#     ax.contour(fvals,epsvals,constrained,levels=[0],linewidths=3,colors=edgecolor)
#     constrained[constrained==0] = nan
#     constrained[~isnan(constrained)] = 1.0
#     ax.contourf(fvals,epsvals,constrained,levels=[0,1],alpha=1,colors=facecolor)
#
#     text_pos1 = [3e12,4e-6]
#     ax.text(text_shift[0]*text_pos1[0],text_shift[1]*text_pos1[1],r'{\bf Black hole superradiance}',fontsize=fs,color=text_col,rotation=text_rot)
#     return
