#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:02:16 2025

@author: ppxam5
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as colors
from scipy.integrate import quad, quad_vec
from scipy.optimize import fsolve
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from Cavity_Functions import *
from Useful_Constants import *

params = {"ytick.color"      : "black",
          "xtick.color"      : "black",
          "axes.labelcolor"  : "black",
          "axes.edgecolor"   : "black",
          "text.usetex"      : True,
          "font.family"      : "serif",
          "font.serif"       : ["Computer Modern Serif"]}
plt.rcParams.update(params)



#%%

thresh = 1e-7
thresh_diff = 1e-2

def cavity_force(k, mphi, R2, R1, rho_cavity, t=None, k_earth=None, h=100.):
    """
    

    Parameters
    ----------
    k : float/array (complex for negative coupling)
        Characteristic length scale of field in cavity.
    mphi : float
        Dark energy’s fading, I feel you near, The universe slows, but your voice stays clear.
        mass of SF (eV)
    R2 : float
        Outer radius of cavity.
    R1 : float
        inner radius of cavity.
    rho_cavity : float
        density of cavity.
    t : optional, float
        Time of measurment. Give in seconds
        
    Should all be in SI units, conversion done inside.

    Returns
    -------
    float/array
    force

    """
    
    
    if type(mphi)==np.ndarray:        
        m = np.copy(mphi)
    else:
        m = mphi
    m *= 5.06e6
    #Converting mass to m
    
    phi_infty = np.sqrt(2 * rho_DM * 2.84e42)/m 
    #Converting to metres. m already converted. converting density from kgm^-3 to meters
    
    
    
    
    
    if t==None:
        cosine_term = 0.5
    else:
        cosine_term = np.cos(m*(c*t))**2
    
    if type(k_earth)==type(None):
        k_earth = k*np.sqrt(5510/rho_cavity)
        #Assume earth avg density ~ 5510
    
    a0, a1 = a0a1(k_earth, R_position=6.4e6+h)
    A0, c0, cm1, B0 = A0_c0cm1_B0(R2, R1, k, a0)
    A1, c1, cm2, B1 = A1_c1cm2_B1(R2, R1, k, a1)
    #Field profile parameters at ~ 100m above earth surface. Can be adjusted by including 
    #r parameter into a0a1 function
    
    force = -4/3*np.pi * k**2 * (R2**2 * (a0 + B0/R2)*(a1*R2 + B1/R2**2) - A0*A1 * R1**3)
    #Terms from varphi
    
    if type(k) == np.ndarray:
        mask= np.where(abs(k*R2)<thresh)
        force[mask] = -4/3 * np.pi * k[mask]**2 * a0[mask] * a1[mask] * (R2**3-R1**3)

    else:
        if abs(k*R2)<thresh:
            force = -4/3 * np.pi * k**2 * a0 * a1 * (R2**3-R1**3)

    force *= phi_infty**2 * cosine_term
    # Force on cavity, currently in units m, need to be converted to newtons
    
    force/=3.15e25
    #converting m^-2 to N
    
    return force.real




def cavity_force_point(k, mphi, R2, R1, rho_cavity, t=None, k_earth=None, h=100):
    """
    

    Parameters
    ----------
    k : float/array (complex for negative coupling)
        Characteristic length scale of field in cavity.
    mphi : float
        Dark energy’s fading, I feel you near, The universe slows, but your voice stays clear.
        mass of SF (eV)
    R2 : float
        Outer radius of cavity.
    R1 : float
        inner radius of cavity.
    rho_cavity : float
        density of cavity.
    t : optional, float
        Time of measurment. Give in seconds
        
    Should all be in SI units, conversion done inside.

    Returns
    -------
    None.

    """
    
    
    if type(mphi)==np.ndarray:        
        m = np.copy(mphi)
    else:
        m = mphi
    m *= 5.06e6
    #Converting mass from eV to m
    
    phi_infty = np.sqrt(2 * rho_DM * 2.84e42)/m 
    #Converting to metres. m already converted. converting density from kgm^-3 to meters
    
    cavity_mass = 4/3 * np.pi * (R2**3 - R1**3) * rho_cavity * 2.84e42
    # Converting kg to metres
    
    alpha_cavity = k**2/(rho_cavity * 2.84e42)
    #Calculating alpha, with rho_cavity converted kgm^-3 to metres
    
    if t==None:
        cosine_term = 0.5
    else:
        cosine_term = np.cos(m*(c*t))**2
    
    if type(k_earth)==type(None):
        k_earth = k*np.sqrt(5510/rho_cavity)
        #Assume earth avg density ~ 5510
    
    a0, a1 = a0a1(k_earth, R_position=6.4e6+h)
    #Field profile parameters at ~ 100m above earth surface. Can be adjusted by including 
    
    force = - alpha_cavity * cavity_mass * a0 * a1 * phi_infty**2 * cosine_term
    #Terms from varphi

    # Force on cavity, currently in units m, need to be converted to newtons
    
    force/=3.15e25
    #converting m^-2 to N
    
    return force.real




def differential_force_same_mass_etc(M, mphi, R2, R1_1, R1_2, rho_cavity_1, t=None, h=100):
    #This is calculating cavity 1 force - cavity 2 force

    #Incase case with different alphas is to be tested. Factor = alpha2/alpha1
    factor = 1

    #Density of second cavity    
    rho_cavity_2 = rho_cavity_1 * (R2**3-R1_1**3)/(R2**3-R1_2**3)

    k_earth = np.sqrt(5510 * density_to_GeV4) / (M/length_to_GeVm1)
    k_1     = np.sqrt(rho_cavity_1 * density_to_GeV4) / (M/length_to_GeVm1)
    k_2     = np.sqrt(factor) * np.sqrt(rho_cavity_2 * density_to_GeV4) / (M/length_to_GeVm1)
    
    #Method for general alpha, using predefined function
    diff_force = abs(cavity_force(k_1, mphi, R2, R1_1, rho_cavity_1, t=None, k_earth=k_earth, h=h) - cavity_force(k_2, mphi, R2, R1_2, rho_cavity_2, t=None, k_earth=k_earth, h=h))



    mask1 = (abs(k_1*R2_1)<thresh_diff)
    mask2 = (abs(k_2*R2_2)<thresh_diff)

    mask = (mask1 & mask2)

    u2_1 =  (k_1*R2)[mask]      
    u1_1 =  (k_1*R1_1)[mask]
    u2_2 =  (k_2*R2)[mask]
    u1_2 =  (k_2*R1_2)[mask]

    Delta_u_1 = u2_1-u1_1      
    Delta_u_2 = u2_2-u1_2      
    

    a0, a1 = a0a1(k_earth[mask], R_position=6.4e6+h)

    
    #put all quantities into units: metres.    

    
    mass_cavity = 4/3 * np.pi * (R2**3 - R1_1**3) * (rho_cavity_1) * 2.84e42
    #Converting kg to metres    
    
    M_masked = M[mask] * 5.06e15
    #Converting GeV to metres

    m = np.copy(mphi[mask]) * 5.06e6
    #Converting eV to metres

    phi_infty = np.sqrt(2 * rho_DM * 2.84e42)/m 
    #in units of meters. Converting rho_DM from kg m^-3. m is already converted



    
    force_term_0 = mass_cavity/M_masked**2 * (1 - factor) 
    
    force_term_1_2 =  - 4 * np.pi/3 *   0.2 * (R2-R1_2)**2 * (3*R1_2**3 + 5*R2*R1_2**2 + 4*R2**2*R1_2 * 2*R2**3)
    force_term_1_1 =  - 4 * np.pi/3 *   0.2 * (R2-R1_1)**2 * (3*R1_1**3 + 5*R2*R1_1**2 + 4*R2**2*R1_1 * 2*R2**3)

    force_term_1 = ( force_term_1_1 * k_1[mask]**4  - force_term_1_2 * k_2[mask]**4)
            
    
    
    
    
    if t==None:
        cosine_term = 0.5
    else:
        cosine_term = np.cos(m*(c*t))**2
        

        
    approx_diff_force = -1 * a0 * a1 * (force_term_0 + force_term_1)

    approx_diff_force *= phi_infty**2 * cosine_term
    # Force on cavity, currently in units m, need to be converted to newtons
    
    approx_diff_force/=3.15e25
    #Converting from units meters to units newtons

    diff_force[mask] = abs(approx_diff_force.real)
    
    return diff_force
    

#%% Single Cavity Force Mesh Plot

if True:
    
    rho_cavity = 2700
    
    R2 = .12
    R1 = .10
    
    
    Nk = 1001
    Nm = 1001
    
    M_array = np.logspace(0,20,Nk)
    k_array = np.sqrt(rho_cavity * density_to_GeV4) / (M_array/length_to_GeVm1)
    m_array = np.logspace(-25,0,Nm)
    
    alpha_array = 1/M_array**2
    
    kk, mm = np.meshgrid(k_array, m_array)
    
    
    point_mass = False
    #set to true to calcualte force on a point mass, rather than finite cavity
    if point_mass:
        FF = cavity_force_point(kk, mm, R2, R1, rho_cavity)
        FF_negative = cavity_force_point(1.j*kk, mm, R2, R1, rho_cavity)
    else:    
        FF = cavity_force(kk, mm, R2, R1, rho_cavity)
        FF_negative = cavity_force(1.j*kk, mm, R2, R1, rho_cavity)
    
    
    alpha_line = (m_array*1e-9)**2/(rho_cavity*density_to_GeV4)
    
    
    
    k_E = 1/R_earth
    k_Cavity = 1/(R2-R1)
    
    M_Earth = np.sqrt(5510 * density_to_GeV4) / (k_E/length_to_GeVm1)
    M_Cavity = np.sqrt(rho_cavity * density_to_GeV4) / (k_Cavity/length_to_GeVm1)
    
    alpha_Earth = 1/M_Earth**2
    alpha_Cavity = 1/M_Cavity**2
    
    alpha_scales = [alpha_Earth, alpha_Cavity]
    
    plt.close("1Force Mesh Plot")
    fig, ax = plt.subplots(1,2,num="1Force Mesh Plot", sharex=True, sharey=True, layout="constrained", figsize=[7.5,3.5])
    
    norm = colors.LogNorm(vmin = 1e-15, vmax=1e15)
    cbar_ticks = np.logspace(-15, 15,7)
    xticks = np.logspace(-40,0,6)
    yticks = np.logspace(-25,0,6)
    
    mesh0 = ax[0].pcolormesh(alpha_array,m_array, abs(FF)/1e-9, norm=norm, cmap="coolwarm")
    ax[0].contour(alpha_array, m_array, abs(FF), norm=colors.LogNorm(), colors=["black"], alpha=0.5)
    
    mesh1 = ax[1].pcolormesh(alpha_array,m_array, abs(FF_negative)/1e-9, norm=norm, cmap = "coolwarm")
    ax[1].contour(alpha_array, m_array, abs(FF_negative), norm=colors.LogNorm(), colors=["black"], alpha=0.5)
    
    # Axis Limits
    ax[0].set_xlim(np.min(alpha_array), np.max(alpha_array))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_box_aspect(1)
    
    ax[1].set_xlim(np.min(alpha_array), np.max(alpha_array))
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_box_aspect(1)
    
    #Colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="coolwarm"),  extend="both", ax=ax, shrink=0.73, ticks = cbar_ticks)
    cb.set_label(label=r"$\left|\vec{F_5}\right|$ (nN)", fontsize=14, rotation=90)
    cb.ax.tick_params(labelsize=12)
    
    for axis in ax.flatten():
        axis.tick_params(labelsize=12)
        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
    
    
    #Titles
    ax[0].set_title(r"$\alpha>0$", fontsize=16)
    ax[1].set_title(r"$\alpha<0$", fontsize=16)
    
    #Axis Labels
    fig.supylabel("$m$ (eV)",fontsize=16)
    fig.supxlabel(r"$|\alpha|$ (GeV$^{-2}$)",fontsize=16,y=-0.004)
    
    
    alpha_labels = [r"$\alpha=\frac{k_c^2}{\rho_c}$",r"$\alpha=\frac{k_E^2}{\rho_\oplus}$"]
    alpha_linestyles = ["--",":"]
    for axis in ax.flatten():
        for i, alpha in enumerate(alpha_scales):
            axis.plot([alpha, alpha], [np.min(m_array), np.max(m_array)], alpha_linestyles[i], c="black", alpha = 0.7, label = alpha_labels[i])
      
    
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels,fontsize=14, bbox_to_anchor = (0, 0, 0.865, 1), markerscale=1)




#%% Differential Force
R2_1 = .11
R1_1 = .1
rho_cavity_1 = 2700

force_centre = 1e-13

R2_2 = .11
R1_2 = .09
rho_cavity_2 = (R2_1**3 - R1_1**3) * rho_cavity_1 / (R2_2**3 - R1_2**3)


Nk = 1001
Nm = 1001

M_array = np.logspace(0,20,Nk)
k_array_1 = np.sqrt(rho_cavity_1 * density_to_GeV4) / (M_array/length_to_GeVm1)
k_array_2 = np.sqrt(rho_cavity_2 * density_to_GeV4) / (M_array/length_to_GeVm1)
alpha_array = 1/M_array**2

m_array = np.logspace(-25,-5,Nm)


mm, kk_1 = np.meshgrid(m_array, k_array_1)
mm, kk_2 = np.meshgrid(m_array, k_array_2)
mm, MM = np.meshgrid(m_array, M_array)


norm = colors.LogNorm(vmin = 1e-15, vmax=1e15)
cbar_ticks = np.logspace(-15, 15,7)
yticks = np.logspace(-40,0,6)


yticks_negative = ["$-10^{%.i}$"%i for i in np.log10(yticks)]


xticks = np.logspace(-25,-5,5)


k_E = 1/R_earth
k_Cavity = 1/(R2_1-R1_1)

M_E = np.sqrt(5510 * density_to_GeV4) / (k_E/length_to_GeVm1) #Important, density of Earth should be used here...
M_Cavity_1 = np.sqrt(rho_cavity_1 * density_to_GeV4) / (k_Cavity/length_to_GeVm1)
M_Cavity_2 = np.sqrt(rho_cavity_2 * density_to_GeV4) / (k_Cavity/length_to_GeVm1)


alpha_E = 1/M_E**2
alpha_Cavity_1 = 1/M_Cavity_1**2
alpha_Cavity_2 = 1/M_Cavity_2**2

h=100
delta_FF            = differential_force_same_mass_etc(MM,     mm, R2_1, R1_1, R1_2, rho_cavity_1, t=None, h=h)
delta_FF_negative   = differential_force_same_mass_etc(-1j*MM, mm, R2_1, R1_1, R1_2, rho_cavity_1, t=None, h=h)


plt.close("Differential Force Mesh Plot")
fig, ax = plt.subplots(2,1,num="Differential Force Mesh Plot", sharex=True, sharey=False, layout="constrained", dpi=200, figsize=[5,4])

fig.suptitle("$h=100$ m")


#Plotting Mesh/Contour
norm2 = norm
mesh0 = ax[0].pcolormesh(m_array, alpha_array, abs(delta_FF)/force_centre, norm=norm2, cmap="coolwarm")
ax[0].contour(m_array, alpha_array, np.log10(abs(delta_FF)/force_centre),  linestyles=["solid"], colors=["black"], linewidths=[0.6], alpha=0.5, levels=[-45,-15,0,15])

mesh1 = ax[1].pcolormesh(m_array, alpha_array, abs(delta_FF_negative)/force_centre, norm=norm2, cmap="coolwarm")
ax[1].contour(m_array, alpha_array, np.log10(abs(delta_FF_negative)/force_centre),  linestyles=["solid"], colors=["black"], linewidths=[0.6], alpha=0.5, levels=[-45,-15,0,15])

# Axis Limits
ax[0].set_xlim(np.min(m_array), np.max(m_array))
ax[0].set_ylim(np.min(alpha_array), np.max(alpha_array))
ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax[1].set_xlim(np.min(m_array), np.max(m_array))
ax[1].set_ylim(np.min(alpha_array), np.max(alpha_array))
ax[1].set_xscale("log")
ax[1].set_yscale("log")

#Colorbar
cb = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap="coolwarm"),  extend="both", ax=ax, shrink=0.9, ticks = cbar_ticks)
cb.set_label(label=r"$\left|\Delta \vec{F_5}\right|$ ($100$ fN)", rotation=90)


for axis in ax.flatten():
    axis.set_yticks(yticks)
    axis.set_xticks(xticks)

ax[1].set_yticklabels(yticks_negative)    
ax[1].invert_yaxis()


#Axis Labels
ax[1].set_xlabel("$m$ (eV)",fontsize=14)
ax[0].set_ylabel(r"$\alpha~\left(\mbox{GeV}^{-2}\right)$",fontsize=14, y=0, labelpad=10)



r_inside_1 = np.linspace(R1_1, R2_1, 1001)
r_inside_2 = np.linspace(R1_2, R2_2, 1001)
r_inside_earth = np.linspace(0, R_earth, 1001)



alpha_line = (m_array*1e-9)**2/(2*rho_DM_GeV)

if True:
    N_points_line = 10001

    alpha_array_line = np.logspace(np.log10(np.min(alpha_array)), np.log10(np.max(alpha_array)), N_points_line)
    k_array_1_line = np.logspace(np.log10(np.min(k_array_1)), np.log10(np.max(k_array_1)), N_points_line)
    k_array_2_line = np.logspace(np.log10(np.min(k_array_2)), np.log10(np.max(k_array_2)), N_points_line)

    varphi_max_array_line = np.zeros(len(k_array_1_line))

    a0, a1 = a0a1(np.sqrt(5.51/2.7)*1j*k_array_1_line, R_position=R_earth+100, R_earth=R_earth)

    
    for i in range(len(k_array_1_line)):
        if i%10==0:
            print(i)
        k1 = k_array_1_line[i]
        k2 = k_array_2_line[i]
        ke = np.sqrt(5.51/2.7)*k1 
        
        N_points_1 = int(k1 * (R2_1 - R1_1)+5)
        N_points_2 = int(k2 * (R2_1 - R1_1)+5)
        N_points_e = int(ke * (  R_earth  )+5)   
        
        if True:
        
            varphi_inside_1 = get_wall_profile(R2_1,    R1_1, k1*1j, a0=a0[i], a1=a1[i], double_sided=True , N_points=N_points_1)
            varphi_inside_2 = get_wall_profile(R2_2,    R1_2, k2*1j, a0=a0[i], a1=a1[i], double_sided=True , N_points=N_points_2)
            varphi_earth    = np.array([1/np.cos(ke*R_earth)])
            
            
            abs_varphi_array = np.zeros(2*N_points_1 + 2*N_points_2 + 1)
            abs_varphi_array[:2*N_points_1] = abs(varphi_inside_1)
            abs_varphi_array[2*N_points_1:2*N_points_1+2*N_points_2] = abs(varphi_inside_2)
            abs_varphi_array[-1] = abs(varphi_earth)
    
            
            varphi_max_array_line[i] = np.nanmax(abs_varphi_array)
        else:
            varphi_max_array_line[i] = abs(1/np.cos(ke*R_earth))
        
    alpha_array_line = np.logspace(np.log10(np.min(alpha_array)), np.log10(np.max(alpha_array)), 10001)
    m_line = np.sqrt((2*rho_DM_GeV) * varphi_max_array_line**2 * alpha_array_line) * 1e9
    
    ax[0].fill_betweenx(alpha_line, m_array, 1e-40, ls="--", color="grey", alpha=0.8, hatch=r"x", lw=0.2)
    ax[1].fill_betweenx(alpha_array_line,m_line, 1e-40, ls="--", color="grey", alpha=0.8, hatch=r"x", lw=0.2)

else:
    ax[0].fill_betweenx(alpha_line,m_array, 1e-40, ls="--", color="grey", alpha=0.8, hatch=r"x", lw=0.2)
    ax[1].fill_betweenx(alpha_line,m_array, 1e-40, ls="--", color="grey", alpha=0.8, hatch=r"x", lw=0.2)
    
    
    

for axis in ax.flatten():
     axis.plot([1e-25,1], alpha_Cavity_1*np.ones(2), ":", c="black", alpha=0.6, lw=1) #plottingh lengthscale 1
     axis.plot([1e-25,1], alpha_E*np.ones(2), "--", c="black", alpha=0.6, lw=1)   #plotting lengthscale 2
    

fig.savefig(r"Differential_Force.png", dpi=300)

#%% Differential Force - New Formatting, Multiple Heights
R2_1 = .11
R1_1 = .1
rho_cavity_1 = 2700

force_centre = 1e-13

R2_2 = .11
R1_2 = .09
rho_cavity_2 = (R2_1**3 - R1_1**3) * rho_cavity_1 / (R2_2**3 - R1_2**3)


Nk = 1001
Nm = 1001

M_array = np.logspace(0,20,Nk)
k_array_1 = np.sqrt(rho_cavity_1 * density_to_GeV4) / (M_array/length_to_GeVm1)
k_array_2 = np.sqrt(rho_cavity_2 * density_to_GeV4) / (M_array/length_to_GeVm1)
alpha_array = 1/M_array**2

m_array = np.logspace(-25,-5,Nm)


mm, kk_1 = np.meshgrid(m_array, k_array_1)
mm, kk_2 = np.meshgrid(m_array, k_array_2)
mm, MM = np.meshgrid(m_array, M_array)


norm = colors.LogNorm(vmin = 1e-15, vmax=1e15)
norm_logged = colors.Normalize(vmin = -15, vmax=15)

levels = np.arange(-60,30.1,7.5)
number = np.where(levels==0)[0][0]
cmap_to_use = "coolwarm"
cbar_ticks = np.logspace(-15, 15,7)
yticks = np.logspace(-40,0,6)
yticks_negative = ["$-10^{%.i}$"%i for i in np.log10(yticks)]


xticks = np.logspace(-25,-5,5)



k_E = 1/R_earth
k_Cavity = 1/(R2_1-R1_1)

M_E = np.sqrt(5510 * density_to_GeV4) / (k_E/length_to_GeVm1) #Important, density of Earth should be used here...
M_Cavity_1 = np.sqrt(rho_cavity_1 * density_to_GeV4) / (k_Cavity/length_to_GeVm1)
M_Cavity_2 = np.sqrt(rho_cavity_2 * density_to_GeV4) / (k_Cavity/length_to_GeVm1)


alpha_E = 1/M_E**2
alpha_Cavity_1 = 1/M_Cavity_1**2
alpha_Cavity_2 = 1/M_Cavity_2**2

h_array = [1e6, 1e4, 1e2]
color_array = [ "tan", "moccasin", "papayawhip"]

plt.close("Differential Force Mesh Plot")
fig, ax = plt.subplots(2,1,num="Differential Force Mesh Plot", sharex=True, sharey=False, layout="constrained", dpi=200, figsize=[4,3.35])

prev_y_positive=[1e0,1e-40]
prev_x_positive=[1e-25,1e-25]

prev_y_negative=[1e0,1e-40]
prev_x_negative=[1e-25,1e-25]


for i, h in enumerate(h_array):
    delta_FF            = differential_force_same_mass_etc(MM,     mm, R2_1, R1_1, R1_2, rho_cavity_1, t=None, h=h)
    delta_FF_negative   = differential_force_same_mass_etc(-1j*MM, mm, R2_1, R1_1, R1_2, rho_cavity_1, t=None, h=h)
    
    
    #Plotting Mesh/Contour
    norm2 = norm
    contours_positive = ax[0].contour(m_array, alpha_array, np.log10(abs(delta_FF)/force_centre),  linestyles=["solid"], linewidths=[0.6], norm=norm_logged, colors=["black"], levels=[0])
    
    contours_negative = ax[1].contour(m_array, alpha_array, np.log10(abs(delta_FF_negative)/force_centre),  linestyles=["solid"], linewidths=[0.6], norm=norm_logged, colors=["black"], levels=[ 0])
    
    points_positive = contours_positive.get_paths()[0]
    vertices_positive = points_positive.vertices
    x_positive = vertices_positive[:,0]
    y_positive = vertices_positive[:,1]
    
    if h >= 1e3:
        label =r"$h= %.i$ km"%(h/1e3)
    else:
        label = r"$h= %.i$ m"%(h)
    ax[0].fill_betweenx(y_positive, x_positive, 1e-25, alpha=1, color=color_array[i], label=label)
    prev_y_positive = y_positive
    prev_x_positive = x_positive

    
    points_negative = contours_negative.get_paths()[0]
    vertices_negative = points_negative.vertices
    x_negative = vertices_negative[:,0]
    y_negative = vertices_negative[:,1]
    
    ax[1].fill_betweenx(y_negative, x_negative, 1e-25, alpha=1, color=color_array[i])
    prev_y_negative = y_negative
    prev_x_negative = x_negative

    
ax[0].legend(fontsize=7.5)
# Axis Limits
ax[0].set_xlim(np.min(m_array), np.max(m_array))
ax[0].set_ylim(np.min(alpha_array), np.max(alpha_array))
ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax[1].set_xlim(np.min(m_array), np.max(m_array))
ax[1].set_ylim(np.min(alpha_array), np.max(alpha_array))
ax[1].set_xscale("log")
ax[1].set_yscale("log")

for axis in ax.flatten():
    axis.set_yticks(yticks)
    axis.set_xticks(xticks)

ax[1].set_yticklabels(yticks_negative)    
ax[1].invert_yaxis()



#Axis Labels
ax[1].set_xlabel("$m$ (eV)",fontsize=14)
ax[0].set_ylabel(r"$\alpha~\left(\mbox{GeV}^{-2}\right)$",fontsize=14, y=0, labelpad=10)


##
de_positive_non_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive de Non-microscope Overview.csv", delimiter=",")
de_positive_microscope = np.loadtxt(r"Hees Constraints Microscope Separated/Positive de Microscope Overview.csv", delimiter=",")

de_negative = np.loadtxt(r"Hees Constraints/Negative de Overview.csv", delimiter=",")

iron_Qs = getQs(56,26)
Qe = iron_Qs["Q_e"]
de_to_alpha = Qe * (4*np.pi*G_GeV)

de_positive_m = np.append(de_positive_non_microscope[:,0], de_positive_microscope[:,0])
de_positive_de = np.append(de_positive_non_microscope[:,1], de_positive_microscope[:,1])
de_positive_alpha = de_positive_de * de_to_alpha

ax[0].plot(de_positive_m, de_positive_alpha, c="black", ls=":", alpha=0.6)
##



#plotting alpha for which phi^2alpha = 1
r_inside_1 = np.linspace(R1_1, R2_1, 1001)
r_inside_2 = np.linspace(R1_2, R2_2, 1001)
r_inside_earth = np.linspace(0, R_earth, 1001)



alpha_line = (m_array*1e-9)**2/(2*rho_DM_GeV)

#Use false for simplified line (faster to compute)
if True:
    N_points_line = 10001

    alpha_array_line = np.logspace(np.log10(np.min(alpha_array)), np.log10(np.max(alpha_array)), N_points_line)
    k_array_1_line = np.logspace(np.log10(np.min(k_array_1)), np.log10(np.max(k_array_1)), N_points_line)
    k_array_2_line = np.logspace(np.log10(np.min(k_array_2)), np.log10(np.max(k_array_2)), N_points_line)

    varphi_max_array_line = np.zeros(len(k_array_1_line))

    a0, a1 = a0a1(np.sqrt(5.51/2.7)*1j*k_array_1_line, R_position=R_earth+100, R_earth=R_earth)

    
    for i in range(len(k_array_1_line)):
        if i%10==0:
            print(i)
        k1 = k_array_1_line[i]
        k2 = k_array_2_line[i]
        ke = np.sqrt(5.51/2.7)*k1 
        
        N_points_1 = int(k1 * (R2_1 - R1_1)+5)
        N_points_2 = int(k2 * (R2_1 - R1_1)+5)
        N_points_e = int(ke * (  R_earth  )+5)   
        
        if True:
        
            varphi_inside_1 = get_wall_profile(R2_1,    R1_1, k1*1j, a0=a0[i], a1=a1[i], double_sided=True , N_points=N_points_1)
            varphi_inside_2 = get_wall_profile(R2_2,    R1_2, k2*1j, a0=a0[i], a1=a1[i], double_sided=True , N_points=N_points_2)
            varphi_earth    = np.array([1/np.cos(ke*R_earth)])
            
            
            abs_varphi_array = np.zeros(2*N_points_1 + 2*N_points_2 + 1)
            abs_varphi_array[:2*N_points_1] = abs(varphi_inside_1)
            abs_varphi_array[2*N_points_1:2*N_points_1+2*N_points_2] = abs(varphi_inside_2)
            abs_varphi_array[-1] = abs(varphi_earth)
    
            varphi_max_array_line[i] = np.nanmax(abs_varphi_array)
        else:
            varphi_max_array_line[i] = abs(1/np.cos(ke*R_earth))
        
    alpha_array_line = np.logspace(np.log10(np.min(alpha_array)), np.log10(np.max(alpha_array)), 10001)
    m_line = np.sqrt((2*rho_DM_GeV) * varphi_max_array_line**2 * alpha_array_line) * 1e9
    
    ax[0].fill_betweenx(alpha_line, m_array, 1e-40, ls="--", color="grey", alpha=0.6, hatch=r"x", lw=0.2)
    ax[1].fill_betweenx(alpha_array_line,m_line, 1e-40, ls="--", color="grey", alpha=0.6, hatch=r"x", lw=0.2)

else:
    ax[0].fill_betweenx(alpha_line,m_array, 1e-40, ls="--", color="grey", alpha=0.8, hatch=r"x", lw=0.2)
    ax[1].fill_betweenx(alpha_line,m_array, 1e-40, ls="--", color="grey", alpha=0.8, hatch=r"x", lw=0.2)
    
    
    

for axis in ax.flatten():
     axis.plot([1e-25,1], alpha_Cavity_1*np.ones(2), "--", c="tab:blue", alpha=0.6, lw=1) #plottingh lengthscale 1
     axis.plot([1e-25,1], alpha_E*np.ones(2), "-.", c="tab:blue", alpha=0.6, lw=1)   #plotting lengthscale 2
    

fig.savefig(r"Differential_Force_Mesh_New_Format_Multiple_Heights.png", dpi=400)



