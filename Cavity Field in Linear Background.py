#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:34:49 2025

@author: ppxam5
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from Cavity_Functions import *
from Useful_Constants import *


params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)
#%%

plot_Surface = False
plot_Mesh = False
plot_Test = False
plot_linear_profile_Test = False
plot_constant_profile_Test = False
plot_c_Test = False
plot_profile_comparison = False
plot_lengthscale_validitiy_Test = True
plot_Interior_parameters = False
plot_sphere_surface_value = True

R2 = .11
R1 = .1

M_cavity = 4 * np.pi/3 * (.11**3 - 0.1**3) * 2700
rho_2 = M_cavity/(4 * np.pi/3 * (.11**3 - 0.09**3))


k=5e2
lengthscale = abs(1/k)


Xmin, Xmax = -350*lengthscale, 350*lengthscale
Ymin, Ymax = -350*lengthscale, 350*lengthscale


k_earth = k*np.sqrt(5.51/2.7)
#Earth vs Aluminium density

a0, a1 = a0a1(k_earth)        


A0, c0, cm1, B0 = A0_c0cm1_B0(R2, R1, k, a0)
A1, c1, cm2, B1 = A1_c1cm2_B1(R2, R1, k, a1)

x_array = np.linspace(Xmin,Xmax, 1000)
y_array = np.linspace(Ymin,Ymax, 1000)

xx, yy = np.meshgrid(x_array, y_array)

r_grid = np.sqrt(xx**2 + yy**2)

thetas_grid = np.arctan2(yy,xx)

values_grid = np.zeros(r_grid.shape, dtype=complex)
values_grid[:,:] = np.nan

inside_mask = r_grid<R1
values_grid[inside_mask] = A0


wall_mask = (r_grid<R2) & (r_grid>R1)
values_grid[wall_mask] = c0*f(r_grid[wall_mask], 0, k)[0] + cm1*f(r_grid[wall_mask], -1, k)[0]


outside_mask = (r_grid>R2)
values_grid[outside_mask] = a0 + B0/r_grid[outside_mask]




inside_mask = r_grid<R1
values_grid[inside_mask] += (A1*r_grid[inside_mask]*np.cos(thetas_grid[inside_mask])).real


wall_mask = (r_grid<R2) & (r_grid>R1)
values_grid[wall_mask] += (c1*f(r_grid[wall_mask], 1, k)[0] + cm2*f(r_grid[wall_mask], -2, k)[0])*np.cos(thetas_grid[wall_mask])


outside_mask = (r_grid>R2)
values_grid[outside_mask] += (a1*r_grid[outside_mask] + B1/r_grid[outside_mask]**2).real*np.cos(thetas_grid[outside_mask])






k_E = 1/R_earth
alpha_E = (k_E/length_to_GeVm1)**2 /(5510*density_to_GeV4)

k_C = 1/(R2-R1)
alpha_C = (k_C/length_to_GeVm1)**2 /(2700*density_to_GeV4)



    
if plot_Surface:
    plt.close("surface")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, num="surface")
    
    surf = ax.plot_surface(xx, yy, values_grid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    theta_arr = np.linspace(0,2*np.pi)
    
    plt.colorbar(surf,)

if plot_Mesh:
    plt.close("Heatmap")
    fig, ax = plt.subplots(1,1, num="Heatmap")
    mesh = ax.pcolormesh(xx/lengthscale,yy/lengthscale, values_grid, cmap="rainbow", vmin=0, vmax=1)
    ax.set_aspect('equal', 'box')
    plt.colorbar(mesh)
    
    circle1x, circle1y = R1 * np.cos(theta_arr), R1 * np.sin(theta_arr)
    circle2x, circle2y = R2 * np.cos(theta_arr), R2 * np.sin(theta_arr)
    
    ax.plot(circle1x/lengthscale, circle1y/lengthscale, c="black", lw=0.3)
    ax.plot(circle2x/lengthscale, circle2y/lengthscale, c="black", lw=0.3)

if plot_Test:
    plt.close("Test1")
    r = np.linspace(0,2,100)
    
    fig_test, ax_test = plt.subplots(1,5,num="Test1", sharex=True, layout="compressed")

    ax_test[0].plot(r,f(k*r,-2,1)[0],"b-",lw=2, alpha=0.1)
    ax_test[0].plot(r, -np.cosh(k*r)/(k*r)**2+np.sinh(k*r)/(k*r),"r--", alpha=0.1)
    ax_test[0].set_xlim(0,2)
    ax_test[0].set_title("p=-2")
    
    ax_test[1].plot(r,f(k*r,-1,1)[0],"b-",lw=2)
    ax_test[1].plot(r, np.cosh(k*r)/(k*r),"r--")
    ax_test[1].set_xlim(0,2)
    ax_test[1].set_title("p=-1")
    
    ax_test[2].plot(r,f(k*r,0,1)[0],"b-",lw=2)
    ax_test[2].plot(r, np.sinh(k*r)/(k*r),"r--")
    ax_test[2].set_xlim(0,2)
    ax_test[2].set_title("p=0")
    
    ax_test[3].plot(r,f(k*r,1,1)[0],"b-",lw=2, alpha=0.1)
    ax_test[3].plot(r, np.sinh(k*r)/(k*r)**2-np.cosh(k*r)/(k*r),"r--", alpha=0.1)
    ax_test[3].set_xlim(0,2)
    ax_test[3].set_title("p=1")
    

    ax_test[4].plot(r, f(k*r,2,1)[0],"b-",lw=2, alpha=0.5)    
    ax_test[4].plot(r, np.sinh(k*r)*((k*r)**2+3)/(k*r)**3-3*np.cosh(k*r)/(k*r)**2,"r--", alpha=0.5)
    ax_test[4].set_title("p=2")
    
    
if plot_linear_profile_Test:
    r_inside = np.linspace(0,R1,1001)
    r_wall = np.linspace(R1,R2,1001)
    r_out = np.linspace(R2,2*R2, 1001)
    
    plt.close("Linear Profile Test")
    fig_linear, ax_linear = plt.subplots(2,1,num="Linear Profile Test", sharex=True)
    fig_linear.suptitle("l=1 Component of Field Profile")
    
    A1_func(R2, R1, k, a1)
    
    ax_linear[0].plot(r_inside, A1_func(R2, R1, k, a1)*r_inside)
    ax_linear[0].plot(r_wall, c1_func(R2, R1, k, a1)*f(r_wall, 1, k)[0] + cm2_func(R2, R1, k, a1)*f(r_wall, -2, k)[0])
    ax_linear[0].plot(r_out, a1*r_out + B1/r_out**2)

    f(r_wall, 1, k)[0]

    ax_linear[1].plot(r_inside, A1_func(R2, R1, k, a1)*np.ones_like(r_inside))
    ax_linear[1].plot(r_wall, c1_func(R2, R1, k, a1)*f(r_wall, 1, k)[1] + cm2_func(R2, R1, k, a1)*f(r_wall, -2, k)[1])
    ax_linear[1].plot(r_out, a1 - 2*B1/r_out**3)

if plot_constant_profile_Test:
    r_inside = np.linspace(0,R1,1001)
    r_wall = np.linspace(R1,R2,1001)
    r_out = np.linspace(R2,2*R2, 1001)
    
    plt.close("Constant Profile Test")
    fig_constant, ax_constant = plt.subplots(2,1,num="Constant Profile Test", sharex=True)
    fig_constant.suptitle("$l=0$ Component of Field Profile")
    ax_constant[0].plot(r_inside, A0*np.ones_like(r_inside))
    ax_constant[0].plot(r_wall, c0*f(r_wall, 0, k, a0)[0] + cm1*f(r_wall, -1, k)[0])
    ax_constant[0].plot(r_out, a0 + B0/r_out)

    ax_constant[1].plot(r_inside, np.zeros_like(r_inside))
    ax_constant[1].plot(r_wall, c0*f(r_wall, 0, k)[1] + cm1*f(r_wall, -1, k)[1])
    ax_constant[1].plot(r_out,  - B0/r_out**2)
    
    ax_constant[1].set_xlabel("r (m)")
    fig_constant.tight_layout()

if plot_profile_comparison:
    a0_plus, a1_plus = a0a1(abs(k_earth))
    a0_minus, a1_minus = a0a1(abs(k_earth)*1j)

    A0_plus, c0_plus, cm1_plus, B0_plus =  A0_c0cm1_B0(R2, R1, abs(k), a0_plus)
    A1_plus, c1_plus, cm2_plus, B1_plus = A1_c1cm2_B1(R2, R1, abs(k), a1_plus)

    A0_minus, c0_minus, cm1_minus, B0_minus = A0_c0cm1_B0(R2, R1, abs(k)*1j, a0_minus)
    A1_minus, c1_minus, cm2_minus, B1_minus = A1_c1cm2_B1(R2, R1, abs(k)*1j, a1_minus)

    r_inside = np.linspace(0,R1,1001)
    r_wall = np.linspace(R1,R2,1001)
    r_out = np.linspace(R2,2*R2, 1001)    

    
    plt.close("Plus/Minus Profile Test")
    fig_pm, ax_pm = plt.subplots(1,1,num="Plus/Minus Profile Test", sharex=True)
    fig_pm.suptitle("Field Profile")
    
    #Positive r    
    ax_pm.plot(r_inside, A0_plus * np.ones_like(r_inside) + A1_plus * r_inside, "b")
    ax_pm.plot(r_wall, c0_plus * f(r_wall, 0, abs(k))[0] + cm1_plus * f(r_wall, -1, abs(k))[0] + c1_plus * f(r_wall, 1, abs(k))[0] + cm2_plus * f(r_wall, -2, abs(k))[0], "b")
    ax_pm.plot(r_out, a0_plus + B0_plus/r_out + a1_plus * r_out + B1_plus/r_out**2, "b")

    ax_pm.plot(r_inside, A0_minus * np.ones_like(r_inside) + A1_minus * r_inside, "r")
    ax_pm.plot(r_wall, c0_minus * f(r_wall, 0, abs(k)*1j)[0] + cm1_minus * f(r_wall, -1, abs(k)*1j)[0] + c1_minus * f(r_wall, 1, abs(k)*1j)[0] + cm2_minus * f(r_wall, -2, abs(k)*1j)[0], "r")
    ax_pm.plot(r_out, a0_minus + B0_minus/r_out + a1_minus * r_out + B1_minus/r_out**2, "r")
    
    #Negative r
    ax_pm.plot(-r_inside, A0_plus * np.ones_like(r_inside) - A1_plus * r_inside, "b")
    ax_pm.plot(-r_wall, c0_plus * f(r_wall, 0, abs(k))[0] + cm1_plus * f(r_wall, -1, abs(k))[0] - c1_plus * f(r_wall, 1, abs(k))[0] - cm2_plus * f(r_wall, -2, abs(k))[0], "b")
    ax_pm.plot(-r_out, a0_plus + B0_plus/r_out - a1_plus * r_out - B1_plus/r_out**2, "b")

    ax_pm.plot(-r_inside, A0_minus * np.ones_like(r_inside) - A1_minus * r_inside, "r")
    ax_pm.plot(-r_wall, c0_minus * f(r_wall, 0, abs(k)*1j)[0] + cm1_minus * f(r_wall, -1, abs(k)*1j)[0] - c1_minus * f(r_wall, 1, abs(k)*1j)[0] - cm2_minus * f(r_wall, -2, abs(k)*1j)[0], "r")
    ax_pm.plot(-r_out, a0_minus + B0_minus/r_out - a1_minus * r_out - B1_minus/r_out**2, "r")

    ax_pm.grid()
    
    ax_pm.set_xlabel("r (m)")
    fig_pm.tight_layout()


if plot_Interior_parameters:
    
    lower_alpha = -20
    upper_alpha = 0
    
    
    upperk = np.log10(np.sqrt(10**upper_alpha * length_to_GeVm1**2 * (2700*density_to_GeV4))) 
    lowerk = np.log10(np.sqrt(10**lower_alpha * length_to_GeVm1**2 * (2700*density_to_GeV4)))
    
    
    k_array_interior = np.logspace(upperk,lowerk,100001)
    k_earth_interior = k_array_interior*np.sqrt(5.51/2.7)


    k_array_interior_negative = k_array_interior * 1.j
    k_earth_interior_negative = k_array_interior_negative*np.sqrt(5.51/2.7)
    
    alpha_array_interior = (k_array_interior/length_to_GeVm1)**2 /(2700*density_to_GeV4)


    
    a0_interior, a1_interior =  a0a1(k_earth_interior) 
    
    a0_interior_negative, a1_interior_negative =  a0a1(k_earth_interior_negative) 


    A0_interior, c0_interior, cm1_interior, B0_interior = A0_c0cm1_B0(R2, R1, k_array_interior, a0_interior)
    A1_interior, c1_interior, cm2_interior, B1_interior = A1_c1cm2_B1(R2, R1, k_array_interior, a1_interior)

    A0_interior_negative, c0_interior_negative, cm1_interior_negative, B0_interior_negative = A0_c0cm1_B0(R2, R1, k_array_interior_negative, a0_interior_negative)
    A1_interior_negative, c1_interior_negative, cm2_interior_negative, B1_interior_negative = A1_c1cm2_B1(R2, R1, k_array_interior_negative, a1_interior_negative)

    
    plt.close("coef interior")
    fig_interior, ax_interior = plt.subplots(2,2, num="coef interior", sharex=True, sharey=True, figsize=[6.5,5], layout="constrained")

    fig_interior.supxlabel(r"$|\alpha|$ (GeV$^{-2}$)", fontsize=16)

    ax_interior[0,0].set_title(r'$\left|A_0/a_0\right|$', fontsize=16)
    ax_interior[0,0].plot(alpha_array_interior, A0_interior/a0_interior, "r", alpha=0.7)
    
    ax_interior[0,0].set_yscale("log")
    ax_interior[0,0].set_ylim(1e-20,1e1)

    ax_interior[0,0].set_ylabel(r"$\alpha>0$", fontsize=16)

    ax_interior[1,0].plot(alpha_array_interior, abs(A0_interior_negative/a0_interior_negative), "r", alpha=0.7)
    
    ax_interior[1,0].set_yscale("log")
    ax_interior[1,0].set_ylim(1e-20,1e5)
    ax_interior[1,0].set_ylabel(r"$\alpha<0$", fontsize=16)


    ax_interior[0,1].set_title(r'$\left|A_1/a_1\right|$', fontsize=16)

    ax_interior[0,1].plot(alpha_array_interior, A1_interior/a1_interior, "r", alpha=0.7)
    
    ax_interior[0,1].set_yscale("log")
    ax_interior[0,1].set_ylim(1e-20,1e1)

    ax_interior[1,1].plot(alpha_array_interior, abs(A1_interior_negative/a1_interior_negative), "r", alpha=0.7)
    
    ax_interior[1,1].set_yscale("log")
    ax_interior[1,1].set_ylim(1e-20,1e5)
    for axis in ax_interior.flatten():
        l1, = axis.plot([alpha_C, alpha_C], [1e-200,1e100], ":", label = r"$|k|=k_c$")
        axis.tick_params(labelsize=12)
        axis.set_yticks(np.logspace(-20,5,6))
        axis.grid()

    fig_interior.legend(handles = [l1], fontsize=14, loc = 'upper right', bbox_to_anchor = (-0.02, -0.045, 1, 1))
    
    
    
    xrange = np.logspace(upper_alpha, lower_alpha, 9)
    ax_interior[0,0].set_xscale("log")
    ax_interior[0,0].set_xlim(np.min(xrange),np.max(xrange))
    ax_interior[0,0].set_xticks(xrange)
    
    fig_interior.savefig(r"Interior Coefficients.png", dpi=300)
if plot_c_Test:

    M_array_test = np.logspace(0,20,10001)
    k_array_test =  1.j*np.sqrt(2700 * density_to_GeV4) / (M_array_test/length_to_GeVm1)    
        
    alpha_array_test = 1/(M_array_test)**2
    
    k_earth_test = k_array_test*np.sqrt(5.51/2.7)

    a0_test, a1_test =  a0a1(k_earth_test) 
    A0_test, c0_test, cm1_test, B0_test = A0_c0cm1_B0(R2, R1, k_array_test, a0_test)
    A1_test, c1_test, cm2_test, B1_test = A1_c1cm2_B1(R2, R1, k_array_test, a1_test)
    
    
    c1_mask = ~(np.isnan(c1_test) + (abs(c1_test)>1e100))
    cm2_mask = ~(np.isnan(cm2_test) + (abs(cm2_test)>1e100))
    cm1_mask = ~(np.isnan(cm1_test) + (abs(cm1_test)>1e100) + (abs(cm1_test)<1e-100) )
    c0_mask = ~(np.isnan(c0_test) + (abs(c0_test)>1e100))
    
    plt.close("coef test")
    fig_test, ax_test = plt.subplots(2,4, num="coef test", sharex=True, sharey=False, figsize=[15,5], layout="constrained")
    fig_test.suptitle("Field Profile Coefficients", fontsize=16)
    fig_test.supxlabel(r"$|\alpha|$ (GeV$^{-2}$)", fontsize=16)
              
            
    try:
        ax_test[0,0].plot(abs(alpha_array_test), abs(A0_test), label = "$|A_0|$")
        print("---------------- \n A0 \n ----------------")
    except:
        ""
    try:
        ax_test[1,0].plot(abs(alpha_array_test), abs(A1_test*lengthscale), label = "$|A_1|$")
        print("---------------- \n A1 \n ----------------")
    except:
        ""
    
    try:
        ax_test[0,1].plot(abs(alpha_array_test[c0_mask]), abs(c0_test[c0_mask]), label = "$|c_0|$")
        print("---------------- \n c0 \n ----------------")
    except:
        ""
    
    try:
        ax_test[0,1].plot(abs(alpha_array_test[cm1_mask]), abs(cm1_test[cm1_mask]), label = "$|c_{-1}|$")
        print("---------------- \n cm1 \n ----------------")
    except:
        "" 
    
    try:
        ax_test[1,1].plot(abs(alpha_array_test[c1_mask]), abs(c1_test[c1_mask]), label="$|c_1|$")
        print("---------------- \n c1 \n ----------------")

    except:
        ""
    
    try:
        ax_test[1,1].plot(abs(alpha_array_test[cm2_mask]), abs(cm2_test[cm2_mask]), label="$|c_{-2}|$")
        print("---------------- \n cm2 \n ----------------")

    except:
        ""
        
    
    ax_test[0,2].plot(abs(alpha_array_test), abs(B0_test/lengthscale), label="$|B_0|$")
    print("---------------- \n B0 \n ----------------")

    ax_test[1,2].plot(abs(alpha_array_test), abs(B1_test/lengthscale**2), label="$|B_1|$")
    print("---------------- \n B1 \n ----------------")

    
    
    ax_test[0,3].plot(abs(alpha_array_test), abs(a0_test), label="$|a_0|$")
    print("---------------- \n a0 \n ----------------")

    ax_test[1,3].plot(abs(alpha_array_test), abs(a1_test*lengthscale), label="$|a_1|$")
    print("---------------- \n a1 \n ----------------")
    
    
    
    for axis in ax_test.flatten():
        axis.legend()
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_xticks(10**np.arange(round(np.min(np.log10(alpha_array_test))),round(np.max(np.log10(alpha_array_test)))+0.1,5))
        axis.set_xlim(np.min(alpha_array_test), np.max(alpha_array_test))


    

if plot_lengthscale_validitiy_Test:
    
    M_array_lengthscale = np.logspace(0,20,100001)
    k_array_lengthscale =  np.sqrt(2700 * density_to_GeV4) / (M_array_lengthscale/length_to_GeVm1)    
        
    alpha_array_lengthscale = 1/(M_array_lengthscale)**2
    
    k_earth_lengthscale = k_array_lengthscale*np.sqrt(5.51/2.7)
    
    plt.close("Lengthscale Test")
    fig_lengthscale, ax_lengthscale = plt.subplots(2,2,num="Lengthscale Test", sharex=True, sharey=True, layout="constrained")
    
    h_array = [100, 10000, 1000000]
    colours_array = ["blue", "red", "green"]
    alphas_array = [0.5, 0.7, 0.5]
    for i, h in enumerate(h_array):
        a0_lengthscale, a1_lengthscale = a0a1(k_earth_lengthscale,R_position=6.4e6+h)
        A0_lengthscale, c0_lengthscale, cm1_lengthscale, B0_lengthscale = A0_c0cm1_B0(R2, R1, k_array_lengthscale, a0_lengthscale)
        A1_lengthscale, c1_lengthscale, cm2_lengthscale, B1_lengthscale = A1_c1cm2_B1(R2, R1, k_array_lengthscale, a1_lengthscale)
    
        a0_lengthscale_negative, a1_lengthscale_negative = a0a1(1j*k_earth_lengthscale,R_position=6.4e6+h)
        A0_lengthscale_negative, c0_lengthscale_negative, cm1_lengthscale_negative, B0_lengthscale_negative = A0_c0cm1_B0(R2, R1, 1j*k_array_lengthscale, a0_lengthscale_negative)
        A1_lengthscale_negative, c1_lengthscale_negative, cm2_lengthscale_negative, B1_lengthscale_negative = A1_c1cm2_B1(R2, R1, 1j*k_array_lengthscale, a1_lengthscale_negative)
    
        
    
  
        lengthscale = h 
        
        numerator0            =  (B0_lengthscale/lengthscale )
        denominator0          =  a0_lengthscale
        
        numerator_negative0   =  (B0_lengthscale_negative/lengthscale )
        denominator_negative0 =  a0_lengthscale_negative
        
        numerator1            =  (B1_lengthscale/lengthscale**2 )
        denominator1          =  a1_lengthscale*lengthscale
        
        numerator_negative1   =  (B1_lengthscale_negative/lengthscale**2 )
        denominator_negative1 =  a1_lengthscale_negative*lengthscale
    
         
        ax_lengthscale[0,0].plot(alpha_array_lengthscale, abs(numerator0/denominator0), c=colours_array[i], alpha = alphas_array[i])
        ax_lengthscale[0,1].plot(alpha_array_lengthscale, abs(numerator1/denominator1), c=colours_array[i],  alpha = alphas_array[i])
    
    
        ax_lengthscale[1,0].plot(alpha_array_lengthscale, abs(numerator_negative0/denominator_negative0), c=colours_array[i], alpha = alphas_array[i])
        ax_lengthscale[1,1].plot(alpha_array_lengthscale, abs(numerator_negative1/denominator_negative1), c=colours_array[i], alpha = alphas_array[i])


    ax_lengthscale[0,0].set_xscale("log")
    ax_lengthscale[0,0].set_yscale("log")

    
    ax_lengthscale[0,0].set_ylabel(r"$\alpha>0$", fontsize=16)
    ax_lengthscale[1,0].set_ylabel(r"$\alpha<0$", fontsize=16)    
    ax_lengthscale[0,0].set_title(r"$\left|\frac{B_0/h}{a_0}\right|$",fontsize=16)
    ax_lengthscale[0,1].set_title(r"$\left|\frac{B_1/h^2}{a_1h}\right|$",fontsize=16)

    
    for axis in ax_lengthscale.flatten():
        axis.set_xlim(np.min(alpha_array_lengthscale), np.max(alpha_array_lengthscale))
        axis.set_ylim(1e-40,1e5)

        l1, = axis.plot([alpha_C, alpha_C], [1e-100, 1e100], ":", label = "$|k|=k_c$")
        axis.tick_params(labelsize=12)
        axis.set_xticks(np.logspace(-40,0,6))
        axis.set_yticks(np.logspace(-40,0,5))
        
        
        axis.grid(which="both")

        fig_lengthscale.supxlabel(r"$|\alpha|$ (GeV$^{-2}$)", fontsize=14)
        
    ax_lengthscale[0,1].legend(fontsize=12, ncol = 2, loc = "upper left")
    fig_lengthscale.savefig(r"Lengthscale_Test.png", dpi=300)
    
    
if plot_sphere_surface_value:
    plt.close("Sphere Surface Value")
    fig_sphere_surface, ax_sphere_surface = plt.subplots(2, 1, num="Sphere Surface Value", sharex=True, sharey=True, figsize = [4.5, 3.5], dpi=200, layout="constrained")
    
    
    alpha_rho_Rsquared_array = np.logspace(-10, 10, 10001)

    #for convenience, set units such that R=1 
    positive_surface_values = 1 + q_func(np.sqrt(alpha_rho_Rsquared_array),1)
    negative_surface_values = abs(1 + q_func(np.sqrt(alpha_rho_Rsquared_array)*1j,1))
    
    ax_sphere_surface[0].plot(alpha_rho_Rsquared_array, positive_surface_values, color = "r", alpha = 0.7, label = r"$\alpha>0$")
    ax_sphere_surface[1].plot(alpha_rho_Rsquared_array, negative_surface_values, color = "r", alpha = 0.7, label = r"$\alpha<0$")
    
    for i, axis in enumerate(ax_sphere_surface.flatten()):
        axis.grid()
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.legend(loc = "lower left", fontsize = 14)
    
    axis.set_xlabel(r"$\left|\alpha \rho_S R^2\right|$", fontsize=14)
    axis.set_xlim(np.min(alpha_rho_Rsquared_array), np.max(alpha_rho_Rsquared_array))
    
    
    fig_sphere_surface.supylabel(r"$\left|\varphi_S(r=R;\alpha)\right|$", fontsize=14, y=0.58)
    
    xticks = np.logspace(np.log10(np.min(alpha_rho_Rsquared_array)), np.log10(np.max(alpha_rho_Rsquared_array)),11)
    axis.set_xticks(xticks)
    fig_sphere_surface.savefig(r"Sphere_Surface_Value.png", dpi=300)
