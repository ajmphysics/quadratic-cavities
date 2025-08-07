#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:28:28 2025

@author: ppxam5
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con

plotting_params = {"ytick.color"      : "black",
          "xtick.color"      : "black",
          "axes.labelcolor"  : "black",
          "axes.edgecolor"   : "black",
          "text.usetex"      : True,
          "font.family"      : "serif",
          "font.serif"       : ["Computer Modern Serif"]}
plt.rcParams.update(plotting_params)



c = con.c
G = con.G
h = con.hbar     
e_charge = con.elementary_charge
M_pl = np.sqrt(h*c/(8*np.pi*G))
rho_DM = 7.13e-22 #kgm-3 This has been calculated correctly (0.4 GeV/cm3)


ma_fa_product_GeV = 5.7e-3
ma_fa_product_m = ma_fa_product_GeV * 2.56e31
#Constants


density_to_GeV4 = 1/(e_charge**4*c**-5*h**-3 * 1e36) #Fine
length_to_GeVm1 = 1/(0.197e-15)   # Fine
kg_to_GeV = c**2/(e_charge * 1e9) #Fine
#Conversion Factors to/from natural units


R = 6.4e6*length_to_GeVm1 # Earth radius in --natural units--
R_earth = 6.4e6 # Earth radius in --m--

M_pl_GeV = np.sqrt(h*c/(8*np.pi*G)) * c**2/(e_charge)/1e9
kappa_GeV = 1/(np.sqrt(2)*M_pl_GeV)
G_GeV = 1/(8*np.pi*M_pl_GeV**2)
rho_DM_GeV = rho_DM*density_to_GeV4
#Quantities in natural units

a_dash_infty = np.sqrt(2*rho_DM_GeV)/(5.7e-3)
