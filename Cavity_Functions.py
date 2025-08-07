#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:18:30 2025

@author: ppxam5
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


#These are just coefficient functions. Direct calcualtion seems more efficient.
def A0_func(R2, R1, k, a0=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])
    
    a0 *= np.ones_like(k)
    a0 = np.array([a0])
    
    denominator= (u1 * np.sinh(u2-u1) + np.cosh(u2-u1))
    A0 = a0/denominator
    
    mask = np.where(np.isinf(np.cosh(u2-u1)))
    A0[mask] = 0
    
    return A0[0]

def c0_func(R2, R1, k, a0=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])
    
    a0 *= np.ones_like(k)
    a0 = np.array([a0])
    
    numerator = a0 * (np.cosh(u1) - u1 * np.sinh(u1))
    denominator =  (u1 * np.sinh(u2-u1) + np.cosh(u2-u1))
    
    
    return (numerator/denominator)[0]

def cm1_func(R2, R1, k, a0=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])
    
    a0 *= np.ones_like(k)
    a0 = np.array([a0])
    
    numerator =  a0 * (u1 * np.cosh(u1) - np.sinh(u1))
    denominator =  (u1 * np.sinh(u2-u1) + np.cosh(u2-u1))
    
    mask = np.where(abs(u2) < 1e-3)
    
    numerator[mask] = a0[mask] * u1[mask]**3 /3
    
    
   
    return (numerator/denominator)[0]

def B0_func(R2, R1, k, a0=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])
    
    a0 *= np.ones_like(k)
    a0 = np.array([a0])
    
    numerator = a0 * ( np.sinh(u2-u1) + (u1 - u2) * np.cosh(u2-u1) - u1 * u2 * np.sinh(u2-u1) )
    denominator =  (u1 * np.sinh(u2-u1) + np.cosh(u2-u1))
    
    B0 = (numerator/denominator)
    
    if u2.dtype != complex:
    
        mask = np.where((u2-u1) > 1.5e1) 
    
        B0[mask] = a0[mask] * (1-u2[mask])
    
    
    return B0[0]/k


def A1_func(R2, R1, k, a1=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])

    a1 *= np.ones_like(k)
    a1 = np.array([a1])
    
    denominator = ( 3*np.sinh(u2 - u1) + 3*u1 * np.cosh(u2 - u1) + u1**2 * np.sinh(u2 - u1) )
    
    A1 = 3 * a1 * u2 /denominator
    
    mask = np.where(np.isinf(np.cosh(u2-u1)))
    
    A1[mask] = 0
    
    return A1[0]


def c1_func(R2, R1, k, a1=1):
    u1 = k*R1
    u2 = k*R2
    
    numerator   = -3 * a1 * R2 * (3*np.cosh(u1) - 3*u1*np.sinh(u1) + u1**2 * np.cosh(u1))
    denominator = 3 * np.sinh(u2-u1) + 3*u1 * np.cosh(u2-u1) + u1**2 * np.sinh(u2-u1) 
    return numerator/denominator

def cm2_func(R2, R1, k, a1=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])
    
    a1 *= np.ones_like(k)
    a1 = np.array([a1])
    
    numerator   = -3 * a1 * R2 * (3* np.sinh(u1) - 3*u1 * np.cosh(u1) + u1**2* np.sinh(u1))
    mask = np.where(abs(u2)<1e-3)
  
    numerator[mask] = -3 * a1[mask] * R2 * 8/(5*4*3*2) * u1[mask]**5
    
    denominator = 3 * np.sinh(u2-u1) + 3*u1 * np.cosh(u2-u1) + u1**2 * np.sinh(u2-u1) 
    return (numerator/denominator)[0]

def B1_func(R2, R1, k, a1=1):
    u1 = np.array([k*R1])
    u2 = np.array([k*R2])
    
    a1 *= np.ones_like(k)
    a1 = np.array([a1])
    
    numerator = np.sinh(u2-u1) * (9 + 3*u2**2 + 3*u1**2 + (u1*u2)**2  - 9*u1*u2) +\
                np.cosh(u2-u1) * (3 * u1*u2 * (u2-u1) - 9*(u2 - u1) )
    
    denominator = 3 * np.sinh(u2-u1) + 3*u1 * np.cosh(u2-u1) + u1**2 * np.sinh(u2-u1) 
    
    mask = np.where(abs(u2)<1e-3)
    numerator[mask] = u1[mask]**2 * u2[mask]**2 * (u2[mask]-u1[mask])

    B1 = (numerator/denominator * -1 * a1 * R2)

    if u2.dtype != complex:
   
        mask = np.where((u2-u1) > 1.5e1) 
   
        B1[mask] = -1 * a1[mask] * R2 * (u2**2 - 3*u2 + 3)[mask] 
    
    return B1[0]/k**2

def f(r,l,k, N=int(100)):
    """
    Parameters
    ----------
    r : arr/float
        Radial coordiante (m).
    l : int
        index for function
    k : float
        Characteristic wavevector (m).

    Returns
    -------
    arr/float
        value of function used in calculating radial profiles.
    """
    
    kr = k*r
    
    #Eigenfunctions for field inside matter under cyllindrical symmetry. Can 
    #comment in/out cases to use scipy spherical bessel functions, or explicit
    #calculation. Changes may be desired if problems are noticed with floating 
    #point precision.
    match l:
    #    case -2:
    #       f = -np.cosh(kr)/kr**2 + np.sinh(kr)/kr
    #       f_deriv = 2*np.cosh(kr)/kr**3 - 2*np.sinh(kr)/kr**2 + np.cosh(kr)/kr
            
    #        return f, k*f_deriv
        
        case -1:
            f = np.cosh(kr)/(kr)                                #This is y0 **NOT** j-1. Sign difference.
            f_deriv = np.sinh(kr)/(kr) - np.cosh(kr)/(kr)**2
            
            return f, k*f_deriv
        
        case 0:
            f = np.sinh(kr)/(kr)
            f_deriv = np.cosh(kr)/(kr) - np.sinh(kr)/(kr)**2
            
            return f, k*f_deriv
        
    #   case 1:
    #       f = np.sinh(kr)/kr**2 - np.cosh(kr)/kr
    #       f_deriv = -2*np.sinh(kr)/kr**3 + 2*np.cosh(kr)/kr**2 - np.sinh(kr)/kr
            
    #        return f, k*f_deriv
    
    if l<0:
        p = -l-1
        f = spherical_yn(p, -1.j*kr)
        f_deriv = (spherical_yn(p, -1.j*kr, derivative=True) * -1.j*k)
                
    else:
        f = spherical_jn(l, -1.j*kr)
        f_deriv = (spherical_jn(l, -1.j*kr, derivative=True) * -1.j*k)
        
    factor = (-1.j)**(l)
    
    f *= factor
    f_deriv *= factor
    
    #I've assumed that only k _or_ r is an array. 

    
    return f, f_deriv


def A0_c0cm1_B0(R2, R1, k, a0):

        
    A0 = A0_func(R2, R1, k, a0)
    
    c0, cm1 = c0_func(R2, R1, k, a0), cm1_func(R2, R1, k, a0) 
    
    B0 = B0_func(R2, R1, k, a0)  
    
    return A0, c0, cm1, B0

def A1_c1cm2_B1(R2, R1, k, a1):
    
    
    A1 = A1_func(R2, R1, k, a1)
    
    c1, cm2 = c1_func(R2, R1, k, a1), cm2_func(R2, R1, k, a1)

    B1 = B1_func(R2, R1, k, a1) 
    
    return A1, c1, cm2, B1

def q_func(k,R):
    """
    Function for q as in the solid sphere case

    Parameters
    ----------
    k : float/array
        wavenumber/growth scale of matter field in matter.
    R : float
        radius of object.

    Returns
    -------
    float
        value for q.

    """
    return (np.tanh(k*R)-k*R)/(k*R)


def a0a1(k_earth, R_position=6.4e6+100, R_earth=6.4e6):
    """
    

    Parameters
    ----------
    k_earth : float/arr
        k for earth given earth density/alpha. 
    r : float, optional
        position of test mass. The default is 6.4e6.
    R : float, optional
        radius of test source mass. The default is 6.4e6+100 (100m above earth surface).
        
    All must have same unit type

    Returns
    -------
    a0, a1: coefficients for linear field profile.

    """
    q = q_func(k_earth,R_earth)
    
    a0 = 1 + q * R_earth/R_position
    a1 = -q * R_earth/R_position**2
    
    return a0, a1



def get_wall_profile(R2, R1, k, a0=1, a1=0, theta=0, double_sided=False, N_points=1001):
    
    r = np.linspace(R1, R2, N_points)
    
    A0, c0, cm1, B0 = A0_c0cm1_B0(R2, R1, k, a0)
    A1, c1, cm2, B1 = A1_c1cm2_B1(R2, R1, k, a1)
    
    order_0_component = c0 * f(r, 0, k)[0] + cm1 * f(r, -1, k)[0]
    order_1_component = c1 * f(r, 1, k)[0] + cm2 * f(r, -2, k)[0]

    if double_sided:
        varphi = np.zeros(2 * len(order_0_component))
        varphi[len(order_0_component):] = order_0_component + order_1_component * np.cos(theta)
        varphi[:len(order_0_component)] = order_0_component - order_1_component * np.cos(theta)
        
    else:
        varphi = ( order_0_component  +  order_1_component )
        
    return varphi

def get_interior_sphere_profile(R, k, N_points=1001):
    r = np.linspace(0,R, N_points)
    
    varphi = 1/np.cosh(k*R) * np.sinc(-1.j * k*r/np.pi)
    return varphi


def getQ_dashs(A,Z):
    Q_dashm = -0.036/A**(1/3) - 0.02 * (A-2*Z)**2/A**2 -1.4e-4 * Z*(Z-1)/A**(4/3)

    Q_dashdm = 0.0017 *(A-2*Z)/A

    Q_dashme = -2.75e-4 *(A-2*Z)/A

    Q_dashe = -4.1e-4 * (A-2*Z)/A + 7.7e-4 * Z*(Z-1)/A**(4/3)
    
    return {"Q'_m":Q_dashm, "Q'_δm":Q_dashdm,"Q'_me":Q_dashme,"Q'_e":Q_dashe}


def getQs(A,Z):
    #approximate FA = 1
    Q_m = 0.093 -0.036/A**(1/3) - 0.02 * (A-2*Z)**2/A**2 -1.4*10**-4 * Z*(Z-1)/A**(4/3)

    Q_dm = 0.0017 *(A-2*Z)/A

    Q_me = 5.5*1e-4 * Z/A

    Q_e = -1.4e-4 + 8.2e-4 * Z/A  + 7.7e-4 * Z*(Z-1)/A**(4/3)
    
    return {"Q_m":Q_m, "Q_δm":Q_dm,"Q_me":Q_me,"Q_e":Q_e}