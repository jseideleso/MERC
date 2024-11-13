
import numpy as np
from numpy import sum as npsum
from numpy import exp as npexp
from numpy import sqrt as npsqrt
from numpy import diff as npdiff
from numpy import multiply as npmultiply
from numpy import add as npadd
from numpy import tensordot as nptensordot
from numpy import dot as npdot
from numpy import trapz as nptrapz
from numpy import log as nplog
from numpy import empty as npempty
from numpy import power as nppow
from numpy import mean as npmean
from numpy import einsum as npeinsum 
from numpy import true_divide as nptrue_divide
from numpy import tile as nptile

from numpy import array as nparray
from numpy import cos as npcos
from bisect import bisect_left, bisect
#for extrapolation
#from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
from scipy.constants import k as k_b
from scipy.constants import c
from scipy.constants import m_e as m_el
from scipy.constants import h as h_SI
import scipy.constants.codata as pyconst
from scipy.special import wofz #for voigt profile
import math
from math import sqrt as msqrt
from math import exp as mexp
from math import log as mlog
from math import fsum as mfsum
from math import e
from math import pi as m_pi
#from bisect import bisect
import os #for file reading and such
import myconstants #constants needed additionally, see same directory
import calc_velocity #all functions needed for the velocity shifts
import data_comparison
import grid_create
import matplotlib.pyplot as plt
import copy
import sys


#-----------------------------------------------------------
#           variables for multiple use
#-----------------------------------------------------------
masses = myconstants.masses
species = myconstants.masses.keys()

#jupiter mass taken from NASA
M_p = myconstants.M_planet
k_b_cm = k_b*10**(7) #cm**2g/(s**2*K)
period = myconstants.period
R_p = myconstants.R_0 #planet radius in cm
G_cgs = 6.6725985*10**(-8) #cm**3/gs**2 
m_e=m_el*1000 #in g
h=h_SI*10**7 #in cgs, ergs*s
line_dict=myconstants.line_dict
Xs=myconstants.solar_abundances
e_r=pyconst.value('classical electron radius')*100 #to make it in cm...
speed_light = c*100 #cm/s

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#-----------------------------------------------------------
#           function to switch from atm to pascal
#-----------------------------------------------------------
def atm_to_pasc(press):
    press_pasc = [p*1013250. for p in press] #in cgs 0.1 Pa
    return press_pasc

def pasc_to_atm(press):
    press_atm = [p/1013250. for p in press] #in atm
    return press_atm

def make_T_P_profile_iso(T,Xfrac,P_top=10**(-13),num=100):
    P_start_grid = np.logspace(nplog(Xfrac*10**(4)), nplog(P_top), num=num, base = e)
  
    mod='log'
    tab_Ts, tab_Ps = [[T,T,T,T,T,T,T,T,T,T], [10.,0.5,0.01,0.001,0.0002,6e-06,8.5e-09,1.e-15,1.e-20,1.e-25]]
    if mod == 'lin':
        T_start_grid = np.interp(P_start_grid, tab_Ps[::-1], tab_Ts[::-1])
    elif mod == 'log':
        T_start_grid = np.interp(np.log(P_start_grid), np.log(tab_Ps[::-1]), tab_Ts[::-1])
    else:
        raise ValueError('Source T grid does not specify interpolation mode')
    P_T_dict = {'P': np.asarray(atm_to_pasc(P_start_grid)),'T':np.asarray(T_start_grid)}
    return P_T_dict

def create_atmo(dict_values,Xfrac):
    Xs['Na']=10**(-5.7) #this is solar

    if 'H2' in Xs:
        del Xs['H2']
    Xs['H2']=1-sum(Xs.values())

    mu_static = sum([masses[name]*Xs[name] for name in species])
    dict_values['mu']= np.ones(len(dict_values['P']))*mu_static

    frac_height = np.array([nptrapz(k_b_cm*dict_values['T'][:i]/(dict_values['mu'][:i]*G_cgs*M_p*dict_values['P'][:i]),dict_values['P'][:i]) for i in range(2, len(dict_values['P'])+1)])
    R_grid = (R_p**(-1)+frac_height)**(-1)

    R_grid = np.insert(R_grid,0,R_p)
    dict_values['z'] = np.asarray(R_grid - R_p)

    dict_values['g'] = np.asarray(G_cgs*M_p*R_grid**(-2.))

    dict_values['scaleH']=(k_b_cm*dict_values['T'])/( dict_values['mu']*(dict_values['g']))

    dict_values['Na'] = nptrue_divide(dict_values['P'],dict_values['T'])*Xs['Na']/k_b_cm
    return dict_values

def calculate_binlimits(total_bin_nb,bin_width):
    binlimits = {}
    for key in line_dict:
        center_bin=round(line_dict[key][0],4)
        left_limit = center_bin-total_bin_nb*0.5*bin_width
        right_limit =center_bin+total_bin_nb*0.5*bin_width
        binlimits[key]=[left_limit,right_limit]
    return binlimits

def make_onegrid(bin_limits, bin_width):
    bin_nb= round((bin_limits[1]-bin_limits[0])/bin_width,0)
    wavelength_grid = np.linspace(bin_limits[0],bin_limits[1],bin_nb)
    return wavelength_grid
def add(x,y): return x+y


def make_lineprofile(T_list,mu_list,grid_lambda,line_key):
    Psi=npexp(-0.5*(np.einsum('z,x->zx',(line_dict[line_key][0]*10**(-7.)*speed_light)/npsqrt(2.*k_b_cm*T_list/mu_list),(grid_lambda**(-1.)-1./line_dict[line_key][0])*10**(7.)))**2.)
    return Psi

    
def make_Voigt_profile(v_t,T_list,P_list,mu_list,grid_lambda,line_key):

    FWHM_gauss = 2.*msqrt(mlog(2.))*(1./(line_dict[line_key][0]*10**(-7.)))*npsqrt(2.*k_b_cm*T_list/(masses['Na'])+v_t**2)
    
    HWHM_l = (line_dict[line_key][1]*0.5/m_pi)*0.5 + 0.071*(T_list*0.0005)**(-0.7)*pasc_to_atm(P_list)*speed_light

    nus = speed_light*10**(7.)/(grid_lambda)
    central_nu = speed_light*10**(7.)/(line_dict[line_key][0]) 

    Psi=voigt(nus,FWHM_gauss,HWHM_l,central_nu,1)

    return Psi


def voigt(freq, FWHM=1, gamma=1, center=0, scale=1):
  
    sigma = FWHM / 2.3548200450309493 #vector in height
    
    z = [msqrt(2.)**(-1)*(freq - center + 1j * a)/b  for a,b in zip(gamma,sigma)]
 
  
    V=npeinsum('ij,i->ij',wofz(z),(msqrt(2. * m_pi) * sigma)**(-1))
   
    return scale * V.real

def calculate_cross_section(v_t,T,P,mu,grid_lambda,line_key):
    Psi_Voigt_z = make_Voigt_profile(v_t,T,P,mu, grid_lambda,line_key)
 
    cross_section = e_r*m_pi*speed_light*Psi_Voigt_z*line_dict[line_key][2] 
    return cross_section


def calculate_opacity(v_t,grid_lambda,values_dict,line_key):
    cross_section_z = calculate_cross_section(v_t,values_dict['T'],values_dict['P'],values_dict['mu'],grid_lambda,line_key)
    opa_depth=np.einsum('ij,i->ij',cross_section_z,values_dict['Na'])
 
    return opa_depth


def nm_to_cm(wavelength_grid):
    cmwave = wavelength_grid*10**(-7.)
    return cmwave

def calculate_rayleigh_cross_section(rayleigh_element, grid_lambda):

    #To apply Sneep&Unbachs formulas for rayleigh, the Loschmidt number has to be calculated with the density at standard conditions
    N_Loschmidt = 2.5469e19 # [cm-3] at P = 1 atm, T = 288.15 K
    
    if (rayleigh_element == 'H2'):
        # See equation (3) of Dalgarno (1962)
        a = 8.14e-13    # [A^4]
        b = 1.28e-6     # [A^6]
        c = 1.61        # [A^8]
        
        # Accurate to O(lambda^(-10))
        cross_sec = a*grid_lambda**(-4.) + b*grid_lambda**(-6.) + c*grid_lambda**(-8.)
    
    elif (rayleigh_element == 'N2'):
        print('RAYLEIGH BY N2')
        ww_long = grid_lambda[np.where(468.16 < grid_lambda < 2057.61)] #in nm
        ww_other = grid_lambda[np.where(np.any([grid_lambda < 254.0005, 2057.61 < grid_lambda], axis = 0))]
        if any(ww_other):
            print("The selected wavelength grid is out of range. Check that your grid is 468.16 < grid_lambda < 2057.61 nm")
        refractive_index = 1. + 10**(-8.)*(6498.2 + 307.43305*10**(12.)/(14.4*10**(9.) - (nm_to_cm(ww_long))**(-2.)))
        king_factor = 1.034 + 3.17*10**(-12.)*(nm_to_cm(grid_lambda))**(-2.)
        cross_sec = 24.*(m_pi**3.)*nm_to_cm(grid_lambda)**(-4.)* (refractive_index**2.-1.)**2. * king_factor /\
            (N_Loschmidt**2. * (refractive_index**2.+2.)**2.)
    elif (rayleigh_element == 'CO2'):
        print('RAYLEIGH BY CO2')
        refractive_index = 1. + 1.1427e6*(5799.25/(128908.9**2. - nm_to_cm(grid_lambda)**(-2.)) +\
                                          120.05/(89223.8**2. - nm_to_cm(grid_lambda)**(-2.)) +\
                                          5.3334/(75037.5**2. - nm_to_cm(grid_lambda)**(-2.)) +\
                                          4.3244/(67837.7**2. - nm_to_cm(grid_lambda)**(-2.)) +\
                                          0.1218145e-4/(2418.136**2. - nm_to_cm(grid_lambda)**(-2.))\
                                          )
        king_factor = 1.1364 + 25.3e-12*nm_to_cm(grid_lambda)**(-2.)
        cross_sec = 24.*(m_pi**3.)*nm_to_cm(grid_lambda)**(-4.)* (refractive_index**2.-1.)**2. * king_factor /\
                                              (N_Loschmidt**2. * (refractive_index**2.+2.)**2.)
    else:
        print('WARNING: RAYLEIGH SCATTERER NOT FOUND!')
        cross_sec = 0.
    return cross_sec

def calculate_spectrum_ratio_base_lat(sectors, T,P,z,grid_lambda,ref_grid, cross_section_rayleigh,opacity_z):
    num_density=nptrue_divide(P,T)*Xs['H2']/k_b_cm
    tau_rayleigh=np.outer(num_density,cross_section_rayleigh)[:-1]
    end_sumation=len(z[:-1])

    
    height_bins = npdiff(list(z))
    tau_b = npempty(shape=(sectors,end_sumation,len(ref_grid)),dtype='float64')
    tau_b_rev = npempty(shape=(sectors,end_sumation,len(ref_grid)),dtype='float64')

    sigma_lambda = 0

    n_layers = np.arange(end_sumation)
    b_shorter = np.asarray([end_sumation-n_layer for n_layer in n_layers])
    b_length = b_shorter+1
    bb = z[:-1]
    b_Rp = (bb+R_p)
    full_lambda = nptile(grid_lambda,(sectors,end_sumation,1))
    vel_planet = calc_velocity.calc_vel_planet(R_p,period)
    for n_layer, b in enumerate(z[:-1]):

        x_grid_binlimit=npempty(b_length[n_layer])
        x_grid_binlimit[0]=0.
        x_grid_binlimit[1:b_length[n_layer]]= npsqrt((z[(n_layer+1):]+ R_p)*(z[(n_layer+1):]+ R_p) - b_Rp[n_layer]*b_Rp[n_layer])
        bins_in_dx = npdiff(x_grid_binlimit)
        cut_opa_z=nptile(opacity_z[n_layer:end_sumation],(sectors,1,1))
        grid_lambda_all = full_lambda[:,:b_shorter[n_layer]]

        vel_planet_LOS = calc_velocity.calc_velocity_circ_LOS(vel_planet,b,x_grid_binlimit[1:], R_p)

        velocity_LOS, velocity_LOS_reverse= calc_velocity.corr_latitude(0.,vel_planet_LOS,sectors)


        wave_grid_shifted=calc_velocity.velocity_shift_latitude(grid_lambda_all, velocity_LOS,speed_light)
        wave_grid_shifted_reverse= calc_velocity.velocity_shift_latitude(grid_lambda_all, velocity_LOS_reverse,speed_light)

        cut_opa_z_wind=calc_velocity.align_grids_lat(wave_grid_shifted, ref_grid, cut_opa_z)
        cut_opa_z_wind_rev =calc_velocity.align_grids_lat(wave_grid_shifted_reverse, ref_grid, cut_opa_z)

        tau_b[:,n_layer]= npdot(bins_in_dx,cut_opa_z_wind)
        tau_b_rev[:,n_layer]=npdot(bins_in_dx,cut_opa_z_wind_rev)

    sigma_lambda= m_pi/sectors*npsum(npeinsum('i,jik->jik',b_Rp*height_bins,(2. - npexp(-tau_rayleigh[np.newaxis,...])*(npexp(-2.*tau_b)+npexp(-2.*tau_b_rev)))),axis=(1,0))

    spectrum_ratio = -np.true_divide(sigma_lambda + m_pi*R_p*R_p,m_pi*myconstants.R_Star*myconstants.R_Star)

    return spectrum_ratio

def calculate_spectrum_ratio_pattern_lat(sectors, T,P,z,grid_lambda,ref_grid, cross_section_rayleigh,opacity_z,v_rot,v_rad,switchP, pattern_id,layer_number=-1):
    #calculate the numerical density
    num_density=nptrue_divide(P,T)*Xs['H2']/k_b_cm
        
    #contribution function
    if layer_number != -1:
        opacity_z[layer_number]=0.
    
    tau_rayleigh=np.outer(num_density,cross_section_rayleigh)[:-1]
    end_sumation=len(z[:-1])

    height_bins = npdiff(list(z))

    tau_b = npempty(shape=(4,sectors,end_sumation,len(ref_grid)),dtype='float64')

    sigma_lambda = 0

    n_layers = np.arange(end_sumation)
    b_shorter = np.asarray([end_sumation-n_layer for n_layer in n_layers])
    b_length = b_shorter+1
    bb = z[:-1]
    b_Rp = (bb+R_p)
    full_lambda = nptile(grid_lambda,(sectors,end_sumation,1))

    switchP_pasc = switchP*100000.
    switch_index = len(P)-bisect(list(reversed(P)),switchP_pasc)

    if switch_index >= len(z):
        switch_index = len(z)-1
    z_switch = z[switch_index]

    vel_planet = calc_velocity.calc_vel_planet(R_p,period)
    
    for n_layer, b in enumerate(z[:-1]):
        x_grid_binlimit=npempty(b_length[n_layer])
        x_grid_binlimit[0]=0.
        x_grid_binlimit[1:b_length[n_layer]]= npsqrt((z[(n_layer+1):]+ R_p)*(z[(n_layer+1):]+ R_p) - b_Rp[n_layer]*b_Rp[n_layer])
        bins_in_dx = npdiff(x_grid_binlimit)

        grid_lambda_all = full_lambda[:,:b_shorter[n_layer]]

        velocity_planet = calc_velocity.calc_velocity_circ_LOS(vel_planet, b, x_grid_binlimit[1:], R_p)
        
        cut_opa_z=nptile(opacity_z[n_layer:end_sumation],(sectors,1,1))
        
        velocity_LOS1, velocity_LOS2, velocity_LOS3, velocity_LOS4, =calc_velocity.calc_velocity_pattern_lat(velocity_planet,v_rot,v_rad, b, x_grid_binlimit[1:], R_p, z_switch, pattern_id,sectors)

        wave_grid_shifted1=calc_velocity.velocity_shift_latitude(grid_lambda_all, velocity_LOS1,speed_light)
        wave_grid_shifted2=calc_velocity.velocity_shift_latitude(grid_lambda_all, velocity_LOS2,speed_light)
        wave_grid_shifted3=calc_velocity.velocity_shift_latitude(grid_lambda_all, velocity_LOS3,speed_light)
        wave_grid_shifted4=calc_velocity.velocity_shift_latitude(grid_lambda_all, velocity_LOS4,speed_light)

        cut_opa_z_wind1=calc_velocity.align_grids_lat(wave_grid_shifted1, ref_grid, cut_opa_z)
        cut_opa_z_wind2=calc_velocity.align_grids_lat(wave_grid_shifted2, ref_grid, cut_opa_z)
        cut_opa_z_wind3=calc_velocity.align_grids_lat(wave_grid_shifted3, ref_grid, cut_opa_z)
        cut_opa_z_wind4=calc_velocity.align_grids_lat(wave_grid_shifted4, ref_grid, cut_opa_z)


        tau_b[0,:,n_layer]= npdot(bins_in_dx,cut_opa_z_wind1)
        tau_b[1,:,n_layer]= npdot(bins_in_dx,cut_opa_z_wind2)
        tau_b[2,:,n_layer]= npdot(bins_in_dx,cut_opa_z_wind3)
        tau_b[3,:,n_layer]= npdot(bins_in_dx,cut_opa_z_wind4)


    sigma_lambda= 0.5*m_pi/sectors*npsum(npeinsum('i,bjik->bjik',b_Rp*height_bins,(1. - npexp(-(2.*tau_b+tau_rayleigh[np.newaxis,np.newaxis,...])))),axis=(2,1,0))


    spectrum_ratio = -(sigma_lambda + m_pi*R_p*R_p)*(m_pi*myconstants.R_Star*myconstants.R_Star)**(-1)

    return spectrum_ratio
