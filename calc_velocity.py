from numpy import empty as npempty
from numpy import sqrt as npsqrt
from numpy import mean as npmean
from numpy import einsum as npeinsum
from numpy.core.multiarray import interp as compiled_interp
from numpy import pi as npi
from numpy import cos as npcos
from numpy import tile as nptile
from numpy import array as nparray
from bisect import bisect

def calc_velocity_LOS(velocity, b, x_bin_limits,R_p):

    velocity_grid_b=npempty(len(x_bin_limits))
    const_height=(b+R_p)**2
    x_squared=x_bin_limits**2
    for i,x in enumerate(x_bin_limits):
        if i == 0:
            y_grid=x_bin_limits
        else:
            y_grid=x_bin_limits[:-i]
        velocity_in_y = velocity*x/npsqrt(x_squared[i]+y_grid**2+const_height)
        velocity_grid_b[i] = npmean(velocity_in_y)
    avg_vel_b = npmean(velocity_grid_b)

    return avg_vel_b


def calc_velocity_circ_LOS(velocity, b, x_bin_limits,R_p):

    velocity_grid_b=npempty(len(x_bin_limits))
    const_height=(b+R_p)**2.
    x_squared = x_bin_limits**2.
    velocity_grid_b = -velocity*(b+R_p)/npsqrt(x_squared+const_height)
    return velocity_grid_b



def calc_velocity_x(velocity, b, x,R_p):
    velocity_grid_b=npempty(len(x))
    const_height=(b+R_p)**2
    velocity_grid_b = velocity*x/npsqrt(x**2+const_height)
    return velocity_grid_b


def calc_vel_latitude(vel,period,radius,nb_slices=3,correct=False):
    center_angle = npi*0.25/nb_slices
  
    cos_angles = nparray([npcos(center_angle*(2.*i+1.)) for i in range(nb_slices)])
    
    if correct == True:
        vel_grid = nptile(vel,(nb_slices,1))
        vel_grid = npeinsum('ij,i->ij',vel_grid,cos_angles)
        
    vel_planet = 2.*npi*radius/(period*24.*3600.)*cos_angles
    vel_grid = vel_grid+vel_planet[:,None]
    return vel_grid
    
def calc_vel_planet(radius,period):
    return 2.*npi*radius/(period*24.*3600.)
    
def corr_latitude(vel,vel_planet,nb_slices=3,correct=True):
    center_angle = npi*0.25/nb_slices
    cos_angles = nparray([npcos(center_angle*(2.*i+1.)) for i in range(nb_slices)])
    vel_grid_wind = nptile(vel,(nb_slices,1))
    if correct == True:
        cos_wind = cos_angles

        vel_grid_wind = npeinsum('ij,i->ij',vel_grid_wind,cos_wind)

    vel_planet = nptile(vel_planet,(nb_slices,1))

    vel_planet_full = npeinsum('ij,i->ij',vel_planet,cos_angles)

    vel_grid = vel_grid_wind+vel_planet_full
    vel_grid_reverse = vel_grid_wind-vel_planet_full

    return vel_grid, vel_grid_reverse

def calc_velocity_pattern_lat(velocity_planet,velocity_rot, velocity_up, b, x,R_p,z_switch,pattern_id,nb_slices=3):

    vel_full=npempty(shape=(4,nb_slices,len(x)),dtype='float64')

    const_height=(b+R_p)**2.

    if b >= z_switch:
        x_switch = 0
    else:
        x_switch = bisect(x,npsqrt((z_switch+R_p)**2.-const_height))
        

    center_angle = npi*0.25/nb_slices
    cos_angles = nparray([npcos(center_angle*(2.*i+1.)) for i in range(nb_slices)])
  
    vel_full_planet = npeinsum('j,i->ij',velocity_planet,cos_angles)

    #############
    ##############
    ##############
    #jet
    cos_angles[4:] = cos_angles[4:]*0.
#    cos_angles[0] = 1.
#    cos_angles[1] = 1.
#    cos_angles[2] = 1.
#    cos_angles[3] = 1.

    #rotational part
    if x_switch != 0:
        slice_lower = -velocity_rot*(b+R_p)/npsqrt(x[:x_switch]**2+const_height)
    else:
        slice_lower = []

    velocity_lower = npeinsum('j,i->ij',slice_lower,cos_angles)

    slice_upper = velocity_up*x[x_switch:]/npsqrt(x[x_switch:]**2+const_height)
    velocity_upper = nptile(slice_upper,(nb_slices,1))  
    
    if pattern_id == "srot_ver_lat":
        vel_full[[0,3],:,:x_switch] = velocity_lower
        vel_full[[1,2],:,:x_switch] = -velocity_lower
    
    elif pattern_id == "dtn_ver_lat":

        ################################
        # Jet implementation
        ###############################
        #full winds across the hemisphere
        #vel_full[:,:,:x_switch] = velocity_lower
        #This is for a one sided DTN
#        two sectors have no wind
        vel_full[[0,3],:,:x_switch] = velocity_lower*0.
#        two sectors have wind going away from the observer
        vel_full[[1,2],:,:x_switch] = -velocity_lower
    else:
        print("calc_velocity.calc_velocity_pattern_lat: pattern id not recognised. Only srot_ver_lat and dtn_ver_lat are valid options.")
        sys.exit()
    #two sectors have the upwards wind point towards the observer
    vel_full[:2,:,x_switch:] = velocity_upper
    #two sectors have the upwards wind point away from the observer
    vel_full[2:,:,x_switch:] = -velocity_upper
        
    #adding the planet rotation, towards and away from the observer
    vel_full[[0,3]] = vel_full[[0,3]] + nptile(vel_full_planet,(2,1,1))
    vel_full[[1,2]] = vel_full[[1,2]] + nptile(-vel_full_planet,(2,1,1))

    return vel_full[0], vel_full[1], vel_full[2], vel_full[3]



def velocity_shift_latitude(wavelength_grid, velocity, speed_light):
    wavelength_grid_shifted=npeinsum('ijk,ij->ijk',wavelength_grid,(1+velocity/speed_light))
    return wavelength_grid_shifted


def align_grids(wavelength_grid_all_shifted, wave_grid_data, tau_grid):
  
    rows, cols = wavelength_grid_all_shifted.shape
    rebinned_tau = npempty((rows,) + wave_grid_data.shape, dtype=tau_grid.dtype)

    for j in range(rows) :
        rebinned_tau[j] = compiled_interp(wave_grid_data,wavelength_grid_all_shifted[j], tau_grid[j])
    return rebinned_tau

def align_grids_lat(wavelength_grid_all_shifted, wave_grid_data, tau_grid):

    third, rows, cols = wavelength_grid_all_shifted.shape
    rebinned_tau = npempty((third*rows,) + wave_grid_data.shape, dtype=tau_grid.dtype)
    wave_shift_flat = wavelength_grid_all_shifted.reshape(third*rows,cols)
    tau_flat = tau_grid.reshape(third*rows,cols)
    for j in range(rows*third) :
        rebinned_tau[j] = compiled_interp(wave_grid_data,wave_shift_flat[j], tau_flat[j])

    return rebinned_tau.reshape(third,rows,len(wave_grid_data))
