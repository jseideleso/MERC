import numpy as np
from scipy.constants import c
from bisect import bisect

def air_index(wavelength, t=15., p=760.):
    n = (1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t) * (64.328 + 29498.1/(146-(1e3/wavelength)**2) + 255.4/(41-(1e3/wavelength)**2)))+1
    return n

def air_model(wave_model):
  
    wave_model_air = np.true_divide(wave_model,air_index(wave_model[0]))

    return wave_model_air


def get_normalisation_zones(wave_grid):
    bin_limits = [0,0,0,0]
    bin_limits[0]=bisect(wave_grid,587.0)
    bin_limits[1]=bisect(wave_grid,588.222)
    bin_limits[2]=bisect(wave_grid,590.324)
    bin_limits[3]=bisect(wave_grid,591.6)
    return bin_limits


def get_norm_average(bin_limits, wave_grid, data):
    data_excerpt1 = data[bin_limits[0]::bin_limits[1]]
    data_excerpt2 = data[bin_limits[2]::bin_limits[3]]
    average = (np.sum(data_excerpt1)+np.sum(data_excerpt2))/(len(data_excerpt1)+len(data_excerpt2))
    return average

def get_norm_offset(average_data, average_model):
    offset = average_data - average_model
    return offset

def grid_shift(wave,wavedata,data,model):
    center_data = np.argmin(data)
    center_model = np.argmin(model)
    shift_grid = wave+(wavedata[center_data]-wave[center_model])
    return shift_grid
