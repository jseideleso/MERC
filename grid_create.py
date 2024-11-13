import numpy as np
from time import time
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy import ndarray,interpolate
from bisect import bisect_left
from numpy import interp as npinterp
from numpy import empty as npempty
import myconstants 
import matplotlib.pyplot as plt


def make_grid(item,star):
    line_center = myconstants.line_dict[item][0]

    limits_list = [(0,line_center-0.05)]
    limits_list.append((0,line_center+0.05))

   
    limits_list.append((1,line_center+0.1))
    limits_list.append((1,line_center-0.1))


    limits_list.append((2,588.5416992187501))
    limits_list.append((2,590.0001953125001))

    sorted_limits = sorted(limits_list,key=itemgetter(1))

    wavelength_grid = list()

    for i in range(len(sorted_limits)-1):
       
        zone_specifier = sorted_limits[i][0]+sorted_limits[i+1][0]
        bin_number = 0.
        if zone_specifier == 3:
            bin_number = 1./0.015
        elif zone_specifier == 1:
            bin_number = 1./0.01
        elif zone_specifier == 0:
            bin_number = 1./0.0005
        else:
             print("grid_create.make_grid(): In unspecified zone! Check grid construction.")

        if i == len(sorted_limits)-1:
            wavelength_grid.extend(np.linspace(sorted_limits[i][1],sorted_limits[i+1][1],num=int(bin_number)))
        else:
            wavelength_grid.extend(np.linspace(sorted_limits[i][1],sorted_limits[i+1][1],num=int(bin_number),endpoint=False))

    return np.asarray(wavelength_grid)


def make_combined_grid_wave(dict_grid):
 
    switch = (myconstants.line_dict["NaII"][0]-myconstants.line_dict["NaI"][0])/2.
    wavelength_switch = myconstants.line_dict["NaII"][0]-switch
    index = {}
    for item in ["NaI","NaII"]:
        index[item] = bisect_left(dict_grid[item],wavelength_switch)
    wave_length = index["NaI"] + len(dict_grid["NaII"])-index["NaII"]
    combined_grid = np.zeros(wave_length)

    combined_grid[:index["NaI"]] = dict_grid["NaI"][:index["NaI"]]
    combined_grid[index["NaI"]:]=dict_grid["NaII"][index["NaII"]:]
    return combined_grid, index

def make_combined_grid_opa(dict_opacities, dict_grid, index, combined_grid):

    wave_length = index["NaI"] + len(dict_grid["NaII"])-index["NaII"]

    combined_opacity = np.zeros(shape=(np.size(dict_opacities["NaI"],0),wave_length))

    rebin_sec1 = interpolate.interp1d( dict_grid["NaII"][:index["NaII"]], dict_opacities["NaII"][:,:index["NaII"]],copy=False, bounds_error=False, fill_value='extrapolate')
    rebinned_sec1 = rebin_sec1(combined_grid[:index["NaI"]])
    rebin_sec2 = interpolate.interp1d( dict_grid["NaI"][index["NaI"]:], dict_opacities["NaI"][:,index["NaI"]:],copy=False, bounds_error=False, fill_value='extrapolate')
    rebinned_sec2 = rebin_sec2(combined_grid[index["NaI"]:])

    combined_opacity[:,:index["NaI"]] = np.add(dict_opacities["NaI"][:,:index["NaI"]],rebinned_sec1)
    combined_opacity[:,index["NaI"]:] = np.add(dict_opacities["NaII"][:,index["NaII"]:],rebinned_sec2)

    return combined_opacity

def interp_1(z, x, y) :
    rows, cols = x.shape
    out = npempty((rows,) + z.shape, dtype=y.dtype)
    for j in range(rows) :
        out[j] = npinterp(z,x[j], y[j])
    return out

def interp_2(x, y, z) :
    rows, cols = x.shape
    row_idx = np.arange(rows).reshape((rows,) + (1,) * z.ndim)
    col_idx = np.argmax(x.reshape(x.shape + (1,) * z.ndim) > z, axis=1) - 1
    ret = np.multiply(np.divide((y[row_idx, col_idx + 1] - y[row_idx, col_idx]),(x[row_idx, col_idx + 1] - x[row_idx, col_idx])),(z - x[row_idx, col_idx]))+y[row_idx, col_idx]

    return ret











