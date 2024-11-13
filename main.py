def framework_contribution(v_rot,v_rad,T0,T1,XfracP,switchP,ww_wyt, wavelengths_list, index_combine, full_grid, T_id, wind_id,sectors):

    spec_rat={}
    atmo = {}

    if T_id == "iso":
        atmo = basic_functions.make_T_P_profile_iso(T0,XfracP)
    else:
        print("T profile Id not found. Options are 'iso' and 'grad'.")
        sys.exit()
    atmo = basic_functions.create_atmo(atmo,XfracP)

    opacity_z={}
    v_t=0.

  
    for item in ["NaI","NaII"]:
        opacity_z[item]= basic_functions.calculate_opacity(v_t,wavelengths_list[item],atmo,item)

    big_opacity= grid_create.make_combined_grid_opa(opacity_z,wavelengths_list, index_combine, full_grid)
    airshift_grid=data_comparison.air_model(full_grid)


    del wavelengths_list
    del opacity_z
    del full_grid


    cross_section_rayleigh = basic_functions.calculate_rayleigh_cross_section('H2',airshift_grid)
    
    number_of_layers = len(atmo['P'][:-1])
    
    contribution_pressure = []
    contribution_specs = []
    


    #loop over all layers and switch off
    for layer_number in range(number_of_layers):
    
        if wind_id == "none":
            spec_rat_full=  basic_functions.calculate_spectrum_ratio_base(atmo['T'],atmo['P'],atmo['z'],airshift_grid,airshift_grid,cross_section_rayleigh,big_opacity)
        elif wind_id == "dtn_ver_lat":
            spec_rat_full= basic_functions.calculate_spectrum_ratio_pattern_lat(sectors,atmo['T'],atmo['P'],atmo['z'],airshift_grid,airshift_grid,cross_section_rayleigh,big_opacity,v_rot,v_rad,switchP,wind_id,layer_number)
      
        else:
            print("invalid wind Id. Options are none, ver, dtn, srot, dtn_ver, srot_ver and the _lat counterparts.")
            sys.exit()


        spec_rat_rebinned = npinterp(ww_wyt,airshift_grid,spec_rat_full)
        
        contribution_pressure.append(atmo['P'][layer_number]/1013250.)
 
        contribution_specs.append(spec_rat_rebinned)
  
    return ww_wyt, contribution_pressure, contribution_specs
