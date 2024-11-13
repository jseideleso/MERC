star = 'WASP121' 

R_Sun = 69570000000. #cm
R_Jupiter = 7.1492*10**9 #cm
M_Jupiter = 1.89813*10**30 #g

if star == 'WASP121':
    M_planet = 1.157*M_Jupiter
    R_Star = 1.44*R_Sun
    R_0 = R_Jupiter*1.865
    period = 1.27492504

else:
    print('Star not recognised. Check data.')
    sys.exit()

line_dict={"NaI": [589.158364, 61600000, 0.641],"NaII": [589.7558147, 61400000, 0.32]}

masses = {'Na': 22.989769282*1.6726219*10**(-24), 'H2': 2.01588*1.6726219*10**(-24)}
solar_abundances = {'Na': 10**(-5.7)}




