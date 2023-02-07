from sympl import AdamsBashforth
from climt import (
	RRTMGLongwave, 
	RRTMGShortwave,
    EmanuelConvection,
    Frierson06LongwaveOpticalDepth, 
    GrayLongwaveRadiation,
    SimplePhysics, 
    DryConvectiveAdjustment, 
    SlabSurface,
    get_default_state,
    get_grid)
import climt
from netCDF4 import Dataset
import math
import numpy as np
import sympl
from datetime import timedelta
import matplotlib.pyplot as plt
import metpy.calc as calc
import csv
import os
from metpy.units import units
import metpy.calc as calc
import datetime
from metpy.units import units
import matplotlib.ticker as ticker

def net_energy_level_in_column(state,diagnostics,diff_acceptable):
	radPres = radiating_pressure(state,diff_acceptable)
	radHt = radPres[1]
	sb_const = 5.67e-08
	lw_up_ntat_OUT = (np.array(state['upwelling_longwave_flux_in_air']).flatten())[radHt]
	lw_up_surf_IN = sb_const * (((np.array(state['surface_temperature']).flatten())[0])**4)
	lw_down_ntat_IN = (np.array(state['downwelling_longwave_flux_in_air']).flatten())[radHt]
	lw_down_surf_OUT = (np.array(state['downwelling_longwave_flux_in_air']).flatten())[0]
	otherSurfFluxes_IN = (np.array(state['surface_upward_latent_heat_flux'] + state['surface_upward_sensible_heat_flux']).flatten())[0]
	sw_up_ntat_OUT = (np.array(state['upwelling_shortwave_flux_in_air']).flatten())[radHt]
	sw_up_surf_IN = (np.array(state['upwelling_shortwave_flux_in_air']).flatten())[0]
	sw_down_ntat_IN = (np.array(state['downwelling_shortwave_flux_in_air']).flatten())[radHt]
	sw_down_surf_OUT = (np.array(state['downwelling_shortwave_flux_in_air']).flatten())[0]
	#sw_up_surf_reflected_IN = albedo * (np.array(state['downwelling_shortwave_flux_in_air']).flatten())[0]
	fluxesIn = [lw_up_surf_IN,lw_down_ntat_IN,otherSurfFluxes_IN,sw_up_surf_IN,sw_down_ntat_IN]#,sw_up_surf_reflected_IN]
	fluxesOut = [lw_up_ntat_OUT,lw_down_surf_OUT,sw_up_ntat_OUT,sw_down_surf_OUT]
	netEn = sum(fluxesIn) - sum(fluxesOut)
	return netEn

def cleaningUp():
	try:
		CSVs = 'output_runModel'
		graphs = 'graphs'
		foldersMain = [CSVs, graphs]
		for item in foldersMain:
			for file in os.listdir(item):
				os.remove(os.path.join(item,file))
	except:
		pass
	return 0.

def gen_co2Inputs():
	try:
		mol_profiles = np.load('molecule_profiles.npz')	
		defaultCO2 = mol_profiles['carbon_dioxide'][:, np.newaxis, np.newaxis]
		print(len(defaultCO2))
		avg = (sum(defaultCO2)/len(defaultCO2))[0][0]
		linIncProfile = [[[avg*0.5]]]
		for elem in range(59):
			a = [[linIncProfile[-1][0][0] + (avg*1.5 - avg*0.5)/len(defaultCO2)]]
			linIncProfile.append(a)
		print(linIncProfile)
		linIncProfile = np.array(linIncProfile)
		linDecProfile = linIncProfile[::-1]
		co2distrs = np.array([defaultCO2, linIncProfile, linDecProfile])
		np.save('co2distrs', co2distrs, allow_pickle = True)
	except:
		pass

def runningModel():
    # Initialize components
	sw = RRTMGShortwave()
	lw = RRTMGLongwave()
	surface = SlabSurface()
	convection = EmanuelConvection()
	boundary_layer = SimplePhysics()
	dryConvection = DryConvectiveAdjustment()
	carbonprofiles = np.load('co2distrs.npy', allow_pickle = True)
	print(carbonprofiles.shape)

    # Set up model state.
	timestep = timedelta(minutes=10)
	grid = get_grid(nx=1, ny=1, nz=60)
	state = get_default_state([lw, sw, surface,
	boundary_layer, convection, dryConvection], grid_state=grid)
	albedo = 0.3
	state['surface_temperature'][:] = 280
	state['air_temperature'][:] = 240
	state['zenith_angle'].values[:] = 76/90*np.pi/2
	state['surface_albedo_for_direct_near_infrared'].values[:] = albedo * 1.5
	state['ocean_mixed_layer_thickness'].values[:] = 1.
	state['surface_albedo_for_direct_shortwave'][:] = albedo
	state['surface_albedo_for_diffuse_shortwave'][:] = np.sin((np.pi)/3) * albedo
	state['area_type'][:] = 'sea'
	tp_profiles = np.load('thermodynamic_profiles.npz')
	mol_profiles = np.load('molecule_profiles.npz')
	state['air_pressure'].values[:] = tp_profiles['air_pressure'][:, np.newaxis, np.newaxis]
	state['air_pressure_on_interface_levels'].values[:] = tp_profiles['interface_pressures'][:, np.newaxis, np.newaxis]
	state['specific_humidity'].values[:] = mol_profiles['specific_humidity'][:, np.newaxis, np.newaxis]*1e-3
	# CHANGE NEXT LINE FOR CO2 DISTRIBUTION USED (defaultcarbon, linInc, linDec);
	state['mole_fraction_of_carbon_dioxide_in_air'].values[:] = carbonprofiles[0]
	print(carbonprofiles[1])
	state['mole_fraction_of_ozone_in_air'].values[:] = mol_profiles['ozone'][:, np.newaxis, np.newaxis]
	state.update()
	print(sum(list(state['mole_fraction_of_carbon_dioxide_in_air'].values[:])))

	time_stepper = AdamsBashforth([lw, sw, surface, convection])

	# Model variables
	diff_acceptable = 0.2
	airPressure_vertCoord = np.array(state['air_pressure_on_interface_levels']).flatten()
	time = datetime.datetime(2020,1,1,0,0,0) # In months (Add 1/168 for each timedelta jump)
	stop = False
	counter = 0
	errorMargin = 1.5
	while stop == False:
		# Updating TendencyComponents
		diagnostics, state = time_stepper(state,timestep)
		state.update(diagnostics)
		counter += 1
		time = time + timestep

		# Updating boundary layer
		boundaryDiagnostics, new_state = boundary_layer(state, timestep)
		state.update(new_state)
		state.update(boundaryDiagnostics)

		#Updating convective adjustment
		convectiveAdjustmentDiagnostics, new_state = dryConvection(state, timestep)
		state.update(new_state)
		state.update(convectiveAdjustmentDiagnostics)

		state['eastward_wind'][:] = 3.
		state.update()

		# Updating appropriate quantities every month
		if counter % 42*4 == 0:
			print(counter)
			print(net_energy_level_in_column(state,diagnostics,diff_acceptable))
		          
		if counter % 500 == 0:
			print("AIR TEMPERATURE")
			print(np.array(state['air_temperature'][:]).flatten())
			print("\n LW NET FLUX")
			print(np.array(state['upwelling_longwave_flux_in_air'][:] - state['downwelling_longwave_flux_in_air'][:]).flatten())
			print("\n SW NET FLUX")
			print(np.array(state['upwelling_shortwave_flux_in_air'][:] - state['downwelling_shortwave_flux_in_air'][:]).flatten())
			print("\n SURFACE TEMPERATURE")
			print(state['surface_temperature'])
			print("\n SURFACE FLUXES")
			print((np.array(state['surface_upward_latent_heat_flux'] + state['surface_upward_sensible_heat_flux']).flatten())[0])
			print("\n SW UP FLUX")
			print(np.array(state['upwelling_shortwave_flux_in_air'][:]).flatten()[-1])
			print("\n SW DOWN FLUX")
			print(np.array(state['downwelling_shortwave_flux_in_air'][:]).flatten()[-1])
        
		# Checking stopping criteria
		if (abs(net_energy_level_in_column(state,diagnostics,diff_acceptable)) < errorMargin):
			stop = True

	print("AIR TEMPERATURE")
	print(np.array(state['air_temperature'][:]).flatten())
	print("\n LW NET FLUX")
	print(np.array(state['upwelling_longwave_flux_in_air'][:] - state['downwelling_longwave_flux_in_air'][:]).flatten())
	print("\n SW NET FLUX")
	print(np.array(state['upwelling_shortwave_flux_in_air'][:] - state['downwelling_shortwave_flux_in_air'][:]).flatten())
	print("\n SURFACE TEMPERATURE")
	print(state['surface_temperature'])
	print("\n SURFACE FLUXES")
	print((np.array(state['surface_upward_latent_heat_flux'] + state['surface_upward_sensible_heat_flux']).flatten())[0])
	print("\n SW UP FLUX")
	print(np.array(state['upwelling_shortwave_flux_in_air'][:]).flatten())
    
    # Calculating output quantities
	timeTaken = time - datetime.datetime(2020,1,1,0,0,0)
	lwFluxNet = np.array(diagnostics['upwelling_longwave_flux_in_air'] - 
      diagnostics['downwelling_longwave_flux_in_air']).flatten()
	swFluxNet = np.array(diagnostics['upwelling_shortwave_flux_in_air'] - 
      diagnostics['downwelling_shortwave_flux_in_air']).flatten()
	sw_heatRate = np.array(diagnostics['air_temperature_tendency_from_shortwave']).flatten()
	lw_heatRate = np.array(diagnostics['air_temperature_tendency_from_longwave']).flatten()
	airTemperatureProf = (np.array(state['air_temperature'])).flatten()
	airPressure_vertCoord = np.array(state['air_pressure_on_interface_levels']).flatten()
	interface_airPressure_vertCoord = np.array(state['air_pressure']).flatten()
	olr = (np.array(diagnostics['upwelling_longwave_flux_in_air'][-1]).flatten())[0]
	convection_heatRate = np.array(diagnostics['air_temperature_tendency_from_convection']).flatten()

	return state, olr, timeTaken, lwFluxNet, swFluxNet, sw_heatRate, lw_heatRate, convection_heatRate, airTemperatureProf, interface_airPressure_vertCoord, airPressure_vertCoord

def output_to_csv(timeTaken, lwFluxNet, swFluxNet, sw_heatRate, lw_heatRate, convection_heatRate, airTemperatureProf, interface_airPressure_vertCoord, airPressure_vertCoord):
	with open('output_runModel/equilibrium.csv', mode='w') as equilibriumCSV:
		equilibriumWriter = csv.writer(equilibriumCSV)
		equilibriumWriter.writerow(lwFluxNet)
		equilibriumWriter.writerow(swFluxNet)
		equilibriumWriter.writerow(sw_heatRate)
		equilibriumWriter.writerow(lw_heatRate)
		equilibriumWriter.writerow(convection_heatRate)
		equilibriumWriter.writerow(airTemperatureProf)
		equilibriumWriter.writerow(str(timeTaken))
		equilibriumWriter.writerow(interface_airPressure_vertCoord)
		equilibriumWriter.writerow(airPressure_vertCoord)
	
	return 0.

def eqProfs():
	dataArr = []
	with open('output_runModel/equilibrium.csv', 'r') as equilibriumFile:
		csvRead = csv.reader(equilibriumFile)
		for row in csvRead:
			dataArr.append(row)
	dataArr =  [ele for ele in dataArr if ele != []] 
	lwFluxNet, swFluxNet, sw_heatRate, lw_heatRate, convection_heatRate, airTemperatureProf, interface_airPressure_vertCoord, airPressure_vertCoord = dataArr[0],dataArr[1],dataArr[2], dataArr[3], dataArr[4], dataArr[5], dataArr[7], dataArr[8]
	
	# POTENTIAL TEMPERATURE
	interface_airPressure_vertCoord = [round((float(ele))) for ele in interface_airPressure_vertCoord]
	airTemperatureProf = [round(float(ele),2) for ele in airTemperatureProf]
	potentialTemperatures = []
	for i in range(60):
		pressure = interface_airPressure_vertCoord[i]
		pressure  = units.Quantity(pressure,"pascal")
		temperature = airTemperatureProf[i]
		temperature = units.Quantity(temperature,"kelvin")
		potentialT = calc.potential_temperature(pressure,temperature)
		potentialTemperatures.append(potentialT.magnitude)
	potentialTemperatures = np.array(potentialTemperatures)
	np.save("graphs/RCE_pT",potentialTemperatures)
	
	# PREP FOR MAIN GRAPH
	interface_airPressure_vertCoord = [round((float(ele)/100),2) for ele in interface_airPressure_vertCoord]
	airPressure_vertCoord = [round((float(ele)/100),2) for ele in airPressure_vertCoord] # Conversion to mbar
	timeTaken = ''.join(dataArr[5])

	fig = plt.figure(figsize=(12,12),dpi=1000)

	# SHORTWAVE FLUX DIVERGENCE

	swFluxNet = [round(float(ele),2) for ele in swFluxNet]
	ax = fig.add_subplot(2,2,1)
	ax.set_yscale('log')
	ax.yaxis.set_major_locator(ticker.MultipleLocator(6))
	ax.yaxis.set_ticks(np.linspace(1000,10,6))
	ax.axes.invert_yaxis()
	ax.set_ylim(1e3, 5.)
	ax.set_xticks(np.linspace(-160,-240,5))
	ax.plot(swFluxNet,airPressure_vertCoord,'-o')
	ax.set_xlabel("A - Shortwave Flux Divergence (" + r'W/m$^2$' + ")")
	ax.set_ylabel("Pressure (mbar)")
	ax.grid()
	ax.set_yticklabels(np.linspace(1000,10,6))

	# LONGWAVE FLUX DIVERGENCE

	lwFluxNet = [round(float(ele),2) for ele in lwFluxNet]
	ax = fig.add_subplot(2,2,2)
	ax.set_yscale('log')
	ax.yaxis.set_major_locator(ticker.MultipleLocator(6))
	ax.yaxis.set_ticks(np.linspace(1000,10,6))
	ax.axes.invert_yaxis()
	ax.set_ylim(1e3, 5.)
	ax.set_xticks(np.linspace(100,240,5))
	ax.plot(lwFluxNet,airPressure_vertCoord,'-o')
	ax.set_xlabel("B - Longwave Flux Divergence ("+ r'W/m$^2$' + ")")
	ax.set_ylabel("Pressure (mbar)")
	ax.grid()
	ax.set_yticklabels(np.linspace(1000,10,6))

	# HEATING RATES

	sw_heatRate = [round(float(ele),2) for ele in sw_heatRate]
	lw_heatRate = [round(float(ele),2) for ele in lw_heatRate]
	ax = fig.add_subplot(2,2,3)
	ax.set_yscale('log')
	ax.yaxis.set_major_locator(ticker.MultipleLocator(6))
	ax.yaxis.set_ticks(np.linspace(1000,10,6))
	ax.axes.invert_yaxis()
	ax.set_ylim(1e3, 5.)
	ax.set_xticks(np.linspace(-30,30,13))
	ax.plot(sw_heatRate,interface_airPressure_vertCoord,'-o',color = "orange", label = "SW")
	ax.plot(lw_heatRate,interface_airPressure_vertCoord,'-o', label = "LW")
	ax.plot(np.array(sw_heatRate) + np.array(lw_heatRate), interface_airPressure_vertCoord, '-o', color='green', label = "Net")
	ax.set_xlabel("C - Heating Rates (K)")
	ax.legend(loc='upper right')
	ax.set_ylabel("Pressure (mbar)")
	ax.grid()
	ax.set_yticklabels(np.linspace(1000,10,6))

	# AIR TEMPERATURE

	ax = fig.add_subplot(2,2,4)
	ax.set_yscale('log')
	ax.yaxis.set_major_locator(ticker.MultipleLocator(6))
	ax.yaxis.set_ticks(np.linspace(1000,10,6))
	ax.axes.invert_yaxis()
	ax.set_ylim(1e3, 5)
	ax.set_xticks(np.linspace(200,300,6))
	ax.plot(airTemperatureProf,interface_airPressure_vertCoord, '-o')
	ax.set_xlabel("D - Air Temperature (K)")
	ax.set_ylabel("Pressure (mbar)")
	ax.grid()
	ax.set_yticklabels(np.linspace(1000,10,6))
	plt.savefig("graphs/baseRCEModel.png")
	return 0.

def constituentPlots():

	rad_sw = RRTMGShortwave()
	rad_lw = RRTMGLongwave()
	surface = SlabSurface()

	timestep = datetime.timedelta(hours=80)
	grid = get_grid(nx=1, ny=1, nz=60)
	state = get_default_state([rad_lw,rad_sw,surface], grid_state = grid)
	time_stepper = AdamsBashforth([rad_sw, rad_lw, surface])

	tp_profiles = np.load('thermodynamic_profiles.npz')
	mol_profiles = np.load('molecule_profiles.npz')
	state['air_pressure'].values[:] = tp_profiles['air_pressure'][:, np.newaxis, np.newaxis]
	state['mole_fraction_of_carbon_dioxide_in_air'].values[:] = mol_profiles['carbon_dioxide'][:, np.newaxis, np.newaxis]
	state['mole_fraction_of_ozone_in_air'].values[:] = mol_profiles['ozone'][:, np.newaxis, np.newaxis]
	state.update()

	specHumidity = list(state['specific_humidity'].values[:].flatten())
	air_pressure = list(state['air_pressure'].values[:].flatten())
	air_pressure = [ele * 1e-2 for ele in air_pressure]

	co2 = list(state['mole_fraction_of_carbon_dioxide_in_air'].values[:].flatten())
	o3 = list(state['mole_fraction_of_ozone_in_air'].values[:].flatten())

	fig = plt.figure(figsize=(12,6),dpi=1000)

	ax = fig.add_subplot(1,2,1)
	ax.set_yscale('log')
	ax.axes.invert_yaxis()
	ax.plot(o3,air_pressure,'-o')
	ax.set_xlabel("A - " + r'$O_3$' + " Distribution (Mole Fraction)")
	ax.set_ylabel("Pressure (mbar)")
	ax.grid()

	ax = fig.add_subplot(1,2,2)
	ax.set_yscale('log')
	ax.axes.invert_yaxis()
	ax.plot(co2,air_pressure,'-o')
	ax.set_xlabel("B - " + r'$CO_2$' + " Distribution (Mole Fraction)")
	ax.set_ylabel("Pressure (mbar)")
	ax.grid()

	plt.savefig("graphs/constituents")

	return 0.

def compute_climatological_normal():

	latitudes = np.linspace(-np.pi/2, np.pi/2, 73)

	olr = Dataset("olr.mon.mean.nc","r",format="NETCDF4")
	olr = np.array(olr.variables['olr'])
	olr = np.array(olr[-7-12*30:-7], dtype=np.float64)
	olr = np.mean(olr.mean(axis=0),axis=1)
	olr = np.average(olr, weights=np.cos(latitudes))

	upFlux = Dataset("ulwrf.sfc.mon.mean.nc","r",format="NETCDF4")
	upFlux = np.array(upFlux.variables['ulwrf'])
	upFlux = np.array(upFlux[-7-12*30:-7],dtype=np.float64)

def surf_airBdry_tempDiff(state):
  	return (state['surface_temperature'] - state['air_temperature'])[0][0][0]

def radiating_pressure(state,diff_acceptable):
	upFlux = np.array(state['upwelling_longwave_flux_in_air']).flatten()
	int_level = 0
	for i in range(1,29):
		if abs(upFlux[i]-upFlux[i-1]) < diff_acceptable:
			int_level = i
			break
		else:
			int_level = 29
	return (np.array(state['air_pressure_on_interface_levels']).flatten())[int_level],int_level

def main():
	cleaningUp()
	print('abc')
	state, olr, timeTaken, lwFluxNet, swFluxNet, sw_heatRate, lw_heatRate, convection_heatRate, airTemperatureProf, interface_airPressure_vertCoord, airPressure_vertCoord = runningModel()
	print(airTemperatureProf)
	output_to_csv(timeTaken, lwFluxNet, swFluxNet, sw_heatRate, lw_heatRate, convection_heatRate,
		airTemperatureProf, interface_airPressure_vertCoord, airPressure_vertCoord)
	eqProfs()



	
if __name__ == "__main__":
	main()

"""
RCE CONTROLS
___________

atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels
atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels
surface_air_pressure
time
air_pressure
air_pressure_on_interface_levels
longitude
latitude
height_on_ice_interface_levels
air_temperature
surface_temperature
specific_humidity
mole_fraction_of_ozone_in_air
mole_fraction_of_carbon_dioxide_in_air
mole_fraction_of_methane_in_air
mole_fraction_of_nitrous_oxide_in_air
mole_fraction_of_oxygen_in_air
mole_fraction_of_cfc11_in_air
mole_fraction_of_cfc12_in_air
mole_fraction_of_cfc22_in_air
mole_fraction_of_carbon_tetrachloride_in_air
surface_longwave_emissivity
longwave_optical_thickness_due_to_cloud
longwave_optical_thickness_due_to_aerosol
cloud_area_fraction_in_atmosphere_layer
mass_content_of_cloud_ice_in_atmosphere_layer
mass_content_of_cloud_liquid_water_in_atmosphere_layer
cloud_ice_particle_size
cloud_water_droplet_radius
zenith_angle
surface_albedo_for_direct_shortwave
surface_albedo_for_direct_near_infrared
surface_albedo_for_diffuse_near_infrared
surface_albedo_for_diffuse_shortwave
shortwave_optical_thickness_due_to_cloud
cloud_asymmetry_parameter
cloud_forward_scattering_fraction
single_scattering_albedo_due_to_cloud
shortwave_optical_thickness_due_to_aerosol
aerosol_asymmetry_parameter
single_scattering_albedo_due_to_aerosol
aerosol_optical_depth_at_55_micron
solar_cycle_fraction
flux_adjustment_for_earth_sun_distance
downwelling_longwave_flux_in_air
downwelling_shortwave_flux_in_air
upwelling_longwave_flux_in_air
upwelling_shortwave_flux_in_air
surface_upward_latent_heat_flux
surface_upward_sensible_heat_flux
surface_thermal_capacity
surface_material_density
upward_heat_flux_at_ground_level_in_soil
heat_flux_into_sea_water_due_to_sea_ice
area_type
soil_layer_thickness
ocean_mixed_layer_thickness
heat_capacity_of_soil
sea_water_density
northward_wind
eastward_wind
surface_specific_humidity"""