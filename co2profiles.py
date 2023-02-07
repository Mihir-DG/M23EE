import numpy as np
#from matplotlib import pyplot as plt

#tp_profiles = np.load('../thermodynamic_profiles.npz')
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