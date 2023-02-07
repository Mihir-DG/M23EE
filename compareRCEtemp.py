import csv
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open('finalData/equilibrium.csv', 'r') as equilibriumFile:
		dataArr = []
		csvRead = csv.reader(equilibriumFile)
		for row in csvRead:
			dataArr.append(row)
		dataArr =  [ele for ele in dataArr if ele != []] 
		base_airTemperatureProf, interface_airPressure_vertCoord, airPressure_vertCoord = dataArr[5], dataArr[7], dataArr[8]

with open('finalData/linInc-equilibrium.csv', 'r') as equilibriumFile:
		dataArr1 = []
		csvRead = csv.reader(equilibriumFile)
		for row in csvRead:
			dataArr1.append(row)
		dataArr1 =  [ele for ele in dataArr1 if ele != []] 
		linInc_airTemperatureProf = dataArr1[5]

with open('finalData/linDec-equilibrium.csv', 'r') as equilibriumFile:
		dataArr2 = []
		csvRead = csv.reader(equilibriumFile)
		for row in csvRead:
			dataArr2.append(row)
		dataArr2 =  [ele for ele in dataArr2 if ele != []] 
		linDec_airTemperatureProf = dataArr2[5]

print(linInc_airTemperatureProf)
print(linDec_airTemperatureProf)
print(base_airTemperatureProf)


interface_airPressure_vertCoord = [round(float(ele),2) for ele in interface_airPressure_vertCoord]
base_airTemperatureProf = [round(float(ele),2) for ele in base_airTemperatureProf]
linInc_airTemperatureProf = [round(float(ele),2) for ele in linInc_airTemperatureProf]
linDec_airTemperatureProf = [round(float(ele),2) for ele in linDec_airTemperatureProf]
fig = plt.figure(figsize=(4,8), constrained_layout=True, dpi=1000)#figsize=(2,4))#,dpi=1000)
ax = fig.add_subplot(1,1,1)

#ax.axes.invert_yaxis()
ax.set_yscale('log')
ax.yaxis.set_major_locator(ticker.MultipleLocator(6))
ax.yaxis.set_ticks(np.linspace(1e5,2000,6))
#ax.axes.invert_yaxis()
ax.set_ylim(1e5, 10)
ax.set_xticks(np.linspace(200,300,6))
ax.plot(base_airTemperatureProf,interface_airPressure_vertCoord, '-o')
ax.plot(linInc_airTemperatureProf,interface_airPressure_vertCoord, '--')
ax.plot(linDec_airTemperatureProf,interface_airPressure_vertCoord, '-+')
ax.set_xlabel("D - Air Temperature (K)")
ax.set_ylabel("Pressure (mbar)")
ax.grid()
ax.set_yticklabels(np.linspace(1000,10,6))
#plt.show()
plt.savefig("graphs/baseRCEModel.png")