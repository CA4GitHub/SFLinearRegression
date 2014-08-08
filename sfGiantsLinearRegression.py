#scipy can load .mat files
import scipy.io
#numpy does linear regression
import numpy
#matplotlib.pyplot plots
import matplotlib.pyplot as plt

theData = scipy.io.loadmat('/Users/flawton/Documents/USC/TA Machine Learning/SFGiants_Yr_WL_R_RA.mat')
theWL = theData['WL']
theRuns = theData['R']
theRunsAgainst = theData['RA']

#check dimensions of data match
if len(theWL)==len(theRuns) and len(theWL)==len(theRunsAgainst):
	numSeasons = len(theWL)
else:
	#print error and exit program if dimensions don't match
	print('Error: Dimension mismatch!')
	exit()

#checked rounding & saw small numerical differences from original data
#type(xdata1[0]) shows the data are numpy.float64 objects
#could convert to float objects, but don't need to do this to illustrate idea
xdata1 = [theRuns[i][0] for i in range(numSeasons)]
xdata2 = [theRunsAgainst[i][0] for i in range(numSeasons)]
xdata3 = list(numpy.array(xdata1, dtype = numpy.int16) - numpy.array(xdata2, dtype = numpy.int16))
ydata = [theWL[i][0] for i in range(numSeasons)]

# Aw = y
A1 = numpy.array( [xdata1, numpy.ones(numSeasons) ] )
w1 = numpy.linalg.lstsq(A1.T,ydata)[0]
print(w1)

plt.plot(xdata1,ydata,'ro')
plt.title('SF Win Percentage vs. Runs Scored By SF')
plt.xlabel('Runs Scored By SF')
plt.ylabel('SF Win Percentage')
plt.grid(True)
plt.show()

A2 = numpy.array( [xdata2, numpy.ones(numSeasons) ] )
w2 = numpy.linalg.lstsq(A2.T,ydata)[0]
print(w2)

plt.plot(xdata2,ydata,'bo')
plt.title('SF Win Percentage vs. Runs Scored Against SF')
plt.xlabel('Runs Scored Against SF')
plt.ylabel('SF Win Percentage')
plt.grid(True)
plt.show()

A3 = numpy.array( [xdata3, numpy.ones(numSeasons) ] )
w3 = numpy.linalg.lstsq(A3.T,ydata)[0]
print(w3)

plt.plot(xdata3,ydata,'bo')
plt.title('SF Win Percentage vs. SF Run Differential')
plt.xlabel('SF Run Differential')
plt.ylabel('SF Win Percentage')
plt.grid(True)
plt.show()
