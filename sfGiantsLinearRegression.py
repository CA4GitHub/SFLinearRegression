#scipy.io can load .mat files
import scipy.io
#numpy does linear regression
import numpy
#matplotlib.pyplot plots
import matplotlib.pyplot as plt

'''Use 95% confidence region = 1-alpha => alpha = 0.05
   Use two-sided test => t_alphaOver2 = t_0.025
   There are 131 data samples. => degrees of freedom = 131 - 2 = 129
   Use table to look up t_0.025 for 129 degrees of freedom.
   I used http://easycalculation.com/statistics/critical-t-test.php
   Note: could use one-sided test for test if slope is > 0 or < 0
'''
t_alphaOver2 = 1.9785


def SSX(x):
	#sum of squares of x
	xDiffSquared = [(x[i]-numpy.average(x))**2 for i in range(len(x))]
	return sum(xDiffSquared)

def SSE(y, yEst):
	#sum of squared errors
	yDiffSquared = [(y[i]-yEst[i])**2 for i in range(len(y))]
	return sum(yDiffSquared)

def coeffOfDeter(SSY,SSE):
	#SSY = SSX(y)
	rSquared = (SSY - SSE)/SSY
	return rSquared

#t-statistic for slope of simple linear regression
def tStatForSlope(slopeEst,s,ssX):
	#s**2 = sumSquareErrors(y)/( len(x) - 2 )
	T = slopeEst/( s/(ssX**0.5) )
	return T

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
yRegEst1 = [w1[0]*xdata1[i] + w1[1] for i in range(numSeasons)]
tStat1 = tStatForSlope(w1[0],(SSE(ydata,yRegEst1)/(numSeasons-2))**0.5,SSX(xdata1))
print(tStat1)
rSquared1 = coeffOfDeter(SSX(ydata),SSE(ydata,yRegEst1))
print(rSquared1)

plt.plot(xdata1,ydata,'ro',xdata1,yRegEst1,'b-')
plt.title('SF Win Percentage vs. Runs Scored By SF')
plt.xlabel('Runs Scored By SF')
plt.ylabel('SF Win Percentage')
plt.grid(True)
plt.show()

A2 = numpy.array( [xdata2, numpy.ones(numSeasons) ] )
w2 = numpy.linalg.lstsq(A2.T,ydata)[0]
print(w2)
yRegEst2 = [w2[0]*xdata2[i] + w2[1] for i in range(numSeasons)]
tStat2 = tStatForSlope(w2[0],(SSE(ydata,yRegEst2)/(numSeasons-2))**0.5,SSX(xdata2))
print(tStat2)
rSquared2 = coeffOfDeter(SSX(ydata),SSE(ydata,yRegEst2))
print(rSquared2)

plt.plot(xdata2,ydata,'ro',xdata2,yRegEst2,'b-')
plt.title('SF Win Percentage vs. Runs Scored Against SF')
plt.xlabel('Runs Scored Against SF')
plt.ylabel('SF Win Percentage')
plt.grid(True)
plt.show()

A3 = numpy.array( [xdata3, numpy.ones(numSeasons) ] )
w3 = numpy.linalg.lstsq(A3.T,ydata)[0]
print(w3)
yRegEst3 = [w3[0]*xdata3[i] + w3[1] for i in range(numSeasons)]
tStat3 = tStatForSlope(w3[0],(SSE(ydata,yRegEst3)/(numSeasons-2))**0.5,SSX(xdata3))
print(tStat3)
rSquared3 = coeffOfDeter(SSX(ydata),SSE(ydata,yRegEst3))
print(rSquared3)

plt.plot(xdata3,ydata,'ro',xdata3,yRegEst3,'b-')
plt.title('SF Win Percentage vs. SF Run Differential')
plt.xlabel('SF Run Differential')
plt.ylabel('SF Win Percentage')
plt.grid(True)
plt.show()

#plot coefficient of determination to compare models
plt.bar([1,2,3],[rSquared1, rSquared2, rSquared3],align='center')
plt.xticks([1,2,3],['SF Runs', 'Runs Scored Against SF', 'SF Run Differential'])
plt.title('Coefficient of Determination for Different Independent Variables')
plt.xlabel('Independent Variable')
plt.ylabel('Coefficient of Determination')
plt.grid(True)
plt.show()