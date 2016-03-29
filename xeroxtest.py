import csv
import matplotlib.pyplot as plt
import numpy
from numpy import genfromtxt
from sklearn import preprocessing
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import sys
from scipy import stats
from xeroxfuncs import outlierlabs, outliervitals, id_features, id_features
import pickle



print "Xeroxtest Running\n\n"
with open('global.pickle') as f:
    global_labs_min, global_labs_max, global_vitals_min, global_vitals_max, global_avg_feat = pickle.load(f)

#print global_labs_min, global_labs_max, global_vitals_min, global_vitals_max, global_avg_feat

with open('classifier.pickle') as f:
	clf = pickle.load(f)

print "Classifier Loaded ::"
print clf



#######	  TESTING STARTS

print "\n\nTESTING STARTS\n\n"

vitalsfile = sys.argv[1]
labsfile = sys.argv[2]
agecsv = sys.argv[3]

vitals = genfromtxt(vitalsfile, delimiter=',')
labs = genfromtxt(labsfile, delimiter=',')
age = genfromtxt(agecsv, delimiter=',')

output1 = vitals[numpy.where(vitals[:,8] == 1)]
print output1.shape
output = output1[:,:2]
print output.shape



outlierlabs(labs)
print "outlierlabs done"
outliervitals(vitals)
print "outliervitals done"
for j in [4,8,9,12,13,16,17,18,20,23,24]:
	labs[1:,j] = numpy.log(labs[1:,j])

for i in range(1,vitals.shape[0]):
   for j in range(2,8):
      if not numpy.isnan(vitals[i][j]):
         vitals[i][j] = (float(vitals[i][j])-global_vitals_min[j])/(global_vitals_max[j]-global_vitals_min[j])

for i in range(1,labs.shape[0]):
   for j in range(2,27):
      if not numpy.isnan(labs[i][j]):
         labs[i][j] = (float(labs[i][j])-global_labs_min[j])/(global_labs_max[j]-global_labs_min[j])

vitals_minmax = vitals[1:,:]
labs_minmax = labs[1:,:]

print "min max norm of labs vitals done"


vitals = vitals_minmax
labs = labs_minmax
age_train = age
age_train = age_train[1:,:]

print "\n\n FEATURE EXTRACTION STARTED \n\n"

featurevector= numpy.empty([0,252])
for i in range(output.shape[0]):
	id = output[i][0]
	#print i, id
	#print "\n\n"
	timelimit = output[i][1]
	vitals_this = vitals[numpy.where(numpy.logical_and(vitals[:,0] == id , vitals[:,1]<=timelimit))]
	labs_this = labs[numpy.where(numpy.logical_and(labs[:,0] == id , labs[:,1]<=timelimit))]
	age_this = age_train
	feat_id = id_features(id-4792, vitals_this, labs_this, age_this)
	featurevector = numpy.concatenate((featurevector,feat_id), axis=0)


for i in range(featurevector.shape[0]):
	for j in range(featurevector.shape[1]):
		if numpy.isnan(featurevector[i][j]):
			featurevector[i][j] = global_avg_feat[j]

print featurevector.shape
print output.shape
numpy.savetxt("TESTING_feature_new.csv", featurevector, delimiter=",")

print "\n\nFEATURES EXTRACTED\n\n"

features = featurevector

#features = genfromtxt("TESTING_feature_new.csv", delimiter=',')

feature = features[:,1:]
print feature.shape
print features.shape
print output.shape

print "\nPREDICTION STARTED\n"

prob = clf.predict_proba(feature)
#numpy.savetxt("prob.csv", prob, delimiter=",")


#prob = genfromtxt("prob.csv", delimiter=',')
print prob.shape
pred = numpy.zeros((prob.shape[0],1))
threshold = 0.69
print threshold
for i in range(prob.shape[0]):
	if prob[i][1]>threshold:
		pred[i][0]=1
	else:
		pred[i][0]=0 

print pred.shape

tosubmit = numpy.concatenate((output,pred), axis=1)
numpy.savetxt("output.csv", tosubmit, fmt='%i', delimiter=",")
print "output.csv generated and saved to disk"
print "end"

