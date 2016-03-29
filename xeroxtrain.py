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
import pickle
import sys
from scipy import stats

def outlierlabs(a):
	for i in range(1, a.shape[0]):
		if not numpy.isnan(a[i][3]):
			if a[i][3]>150:
				a[i][3] = a[i][3]-50

		if not numpy.isnan(a[i][4]):
			a[i][4] = float(a[i][4])/10

		if not numpy.isnan(a[i][5]):
			if a[i][5]<50:
				a[i][5] = numpy.nan
			
			if a[i][5]>200:
				a[i][5] = numpy.nan

		if not numpy.isnan(a[i][6]):
			if a[i][6]>20:
				a[i][6] = numpy.nan

		if not numpy.isnan(a[i][8]):
			if a[i][8]>200:
				a[i][8] = numpy.nan

		if not numpy.isnan(a[i][9]):
			if a[i][9]>20:
				a[i][9] = numpy.nan

		if not numpy.isnan(a[i][10]):
			if a[i][10]>25:
				a[i][10] = numpy.nan

		if not numpy.isnan(a[i][12]):
			if a[i][12]>1400:
				a[i][12] = numpy.nan

		if not numpy.isnan(a[i][13]):
			if a[i][13]>30:
				a[i][13] = numpy.nan

		if not numpy.isnan(a[i][21]):
			if a[i][21]>100:
				a[i][21] = numpy.nan

		if not numpy.isnan(a[i][23]):
			if a[i][23]>1000:
				a[i][23] = numpy.nan

def outliervitals(b):
	for i in range(1, b.shape[0]):
		if not numpy.isnan(b[i][2]):
			if b[i][2]>250:
				b[i][2] = numpy.nan

		if not numpy.isnan(b[i][4]):
			if b[i][4]>=200:
				b[i][4] = numpy.nan
			if b[i][4]<25:
				b[i][4] = numpy.nan

		if not numpy.isnan(b[i][5]):
			if b[i][5]>=80:
				b[i][5] = numpy.nan

		if not numpy.isnan(b[i][6]):
			if b[i][6]>100:
				b[i][6] = 100

		if not numpy.isnan(b[i][7]):
			if b[i][7]<85:
				b[i][7] = b[i][7]+10
			if b[i][7]>109:
				b[i][7] = 104

def id_features(i,idvitals,idlabs,age_train):
	print i
	feat_id = numpy.zeros(4)
	feat_id[0] = i
	agg = age_train[numpy.where(age_train[:,0] == id)]
	feat_id[1] = age_train[i-1,1]
	feat_id[2] = float(idvitals[idvitals.shape[0]-1][1])/3660
	idvitals_icu = idvitals[numpy.where(idvitals[:,8] == 1)]
	if idvitals_icu.shape[0]>0:
		feat_id[3] = (float(idvitals_icu[idvitals_icu.shape[0]-1][1])-idvitals_icu[0][1])/3660

	min_vitals = numpy.nanmin(idvitals, axis = 0)
	max_vitals =  numpy.nanmax(idvitals, axis = 0)
	avg_vitals =  numpy.nanmean(idvitals, axis = 0)
	median_vitals = stats.nanmedian(idvitals, axis=0)
	std_vitals = numpy.nanstd(idvitals, axis=0)
	first_vitals = idvitals[0,:]
	last_vitals = idvitals[idvitals.shape[0]-1,:]
	number_vitals = idvitals[idvitals.shape[0]-1,:]
	for j in range(2,8):
		for i in range(idvitals.shape[0]):
			if not numpy.isnan(idvitals[i][j]):
				first_vitals[j] = idvitals[i][j]

	for j in range(2,8):
		for i in range(1,idvitals.shape[0]):
			if not numpy.isnan(idvitals[idvitals.shape[0]-i][j]):
				last_vitals[j] = idvitals[idvitals.shape[0]-i][j]

	for j in range(2,8):
		for i in range(1,idvitals.shape[0]):
			number_vitals[j] = numpy.count_nonzero(~numpy.isnan(idvitals[i][j]))


	min_labs = numpy.nanmin(idlabs, axis = 0)
	max_labs =  numpy.nanmax(idlabs, axis = 0)
	avg_labs =  numpy.nanmean(idlabs, axis = 0)
	median_labs = stats.nanmedian(idlabs, axis=0)
	std_labs = numpy.nanstd(idlabs, axis=0)
	first_labs = idlabs[0,:]
	last_labs = idlabs[idlabs.shape[0]-1,:]
	number_labs = idlabs[idlabs.shape[0]-1,:]
	for j in range(2,27):
		for i in range(idlabs.shape[0]):
			if not numpy.isnan(idlabs[i][j]):
				first_labs[j] = idlabs[i][j]

	for j in range(2,27):
		for i in range(1,idlabs.shape[0]):
			if not numpy.isnan(idlabs[idlabs.shape[0]-i][j]):
				last_labs[j] = idlabs[idlabs.shape[0]-i][j]

	for j in range(2,27):
		for i in range(1,idlabs.shape[0]):
			number_labs[j] = numpy.count_nonzero(~numpy.isnan(idlabs[i][j]))

	feat_id = numpy.append(feat_id, min_vitals[2:8])
	feat_id = numpy.append(feat_id, max_vitals[2:8])
	feat_id = numpy.append(feat_id, avg_vitals[2:8])
	feat_id = numpy.append(feat_id, median_vitals[2:8])
	feat_id = numpy.append(feat_id, std_vitals[2:8])
	feat_id = numpy.append(feat_id, first_vitals[2:8])
	feat_id = numpy.append(feat_id, last_vitals[2:8])
	feat_id = numpy.append(feat_id, number_vitals[2:8])

	feat_id = numpy.append(feat_id, min_labs[2:])
	feat_id = numpy.append(feat_id, max_labs[2:])
	feat_id = numpy.append(feat_id, avg_labs[2:])
	feat_id = numpy.append(feat_id, median_labs[2:])
	feat_id = numpy.append(feat_id, std_labs[2:])
	feat_id = numpy.append(feat_id, first_labs[2:])
	feat_id = numpy.append(feat_id, last_labs[2:])
	feat_id = numpy.append(feat_id, number_labs[2:])

	#print feat_id.shape
	feat_id = numpy.reshape(feat_id,(1,252))
	#print feat_id.shape
	#print feat.shape
	return feat_id

def features(vitals, labs, age_train):
	feat = numpy.empty([0,252])
	for i in range(1,3595):
		
		idvitals = vitals[numpy.where(vitals[:,0] == i)]
		idlabs = labs[numpy.where(labs[:,0] == i)]
		feat_id = id_features(i,idvitals,idlabs,age_train)
		feat = numpy.concatenate((feat,feat_id), axis=0)

	return feat


print "\n\n TRAINING STARTS"


a = genfromtxt('id_time_labs_train.csv', delimiter=',')       #lab values training data
outlierlabs(a)
print "outlierlabs done"
b = genfromtxt('id_time_vitals_train.csv', delimiter=',')       #vitals values training data
outliervitals(b)
print "outliervitals done"
vitals = b
labs = a

for j in [4,8,9,12,13,16,17,18,20,23,24]:
	labs[1:,j] = numpy.log(labs[1:,j])

labs_min =  numpy.nanmax(labs, axis=0)
labs_max =  numpy.nanmin(labs, axis = 0)
vitals_min =  numpy.nanmax(vitals, axis=0)
vitals_max =  numpy.nanmin(vitals, axis = 0)
global_labs_min = labs_min
global_labs_max = labs_max
global_vitals_min = vitals_min
global_vitals_max = vitals_max

for i in range(1,vitals.shape[0]):
   for j in range(2,8):
      if not numpy.isnan(vitals[i][j]):
         vitals[i][j] = (float(vitals[i][j])-vitals_min[j])/(vitals_max[j]-vitals_min[j])

for i in range(1,labs.shape[0]):
   for j in range(2,27):
      if not numpy.isnan(labs[i][j]):
         labs[i][j] = (float(labs[i][j])-labs_min[j])/(labs_max[j]-labs_min[j])

print "min max norm of labs vitals done"

vitals_minmax = vitals[1:,:]
labs_minmax = labs[1:,:]

numpy.savetxt("minmax_without_nan_vitals_new.csv", vitals_minmax, delimiter=",")
numpy.savetxt("minmax_without_nan_labs_new.csv", labs_minmax, delimiter=",")


vitals = vitals_minmax
labs = labs_minmax
age_train = genfromtxt('id_age_train.csv', delimiter=',')
age_train = age_train[1:,:]

print "\n\n Feature Extraction Started \n\n"

feat = features(vitals, labs, age_train)

numpy.savetxt("features_model_with_nans.csv", feat, delimiter=",")

avg_feat =  numpy.nanmean(feat, axis = 0)
global_avg_feat = avg_feat
for i in range(feat.shape[0]):
	for j in range(feat.shape[1]):
		if numpy.isnan(feat[i][j]):
			feat[i][j] = avg_feat[j]

numpy.savetxt("features_model_without_nans.csv", feat, delimiter=",")

print "\n\n Feature Extraction done \n\n"

print global_labs_min, global_labs_max, global_vitals_min, global_vitals_max, global_avg_feat
with open('global.pickle', 'w') as f:
    pickle.dump([global_labs_min, global_labs_max, global_vitals_min, global_vitals_max, global_avg_feat], f)



features = feat
#features = genfromtxt('features_model_without_nans.csv', delimiter=',')
features = features[:,1:]
labels = genfromtxt('id_label_train.csv', delimiter=',')
labels = labels[1:,1:]

print features.shape
print labels.shape


X = features
y = labels
y = numpy.reshape(y,(y.shape[0],))
print y.shape
print X.shape

print "\n\nFitting Started\n\n"

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X, y)
y_pred = clf.predict(X)
print metrics.accuracy_score(y, y_pred)
prob = clf.predict_proba(X)
#numpy.savetxt("prob_trainingphase.csv", prob, delimiter=",")
print "\n\n Fitting Done\n\n"
print "The Classifier is::\n\n"

print clf
with open('classifier.pickle', 'w') as f:
    pickle.dump(clf, f)
