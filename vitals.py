import csv 
with open('Traning_Dataset/id_time_vitals_train.csv', 'rb') as csvfile:
	spamreader = csv.DictReader(csvfile, delimiter=',')
	v1count = 0
	v2count = 0
	v3count = 0
	v4count = 0
	v5count = 0
	v6count = 0
	icu1 = 0
	icu0 = 0

	for row in spamreader:
		if row['V1'] != "NA":
			v1count = v1count+1

		if row['V2'] != "NA":
			v2count = v2count+1

		if row['V3'] != "NA":
			v3count = v3count+1

		if row['V4'] != "NA":
			v4count = v4count+1

		if row['V5'] != "NA":
			v5count = v5count+1

		if row['V6'] != "NA":
			v6count = v6count+1

		if int(row['ICU']) == 0:
			icu0 = icu0+1

		if int(row['ICU']) == 1:
			icu1 = icu1+1

	print v1count, v2count, v3count, v4count, v5count, v6count, icu0, icu1
	print v1count/3594, v2count/3594, v3count/3594, v4count/3594, v5count/3594, v6count/3594, icu0/2594, icu1/3594
 