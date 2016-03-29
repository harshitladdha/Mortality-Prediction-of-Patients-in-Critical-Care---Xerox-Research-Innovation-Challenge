import csv 
with open('Traning_Dataset/id_label_train.csv', 'rb') as csvfile:
	spamreader = csv.DictReader(csvfile, delimiter=',')
	total=0
	count=0
	for row in spamreader:
		total = total+1
		if int(row['LABEL'])==1:
			count = count+1

	print count, total
	print float(float(count)/total)*100


	#6.81691708403   No of died patients
