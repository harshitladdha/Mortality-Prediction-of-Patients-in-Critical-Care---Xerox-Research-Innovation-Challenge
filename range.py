import csv 
with open('Traning_Dataset/id_time_labs_train.csv', 'rb') as csvfile:
	spamreader = csv.DictReader(csvfile, delimiter=',')
	Matrix = [[0 for x in range(26)] for x in range(3)]

	for j in range(26):
		Matrix[0][j] = 99999999999999999


	for row in spamreader:
		indexj = 0
		for j in ['TIME','L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11','L12','L13','L14','L15','L16','L17','L18','L19','L20','L21','L22','L23','L24','L25']:
			
			if row[j]!="NA":
				if float(row[j]) < Matrix[0][indexj] :
					Matrix[0][indexj] = float(row[j])

			if row[j]!="NA":
				if float(row[j]) > Matrix[1][indexj] :
					Matrix[1][indexj] = float(row[j])

			if row[j]!="NA":
				Matrix[2][indexj] = Matrix[2][indexj] + float(row[j])
 
			indexj = indexj+1


	for i in range(3):
		for j in range(26):
			print str(Matrix[i][j]) + ",",
		print "\n"
