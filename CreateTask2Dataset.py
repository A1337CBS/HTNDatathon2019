#This code prepares the data according to the input template for our Model
import glob
import re
import csv

#Open/Create a .csv file with a delimiter of one tab space
wtr = csv.writer(open('task2Dataset.csv', 'wb'),delimiter='\t', quoting=csv.QUOTE_NONE, escapechar=' ')

#Get list of all file names 
listOfFileNames = glob.glob("train/*.task2.labels")
listOfFileNames = sorted(listOfFileNames)
print(len(listOfFileNames))
allSentences = []
for file1 in listOfFileNames:
    with open(file1) as f:
        lines = f.readlines()
        file1 = re.sub('[^0-9]', '', file1)
        #file1 = file1[1:]
        file1 = file1[:-1]
        #print(len(lines))
        for line in lines:
            lineParts = line.split('\t')
            articleFileName = "train/article"+file1+".txt"
            with open(articleFileName) as f1:
                sentences = f1.readlines()
                print(lineParts[0]+"\t"+lineParts[1]+"\t"+lineParts[2].rstrip()+"\t"+sentences[int(lineParts[1])-1])
                cleanDataRow = [lineParts[0],lineParts[1],lineParts[2].rstrip(),sentences[int(lineParts[1])-1]]
                wtr.writerow(cleanDataRow)
                allSentences.append(cleanDataRow)