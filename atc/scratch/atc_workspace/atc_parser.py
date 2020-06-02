'''
Script for parsing ATC.csv file from its found format to one that is useful
ATC.csv is found at the following website:
https://bioportal.bioontology.org/ontologies/ATC
Downloaded version: 2019AB released 11/04/2019 uploaded 11/18/2019
This file was used instead of the WHO's query database which would need to
individually query each drug compound. This is a compiled database of the same.
The WHO codes in atc_to_class.tsv are from Wikipedia
https://en.wikipedia.org/wiki/ATC_code_A for A classification, etc.
Can we find the additional classifications, since this only shows level 1 and level 2?
'''


#import numpy as np
import pandas as pd

def parse_atc_to_class(file_name):
	lines = [item.strip("\n") for item in open(file_name).readlines()]
	tab = {}
	for l in lines:
		s = l.split(",")
		tab[s[0]] = s[1]
	return tab

atc_file = pd.read_csv('ATC.csv')
atc_to_class = parse_atc_to_class('atc_to_class.tsv')

#print(atc_file[0:10])
#print(atc_to_class)
#print(atc_file["Class ID"])
#atc_file[['asdf','ATC']] = atc_file["Class ID"].str.split(" ",expand=True,)
#print(atc_file["ATC"])

headers = ["Drug","Class_ID","Classification"]
out_file = pd.DataFrame(columns=headers)
for i in range(len(atc_file)):
	class_id = atc_file["Class ID"][i].split("/")[-1]
	drug = atc_file["Preferred Label"][i]
	#print(class_id+"\t"+drug)
	classif = "NA"
	#print(class_id[0:3])
	if(class_id[0:3] in atc_to_class):
		classif = atc_to_class[class_id[0:3]]
	#print(drug+'\t'+classif)
	dic = {headers[0]:drug,
			headers[1]:class_id,
			headers[2]:classif}
	out_file = out_file.append(dic, ignore_index=True)
out_file.to_csv("ATC_Mapping.csv",index=False)
	

