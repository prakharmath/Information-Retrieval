from bs4 import BeautifulSoup
import os
import re
import json
import sys
import time
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
start=time.time()  #Start time of the program
current_directory_path = os.path.dirname(os.path.realpath(__file__))  #Current directory
data_path=current_directory_path+"/data"
type_of_participants={}
type_of_participants["company participants"]=1   #Type of Participants
type_of_participants["conference call participants"]=1
type_of_participants["corporate participants"]=1
type_of_participants["company representatives"]=1
type_of_participants["companyparticipants"]=1 
type_of_participants["conference callparticipants"]=1
type_of_participants["corporateparticipants"]=1
type_of_participants["companyrepresentatives"]=1
type_of_participants["executives"]=1
type_of_participants["analysts"]=1
list_of_months={}
list_of_months["January"]=1
list_of_months["February"]=1  #Type of Months
list_of_months["March"]=1
list_of_months["April"]=1
list_of_months["May"]=1
list_of_months["June"]=1
list_of_months["July"]=1
list_of_months["August"]=1
list_of_months["September"]=1
list_of_months["October"]=1
list_of_months["November"]=1
list_of_months["December"]=1
idx=1
list_of_files = os.listdir(data_path)
os.mkdir("ECTNestedDict")
os.mkdir("ECTText")
for filename in list_of_files:
	print("Parsing File "+str(idx))
	soup = BeautifulSoup(open(data_path+"/"+filename), "html.parser") #Parsing all files through soup
	all_tags=soup.find_all('p')  #Finding all tags
	all_span_tags=soup.find_all('span') #Finding all span tags
	bold_tags=soup.find_all('strong')  #Finding all bold tags
	nested_dictionary={}  #Output nested dictionary
	for i in range(len(all_tags)):  #Loop for parsing date
		x=all_tags[i]
		text=x.text 
		text=' '.join(text.split())
		tokens=text.strip()
		date_found_flag=False 
		for month in list_of_months:
			if month in tokens:
				match = re.search(month+r' \d{1,2}, \d{4}', text)  #Searching using regular expression 
				if match:
					nested_dictionary["Date"]=match.group(0)
					date_found_flag=True
					break
		if(date_found_flag):
			break
	nested_dictionary["Participants"]=[]  
	last_position=0
	for i in range(len(all_tags)): #Loop for parsing participants and their details
		x=all_tags[i]
		text=x.text 
		text=' '.join(text.split())
		tokens=text.strip()
		if(tokens.lower() in type_of_participants.keys()):
			j=i+1
			while(j<len(all_tags)):
				text=all_tags[j].text 
				text = text.replace('–', '-')
				if '-' in text:
					name = text.split('-')[0].strip()
					nested_dictionary["Participants"].append(name)
					last_position=j
				else:
					break
				j+=1
	while(all_tags[last_position+1].text.lower() in type_of_participants.keys()):
		last_position+=1
	while(len(all_tags[last_position+1].text)>50): #Searching for Presentations
		last_position-=1
	presentations_dictionary={}
	question_answer_pos=0
	for i in range(0,len(all_tags)): #Parsing Presentations
		text=all_tags[i]
		if(len(text)==0):
			continue
		lower_text=text.text.lower().strip()
		if(lower_text=="question-and-answer session" or lower_text=="questions-and-answer session" or lower_text=="question-and-answers session" or lower_text=="question-and answer session"):
			question_answer_pos=i
			break
		try:
			element = text.contents[0].contents
			name = text.text
			name = re.sub(u"(\u2018|\u2019)", "'", name)
			if name not in presentations_dictionary.keys():
			    presentations_dictionary[name] = []
		except AttributeError:
			dialogue = text.contents[0]
			dialogue = re.sub(u"(\u2018|\u2019)", "'", dialogue)
			if name != '' and name in presentations_dictionary.keys() and name.lower() not in type_of_participants.keys():
				presentations_dictionary[name].append(dialogue)
			pass
	nested_dictionary["Presentations"]=presentations_dictionary
	question_answer_text = soup.find_all('span')[1 : ]
	question_answer_dictionary = {}
	i=0
	list_of_dialogues=[]
	new_dictionary={}
	while(i<len(question_answer_text)):  #Parsing Question Answers
		list_of_dialogues.clear()
		new_dictionary.clear()
		x = question_answer_text[i].find_next('p')
		while x is not None and i + 1 < len(question_answer_text) and x.get_text() != question_answer_text[i + 1].get_text():
		    list_of_dialogues.append(x.get_text())
		    x = x.find_next('p')
		new_dictionary[question_answer_text[i].get_text()] = list_of_dialogues
		if i == len(question_answer_text) - 1:
			if(x is not None):
				list_of_dialogues.append(x.get_text())
				new_dictionary[question_answer_text[i].get_text()] = list_of_dialogues
		question_answer_dictionary[i + 1] = new_dictionary
		i+=1
	nested_dictionary["Question Answers"]=question_answer_dictionary
	with open("ECTNestedDict/"+str(idx)+"_dictionary.json", "w") as outfile:
		json.dump(nested_dictionary, outfile,indent=2)  #Dumping the nested dictionary as JSON
	original_stdout = sys.stdout
	with open("ECTText/"+str(idx)+"_textfile.txt","w") as outfile:
		sys.stdout=outfile
		print(soup.get_text().lower())  #Dumping the text to make the text corpus(We can comment out this line and uncomment the following ones if we want to build the text corpus from the dictionary itself , if we want to build directly from html pages run the code as it is)
		# print("date")
		# print(nested_dictionary["Date"].lower())
		# print("participants")
		# for y in nested_dictionary["Participants"]:
		# 	print(y.lower())                                  #Code for dumping the text from the dictionary(Uncomment this part and comment out line 133 if you want to build the text corpus from the dictionary otherwise leave it as it is )
		# print("presentations")
		# for x in nested_dictionary["Presentations"]:
		# 	print(x.lower())
		# 	for y in nested_dictionary["Presentations"][x]:
		# 		print(y.lower())
		# print("question answers")
		# for x in nested_dictionary["Question Answers"]:
		# 	for y in nested_dictionary["Question Answers"][x]:
		# 		print(y.lower())
		# 		for z in nested_dictionary["Question Answers"][x][y]:
		# 			print(z.lower())
	sys.stdout = original_stdout
	idx+=1
end=time.time()
print("Time Elapsed:"+str(end-start))  #Printing the end time







