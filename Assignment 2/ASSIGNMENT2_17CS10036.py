#Prakhar Bindal(17CS10036)
from bs4 import BeautifulSoup
import os
import re
import json
import pickle
import math
import sys
import time
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
start=time.time()  #Start time of the program
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
def nltk_tag_to_wordnet_tag(nltk_tag):  #NLTK Tag functions for proper lemmatization by identifying the type of the word
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return wordnet.NOUN
stop_words = set(stopwords.words('english')) #Set of stop words of english language
idf_t_dictionary={}
tf_t_d_dictionary={}  #Dictionaries for storing idf and tf scores
data_path='../Dataset/Dataset'
list_of_files = os.listdir(data_path)
with open('../Dataset/StaticQualityScore.pkl', 'rb') as f: #Loading Static Quality Scores
    Static_Quality_Scores = pickle.load(f)
with open('../Dataset/Leaders.pkl','rb') as f:  #Loading Leaders
	Leaders=pickle.load(f)
number_of_docs=0
count=1
tokens=set()
word_lemmatizer = WordNetLemmatizer()
for file_name in list_of_files: #Extracting data and building idf and tf dictionaries
	number_of_docs+=1
	doc_number=""
	for x in file_name:
		if(x=='.'):
			break
		doc_number+=x
	doc_number=int(doc_number)
	print(str(count)+'.'+"Lemmatizing and Tokenizing Doc "+str(doc_number))
	count+=1
	soup = BeautifulSoup(open(data_path+"/"+file_name), "html.parser")
	all_text=soup.get_text().lower()
	word_tokens=word_tokenize(all_text)
	tagged=nltk.pos_tag(word_tokens)
	token_is_present={}
	for word, tag in tagged:
		if(word in stop_words or not word[0].isalpha()):
			continue
		wntag = nltk_tag_to_wordnet_tag(tag)
		lemma = word_lemmatizer.lemmatize(word, pos=wntag)
		if(lemma not in tf_t_d_dictionary.keys()):
			tf_t_d_dictionary[lemma]={}
		if(doc_number in tf_t_d_dictionary[lemma].keys()):
			tf_t_d_dictionary[lemma][doc_number]+=1
		else:
			tf_t_d_dictionary[lemma][doc_number]=1
		token_is_present[lemma]=1
	for x in token_is_present:
		tokens.add(x)
		if(x not in idf_t_dictionary.keys()):
			idf_t_dictionary[x]=1
		else:
			idf_t_dictionary[x]+=1
print("Number of Tokens:"+str(len(tokens)))
list_of_terms=idf_t_dictionary.keys()
term_index_dictionary={}
idx=0
print("Building Inverted Positional Index Dictionary")
for x in list_of_terms:
	term_index_dictionary[x]=idx 
	idx+=1
inverted_positional_index_dictionary={}
for t in tf_t_d_dictionary.keys():
	t_dictionary=tf_t_d_dictionary[t]
	idf_t=math.log(number_of_docs*1.0/idf_t_dictionary[t],10)
	inverted_positional_index_dictionary[(t,idf_t)]=[]
	for doc_number in t_dictionary.keys():
		inverted_positional_index_dictionary[(t,idf_t)].append((doc_number,math.log(1+t_dictionary[doc_number],10)))
print("Building Local and Global Champions List")
champions_list_local={}
champions_list_global={}
for t in tf_t_d_dictionary.keys():
	t_dictionary=tf_t_d_dictionary[t]
	champions_list_local[t]=[]
	champions_list_global[t]=[]
	for doc_number in t_dictionary:
		frequency=t_dictionary[doc_number]
		tf_d_score=math.log10(1+frequency)
		idf_t=math.log10(number_of_docs*1.0/idf_t_dictionary[t])
		champions_list_local[t].append((tf_d_score,doc_number))
		champions_list_global[t].append((tf_d_score*idf_t+Static_Quality_Scores[doc_number],doc_number))
	champions_list_local[t].sort(reverse=True)
	champions_list_global[t].sort(reverse=True)
	while(len(champions_list_local[t])>50):
		champions_list_local[t].pop(-1)
	while(len(champions_list_global[t])>50):
		champions_list_global[t].pop(-1)
	temporary_champions_list_local=[]
	temporary_champions_list_global=[]
	for x in champions_list_local[t]:
		temporary_champions_list_local.append(x[1])
	for x in champions_list_global[t]:
		temporary_champions_list_global.append(x[1])
	champions_list_local[t]=temporary_champions_list_local
	champions_list_global[t]=temporary_champions_list_global
doc_vector=np.zeros((number_of_docs,len(list_of_terms)))
idx=0
print("Building Document Vector")
for word in list_of_terms:
	tf_dictionary=tf_t_d_dictionary[word]
	for doc in tf_dictionary.keys():
		doc_vector[doc][idx]=math.log(1+tf_dictionary[doc],10)*math.log(number_of_docs/idf_t_dictionary[word],10)
	idx+=1
for doc_number in range(number_of_docs):
	doc_vector[doc_number]/=np.linalg.norm(doc_vector[doc_number])
print("Building Followers Dictionary for Cluster Pruning")
Followers={}
for doc_number in range(number_of_docs):
	max_score=-1
	leader_doc=-1
	current_doc_vector=doc_vector[doc_number]
	for leader_doc_number in Leaders:
		if(doc_number==leader_doc_number):
			continue
		score=np.dot(current_doc_vector,doc_vector[leader_doc_number])
		if(score>max_score):
			max_score=score 
			leader_doc=leader_doc_number
	if(leader_doc not in Followers.keys()):
		Followers[leader_doc]=[]
		Followers[leader_doc].append(doc_number)
	else:
		Followers[leader_doc].append(doc_number)
for leader_doc_number in Leaders:
	if(leader_doc_number not in Followers.keys()):
		Followers[leader_doc_number]=[]
		Followers[leader_doc_number].append(leader_doc_number)
	else:
		Followers[leader_doc_number].append(leader_doc_number)
query_file_name=str(sys.argv[1])
Output_File_Object=open("RESULTS2_17CS10036.txt","w")
original_stdout=sys.stdout
with open(query_file_name,"r+", encoding="utf-8") as query_file:
	for line in query_file.readlines():
		query=""
		word_tokens = word_tokenize(line)
		tagged = nltk.pos_tag(word_tokens)
		for word, tag in tagged:
			if (word not in stop_words) and (word[0].isalpha()) :
				wntag = nltk_tag_to_wordnet_tag(tag)
				lemma = word_lemmatizer.lemmatize(word, pos=wntag)
				query+= lemma+" "
		print("Calculating Score for the query "+query)
		sys.stdout=Output_File_Object
		print(query)
		query_vector=np.zeros(len(list_of_terms))
		for word in query.split():
			if word not in term_index_dictionary.keys():
				continue
			query_vector[term_index_dictionary[word]]=math.log(number_of_docs*1.0/idf_t_dictionary[word],10)
		norm=np.linalg.norm(query_vector)
		if(norm!=0):
			query_vector/=norm
		tf_idf_scores=[]
		for doc in range(number_of_docs):
			tf_idf_scores.append((np.dot(doc_vector[doc],query_vector),doc))
		tf_idf_scores.sort(reverse=True)
		for i in range(min(len(tf_idf_scores),10)):
			if(i!=0):
				print(',',end='')
			if(tf_idf_scores[i][0]==0.0):
				break
			print('<'+str(tf_idf_scores[i][1])+','+str(tf_idf_scores[i][0])+'>',end='')
		print('\n')
		champions_list_local_docs=set()
		for word in query.split():
			if word not in champions_list_local.keys():
				continue
			for y in champions_list_local[word]:
				champions_list_local_docs.add(y)
		champions_list_local_score=[]
		for doc in champions_list_local_docs:
			champions_list_local_score.append((np.dot(doc_vector[doc],query_vector),doc))
		champions_list_local_score.sort(reverse=True)
		for i in range(min(len(champions_list_local_score),10)):
			if(i!=0):
				print(',',end='')
			if(champions_list_local_score[i][0]==0.0):
				break
			print('<'+str(champions_list_local_score[i][1])+','+str(champions_list_local_score[i][0])+'>',end='')
		print('\n')
		champions_list_global_docs=set()
		for word in query.split():
			if word not in champions_list_global.keys():
				continue
			for y in champions_list_global[word]:
				champions_list_global_docs.add(y)
		champions_list_global_score=[]
		for doc in champions_list_global_docs:
			champions_list_global_score.append((np.dot(doc_vector[doc],query_vector),doc))
		champions_list_global_score.sort(reverse=True)
		for i in range(min(len(champions_list_global_score),10)):
			if(i!=0):
				print(',',end='')
			if(champions_list_global_score[i][0]==0.0):
				break
			print('<'+str(champions_list_global_score[i][1])+','+str(champions_list_global_score[i][0])+'>',end='')
		print('\n')
		leader_query=-1
		max_score=-1
		for leader_doc_number in Leaders:
			current_doc_vector=doc_vector[leader_doc_number]
			score=np.dot(current_doc_vector,query_vector)
			if(score>max_score):
				max_score=score 
				leader_query=leader_doc_number
		cluster_pruning_scores=[]
		for doc in Followers[leader_query]:
			cluster_pruning_scores.append((np.dot(doc_vector[doc],query_vector),doc))
		cluster_pruning_scores.sort(reverse=True)
		for i in range(min(10,len(cluster_pruning_scores))):
			if(i!=0):
				print(',',end='')
			if(cluster_pruning_scores[i][0]==0.0):
				break
			print('<'+str(cluster_pruning_scores[i][1])+','+str(cluster_pruning_scores[i][0])+'>',end='')
		print('\n\n')
		sys.stdout=original_stdout
end=time.time()
print("Time Elapsed:"+str(end-start))

