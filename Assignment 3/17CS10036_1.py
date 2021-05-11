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
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import PorterStemmer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
dataset_directory = sys.argv[1]
out_file = sys.argv[2]
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
stop_words = set(stopwords.words('english'))
current_directory_path = os.path.dirname(os.path.realpath(__file__))  #Current directory
data_path=dataset_directory
class1_train_tf={}
class2_train_tf={}
class1_test_tf={}
class2_test_tf={}
train_docs_class1=[]
train_docs_class2=[]
test_docs_class1=[]
test_docs_class2=[]
idf_t_dictionary={}
num_docs=0
list_of_files = os.listdir(os.path.join(data_path,"class1","train"))
word_lemmatizer = WordNetLemmatizer()
count=1
print("Lemmatizing and Tokenizing Training data of class 1")
tokens=set()
for file_name in list_of_files:
	print(str(count)+'.'+"Lemmatizing and Tokenizing Doc "+file_name)
	count+=1
	try:
		file = open(data_path+"/class1/train/"+file_name,encoding='unicode_escape')
		text_data=file.read().lower()
	except:
		file = open(data_path+"/class1/train/"+file_name,encoding='utf-8')
		text_data=file.read().lower()
	try:
		word_tokens=word_tokenize(text_data)
	except:
		word_tokens=word_tokenize(text_data.decode('utf-8'))
	tagged=nltk.pos_tag(word_tokens)
	token_is_present={}
	train_docs_class1.append(file_name)
	for word, tag in tagged:
		if(word in stop_words or not word[0].isalpha()):
			continue
		wntag = nltk_tag_to_wordnet_tag(tag)
		lemma = word_lemmatizer.lemmatize(word, pos=wntag)
		if(lemma not in class1_train_tf.keys()):
			class1_train_tf[lemma]={}
		if(file_name not in class1_train_tf[lemma].keys()):
			class1_train_tf[lemma][file_name]=1
		else:
			class1_train_tf[lemma][file_name]+=1
		if(lemma not in token_is_present):
			token_is_present[lemma]=1
	for x in token_is_present:
		tokens.add(x)
		if x not in idf_t_dictionary.keys():
			idf_t_dictionary[x]=1
		else:
			idf_t_dictionary[x]+=1
	num_docs+=1
count=1
list_of_files = os.listdir(os.path.join(data_path,"class1","test"))
print("Lemmatizing and Tokenizing Testing data of class 1")
for file_name in list_of_files:
	print(str(count)+'.'+"Lemmatizing and Tokenizing Doc "+file_name)
	count+=1
	try:
		file = open(data_path+"/class1/test/"+file_name,encoding='unicode_escape')
		text_data=file.read().lower()
	except:
		file = open(data_path+"/class1/test/"+file_name,encoding='utf-8')
		text_data=file.read().lower()
	try:
		word_tokens=word_tokenize(text_data)
	except:
		word_tokens=word_tokenize(text_data.decode('utf-8'))
	tagged=nltk.pos_tag(word_tokens)
	token_is_present={}
	test_docs_class1.append(file_name)
	for word, tag in tagged:
		if(word in stop_words or not word[0].isalpha()):
			continue
		wntag = nltk_tag_to_wordnet_tag(tag)
		lemma = word_lemmatizer.lemmatize(word, pos=wntag)
		if(lemma not in class1_test_tf.keys()):
			class1_test_tf[lemma]={}
		if(file_name not in class1_test_tf[lemma].keys()):
			class1_test_tf[lemma][file_name]=1
		else:
			class1_test_tf[lemma][file_name]+=1
		if(lemma not in token_is_present):
			token_is_present[lemma]=1
count=1
list_of_files = os.listdir(os.path.join(data_path,"class2","train"))
print("Lemmatizing and Tokenizing Training data of class 2")
for file_name in list_of_files:
	print(str(count)+'.'+"Lemmatizing and Tokenizing Doc "+file_name)
	count+=1
	try:
		file = open(data_path+"/class2/train/"+file_name,encoding='unicode_escape')
		text_data=file.read().lower()
	except:
		file = open(data_path+"/class2/train/"+file_name,encoding='utf-8')
		text_data=file.read().lower()
	try:
		word_tokens=word_tokenize(text_data)
	except:
		word_tokens=word_tokenize(text_data.decode('utf-8'))
	tagged=nltk.pos_tag(word_tokens)
	token_is_present={}
	train_docs_class2.append(file_name)
	for word, tag in tagged:
		if(word in stop_words or not word[0].isalpha()):
			continue
		wntag = nltk_tag_to_wordnet_tag(tag)
		lemma = word_lemmatizer.lemmatize(word, pos=wntag)
		if(lemma not in class2_train_tf.keys()):
			class2_train_tf[lemma]={}
		if(file_name not in class2_train_tf[lemma].keys()):
			class2_train_tf[lemma][file_name]=1
		else:
			class2_train_tf[lemma][file_name]+=1
		if(lemma not in token_is_present):
			token_is_present[lemma]=1
	for x in token_is_present:
		tokens.add(x)
		if x not in idf_t_dictionary.keys():
			idf_t_dictionary[x]=1
		else:
			idf_t_dictionary[x]+=1
	num_docs+=1
count=1
list_of_files = os.listdir(os.path.join(data_path,"class2","test"))
print("Lemmatizing and Tokenizing Testing data of class 2")
for file_name in list_of_files:
	print(str(count)+'.'+"Lemmatizing and Tokenizing Doc "+file_name)
	count+=1
	try:
		file = open(data_path+"/class2/test/"+file_name,encoding='unicode_escape')
		text_data=file.read().lower()
	except:
		file = open(data_path+"/class2/test/"+file_name,encoding='utf-8')
		text_data=file.read().lower()
	try:
		word_tokens=word_tokenize(text_data)
	except:
		word_tokens=word_tokenize(text_data.decode('utf-8'))
	tagged=nltk.pos_tag(word_tokens)
	token_is_present={}
	test_docs_class2.append(file_name)
	for word, tag in tagged:
		if(word in stop_words or not word[0].isalpha()):
			continue
		wntag = nltk_tag_to_wordnet_tag(tag)
		lemma = word_lemmatizer.lemmatize(word, pos=wntag)
		if(lemma not in class2_test_tf.keys()):
			class2_test_tf[lemma]={}
		if(file_name not in class2_test_tf[lemma].keys()):
			class2_test_tf[lemma][file_name]=1
		else:
			class2_test_tf[lemma][file_name]+=1
		if(lemma not in token_is_present):
			token_is_present[lemma]=1
print("Building Doc vectors")
doc_vector=np.zeros((2000,len(tokens)))
document_mapping={}
doc_idx=0
for x in train_docs_class1:
	document_mapping[x]=doc_idx
	doc_idx+=1
for x in train_docs_class2:
	document_mapping[x]=doc_idx 
	doc_idx+=1
for x in test_docs_class1:
	document_mapping[x]=doc_idx
	doc_idx+=1
for x in test_docs_class2:
	document_mapping[x]=doc_idx 
	doc_idx+=1
terms_mapping={}
idx=0
for x in tokens:
	terms_mapping[x]=idx
	idx+=1
idx=0
for word in tokens:
	try:
		tf_dictionary=class1_train_tf[word]
		for doc in tf_dictionary.keys():
			doc_vector[document_mapping[doc]][idx]=math.log(1+tf_dictionary[doc],10)*math.log(num_docs/idf_t_dictionary[word],10)
	except:
		pass
	idx+=1
idx=0
for word in tokens:
	try:
		tf_dictionary=class2_train_tf[word]
		for doc in tf_dictionary.keys():
			doc_vector[document_mapping[doc]][idx]=math.log(1+tf_dictionary[doc],10)*math.log(num_docs/idf_t_dictionary[word],10)
	except:
		pass
	idx+=1
idx=0
for word in tokens:
	try:
		tf_dictionary=class1_test_tf[word]
		for doc in tf_dictionary.keys():
			doc_vector[document_mapping[doc]][idx]=math.log(1+tf_dictionary[doc],10)*math.log(num_docs/idf_t_dictionary[word],10)
	except:
		pass
	idx+=1
idx=0
for word in tokens:
	try:
		tf_dictionary=class2_test_tf[word]
		for doc in tf_dictionary.keys():
			doc_vector[document_mapping[doc]][idx]=math.log(1+tf_dictionary[doc],10)*math.log(num_docs/idf_t_dictionary[word],10)
	except:
		pass
	idx+=1
for doc_number in range(doc_idx):
	norm=np.linalg.norm(doc_vector[doc_number])
	if(norm!=0):
		doc_vector[doc_number]/=norm
train_docs=doc_vector[0:num_docs]
test_docs=doc_vector[num_docs:doc_idx]
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for x in train_docs_class1:
	X_train.append(doc_vector[document_mapping[x]])
	y_train.append(1)
for x in train_docs_class2:
	X_train.append(doc_vector[document_mapping[x]])
	y_train.append(0)
for x in test_docs_class1:
	X_test.append(doc_vector[document_mapping[x]])
	y_test.append(1)
for x in test_docs_class2:
	X_test.append(doc_vector[document_mapping[x]])
	y_test.append(0)
f1_multinomial={}
f1_bernoulli={}
for n in [1,10,100,1000,10000]:
	print("Running for num_features="+str(n))
	print("Selecting features according to mutual information")
	mutual_info_selector = SelectKBest(mutual_info_classif, k = n)
	X_train_kbest = mutual_info_selector.fit_transform(X_train, y_train)
	X_test_kbest = mutual_info_selector.transform(X_test)
	print("Training Multinomial Naive Bayes Classifier")
	classif_multi_nb = MultinomialNB()
	classif_multi_nb.fit(X_train_kbest, y_train)
	print("Fitting Test Data to Multinomial Naive Bayes Classifier")
	y_pred_multi_nb = classif_multi_nb.predict(X_test_kbest)
	f1_multinomial[n]= f1_score(y_test, y_pred_multi_nb,average='macro')
	print("Training Bernoulli Naive Bayes Classifier")
	classif_bernoulli_nb = BernoulliNB()
	classif_bernoulli_nb.fit(X_train_kbest, y_train)
	print("Fitting Test Data to Bernoulli Naive Bayes Classifier")
	y_pred_bernoulli_nb = classif_bernoulli_nb.predict(X_test_kbest)
	f1_bernoulli[n] = f1_score(y_test, y_pred_bernoulli_nb,average='macro')
with open(out_file,'w+',encoding='utf-8') as outfile:
	outfile.write("NumFeature\t")
	for n in [1,10,100,1000,10000]:
		outfile.write(str(n)+'\t')
	outfile.write("\nMultinomialNB\t")
	for x in f1_multinomial:
		outfile.write(str(f1_multinomial[x])+'\t')
	outfile.write("\nBernoulliNB\t")
	for x in f1_bernoulli:
		outfile.write(str(f1_bernoulli[x])+'\t')
end=time.time()
print("Time Elapsed:"+str(end-start))

