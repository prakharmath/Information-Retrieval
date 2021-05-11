from bs4 import BeautifulSoup
import os
import re
import json
import sys
import time
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
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
start=time.time()
current_directory_path = os.path.dirname(os.path.realpath(__file__))
corpus_path=current_directory_path+"/"+"ECTText"
list_of_files = os.listdir(corpus_path)
stop_words = set(stopwords.words('english'))
idx=1
tokens=set()
inverted_indexed_dictionary={}
for filename in list_of_files:
	print("Removing Stop words,Punctuations,Lemmatizing and building inverted_indexed_dictionary for file "+str(idx))
	corpus = open(corpus_path+"/"+filename,errors='ignore')
	table = str.maketrans('', '', string.punctuation)
	word_lemmatizer = WordNetLemmatizer()
	pos=-1
	while True:
	    line = corpus.readline()
	    # word_tokens = word_tokenize(line)
	    # word_token = [w.translate(table) for w in word_tokens]
	    # processed = [word_lemmatizer.lemmatize(w) for w in word_token if not w in stop_words] 
	    # for r in processed:
		   #  if(len(r)==0):
		   #  	continue
		   #  pos+=1
		   #  tokens.add(r)
		   #  if r not in inverted_indexed_dictionary.keys():
		   #  	inverted_indexed_dictionary[r]={}
		   #  if idx not in inverted_indexed_dictionary[r].keys():
		   #  	inverted_indexed_dictionary[r][idx]=[]
		   #  inverted_indexed_dictionary[r][idx].append(pos)
	    word_tokens = word_tokenize(line)  #Properly lemmatizing the corpus by using pos tag. This will take more time than normal lematization but will do proper lematization. If you want results early kindly use the commented out code for lemmatization
	    nltk_tag=nltk.pos_tag(word_tokens)
	    for x in nltk_tag:
	    	if x[0] in stop_words or not x[0][0].isalpha():
	    		continue
	    	tag=nltk_tag_to_wordnet_tag(x[1])
	    	r=word_lemmatizer.lemmatize(x[0],tag)
	    	if(len(r)==0):
	    		continue
	    	if(type(r)!=str):
	    		print(type(r))
	    	tokens.add(r)
	    	pos+=1
	    	if r not in inverted_indexed_dictionary.keys():
	    		inverted_indexed_dictionary[r]={}
	    	if idx not in inverted_indexed_dictionary[r].keys():
	    		inverted_indexed_dictionary[r][idx]=[]
	    	inverted_indexed_dictionary[r][idx].append(pos)
	    if not line:
	        break
	idx+=1
print("Dumping the build inverted_indexed_dictionary as json for reading in the next task")
with open("inverted_indexed_dictionary.json", "w") as outfile:  #Dumping the inverted indexed dictionary as json
		json.dump(inverted_indexed_dictionary, outfile,indent=2,sort_keys=True)
#print("Number of tokens:"+str(len(tokens))) 
end=time.time()
print("Time Elapsed:"+str(end-start))