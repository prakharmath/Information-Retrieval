from bs4 import BeautifulSoup
import os
import re
import json
import sys
import time
import string
import nltk
start=time.time()
current_directory_path = os.path.dirname(os.path.realpath(__file__))
print("Loading the inverted indexed dictionary from the json file")
with open('inverted_indexed_dictionary.json') as f:
  data = json.load(f)
print("Successfully loaded the inverted indexed dictionary from the json file")
list_of_keys=sorted(data.keys())
reverse_list_of_keys=[i[::-1] for i in list_of_keys] 
reverse_list_of_keys=sorted(reverse_list_of_keys)
number_of_tokens=len(list_of_keys)
def prefix_queries(prefix_query): #Binary search for the queries of the type a*
	low=0
	high=number_of_tokens-1
	lower_bound_index=-1
	while(low<=high):
	 	mid=low+(high-low)//2
	 	if(list_of_keys[mid]>=prefix_query):
	 		lower_bound_index=mid
	 		high=mid-1
	 	else:
	 		low=mid+1
	result=[] 
	N=len(prefix_query)
	if(lower_bound_index!=-1):
		while(lower_bound_index<number_of_tokens):
			if(len(list_of_keys[lower_bound_index])<N):
				break
			if(str(list_of_keys[lower_bound_index][0:N])!=prefix_query):
				break
			result.append(list_of_keys[lower_bound_index])
			lower_bound_index+=1
	return result
def suffix_queries(suffix_query): #Binary search for the queries of the type *a(reversing the string and then apply prefix search)
	low=0
	high=number_of_tokens-1
	lower_bound_index=-1
	N=len(suffix_query)
	while(low<=high):
	 	mid=low+(high-low)//2
	 	if(reverse_list_of_keys[mid]>=suffix_query):
	 		lower_bound_index=mid
	 		high=mid-1
	 	else:
	 		low=mid+1
	result=[] 
	if(lower_bound_index!=-1):
		while(lower_bound_index<number_of_tokens):
			if(len(reverse_list_of_keys[lower_bound_index])<N):
				break
			if(str(reverse_list_of_keys[lower_bound_index][0:N])!=suffix_query):
				break
			result.append(reverse_list_of_keys[lower_bound_index][::-1])
			lower_bound_index+=1
	return result
def Print_Result(Result):
	for result in Result:
		print('"'+result+'"'+":",end='')
		flag=0
		for x in data[result]:
			for y in data[result][x]:
				if(flag==1):
					print(',',end='')
				print("<"+str(x)+","+str(y)+">",end='')
				flag=1
		print(";",end='')
	print("")
Query_File_Name=str(sys.argv[1])
Input_File_Object = open(Query_File_Name,"r")
Queries=Input_File_Object.readlines()
Output_File_Object=open("RESULTS1_17CS10036.txt","w")
original_stdout=sys.stdout
sys.stdout=Output_File_Object
for query in Queries:
	query=query.strip()
	n=len(query)
	Result=[]
	if(query[n-1]=='*'):  #Prefix Query
		query=str(query[0:n-1])
		prefix_query=query
		Result=prefix_queries(prefix_query)
	elif(query[0]=='*'): #Suffix query
		suffix_query=str(query[1:n])
		suffix_query=suffix_query[::-1]
		Result=suffix_queries(suffix_query) 
	else:  #If its a composite query splitting the string and then apply prefix and suffix queries and then taking intersection
		pos=-1
		for i in range(0,n):
			if(query[i]=='*'):
				pos=i
				break 
		prefix_query=query[0:pos]
		suffix_query=query[pos+1:n]
		suffix_query=suffix_query[::-1]
		result_prefix=prefix_queries(prefix_query)
		result_suffix=suffix_queries(suffix_query)
		Final_Result=list(set(result_prefix)&set(result_suffix))
		Result=[]
		for x in Final_Result:
			if(len(x)>=len(prefix_query)+len(suffix_query)):
				Result.append(x)
	if(len(Result)==0):
		sys.stdout=original_stdout
		print("No Matching Result found for the query "+query)
		sys.stdout=Output_File_Object
	Print_Result(Result)
sys.stdout=original_stdout
end=time.time()
print("Time Elapsed:"+str(end-start))
