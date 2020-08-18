import glob
import os
import nltk
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from collections import defaultdict
from scipy import spatial


def fentchfiles():
    os.chdir(r'C:\Users\UGTECH\Desktop\case\summaries')
    myfiles = glob.glob('*.txt')
    
    return myfiles


################ read files #######################################
def preprocessing():
    files = fentchfiles()
    
    stop_words = set(stopwords.words('english')) 
    data = []
    
    stemmer = PorterStemmer()
    for file in files:
        with open(file , 'r') as f:
            lines = f.read()
            tokenizer = RegexpTokenizer(r'\w+')
            word_tokens = tokenizer.tokenize(lines)
            filtertxt = []
            for word in word_tokens:
                if word not in stop_words:
                    tokens = stemmer.stem(word)
                    filtertxt.append( tokens)               
            data.append(filtertxt)
            

    return data 
############################################


# def preprocessing():
#     files = fentchfiles()
    
#     data = []
#     stop_words = set(stopwords.words('english')) 
#     stemmer = PorterStemmer()
    
#     for file in files:
#         with open(file , 'r') as f:
#             lines = f.read()
#             sentences = sent_tokenize(lines)
            
#             for sent in sentences:
#                 data.append([sent])
                
#     docs = []
#     for line in data:
#         print(line)
#         tokenizer = RegexpTokenizer(r'\w+')
#         word_tokens = tokenizer.tokenize(line)
#         filtertxt = []
#         for word in word_tokens:
#             if word not in stop_words:
#                 tokens = stemmer.stem(word)
#                 filtertxt.append(tokens)
        
#         docs.append(filtertxt)
#     return docs



# data = preprocessing()
# print(data)


################## inverted index #############################
# def create_index():
#     data = preprocessing()
#     index = defaultdict(list)
    
#     for i , tokens in enumerate(data):
#         for token in tokens:
#             if token in index:
                
#                 index[token][0] = index[token][0] + 1
#                 if i not in index[token][1]:
#                     index[token][1].append(i)
                    
            
            
            
#             else:
#                 index[token].append(1)
#                 index[token].append([])
# #                 index[token][1].append({})
#                 index[token][1].append(i)
            
#     return index

####################################

########################## POSITIONAL INDEX #################
# def create_index():
#     data = preprocessing()
#     index = defaultdict(list)
    
#     posting = {}
#     for i , tokens in enumerate(data):
#         pos = 1;
#         for token in tokens:
#             if token in index:
                
#                 index[token][0] = index[token][0] + 1
                
#                 if i in index[token][1]:
#                     index[token][1][i].append(pos)
                   
#                 else:
#                     index[token][1][i] = [pos]
            
            
            
#             else:
#                 index[token].append(1)
#                 index[token].append(posting)
#                 index[token][1][i] = [pos]
                
#             pos += 1;
            
#     return index

###########################################

def create_index():
    data = preprocessing()
    index = defaultdict(list)
    count = 0;
    for i , tokens in enumerate(data):
        count+=1;
        for token in tokens:
            if token in index:
                
                if i not in index[token][1]:
                    index[token][1][i] = 1
                    index[token][0] = index[token][0] + 1
                else:
                    index[token][1][i] += 1
                    
                    
            
            
            
            else:
                index[token].append(1)
                index[token].append({})
                index[token][1][i] = 1
            
    return (index,count)


def tfidf():
    index , count =  create_index();
    values = defaultdict(dict)
    
    
   
    for term in index.keys():
        for key , value in index[term][1].items():
            values[term][key] = (np.log(count/index[term][0]) * value/count )
    
    
    return (values,count)


def query(doc):
    occur = {}
    stemmer = PorterStemmer()

    
    num = 0;
    for word in doc.split(' '):
        word = stemmer.stem(word)
        num+=1
        if word in occur.keys():
            occur[word]+= 1
            
        else:
            occur[word] = 1
        
        
    quer = defaultdict(dict)
    
    index , count = create_index()
    
    for term in index.keys():
    
        if term in occur.keys():
            for key , value in index[term][1].items():
                quer[term][key] = (np.log(count/index[term][0]) * occur[term]/num)
        
        
        else:
            i = 0
            while ( i < count):
                quer[term][i] =0
                i+=1
             
        
    
    return quer
    
    
    
# res = query(" v abraham out out beyond")   


# print(res)
# index = tfidf()
# print(index)    
    
# f = fentchfiles();
# print(f)



def matrix(que):
    index , count = tfidf()
    quer = query(que)
    
#     for token in index.keys(): 
    df = pd.DataFrame(index , columns = index.keys())
    df.fillna(0 , inplace = True)
    df = df.T
    
    df1 = pd.DataFrame(quer)
    df1.fillna(0 , inplace = True)
    df1 = df1.T
    
    return (df ,df1)


def cosine(v1 , v2):
    return (np.dot(v1 , v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))



def similarity(que):
    
    df , df1 = matrix(que)
    values = {}
    files = fentchfiles()
    display = []
    for i in df.columns:
        if((df1.loc[: , i] == 0).all() == False):
            values[i] = ( 1 - spatial.distance.cosine(df.loc[: , i] , df1.loc[: , i]))
        
        
    values = [(key) for (key, value) in sorted(values.items(), key=lambda x: x[1] , reverse = True)]
    
    
    
    
    
    for value in values:
        with open(files[value] , 'r') as f:
            print(f.read())

    
    return 
    
    
    
    

que = "even can investigation "

index = create_index()

print(index)
    