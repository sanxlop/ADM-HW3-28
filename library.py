#========================== Import ==========================
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import numpy as np
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import json
import collections
import itertools
#nltk.download('punkt')
#nltk.download('stopwords')
import library #library of functions

# To print results
BOLD = '\033[1m'
END = '\033[0m'

#========================== Step 3: Search Engine ==========================

def cleanString(data):
    """
    Function: cleanString(data)
    - Input: data --> (string)
    - Description: This function takes all documents and removes punctuation, stop words and do stemming
    """
    # Removes punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(data)
    # Removes stopwords
    stopWords = set(stopwords.words('english'))
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    # Stemming of words
    porter = PorterStemmer()
    result = [porter.stem(word) for word in wordsFiltered]
    # Return cleaned string
    return result

def modifyDocs(n):
    """
    Function: modifyDocs()
    - Input: n --> (int) number of files to clean
    - Description: This function modifies all documents using cleanString to clean "description" and "title"
    """
    # Read documents to clean "description" and "title" data
    for i in range(0, n):
        # Read data frame of the document i
        doc = pd.read_csv('documents/doc_'+str(i)+'.tsv', sep='\t', encoding='utf-8')
        # Edit data frame with cleaned values
        doc.at[0, 'title']  = ' '.join(cleanString(doc.iloc[0]["title"]))
        doc.at[0, 'description']= ' '.join(cleanString(doc.iloc[0]["description"]))
        # Re-write results in the same document
        doc.to_csv('documentsCleaned/doc_'+str(i)+'.tsv', sep='\t', encoding='utf-8', index = False)


#------- 3.1.1 Create your Index -------

def inverted_index_add(inverted_index, doc_id, doc, index_column):
    """
    Function: inverted_index_add(inverted_index, doc_id, doc, index_column)
    - Input: inverted_index --> (dic) Dictionary to save results
    - Input: doc_id --> (string) Name of the doc
    - Input: doc --> (pandas.df) Document to explore
    - Input: index_column --> (string) Column to explore
    - Description: This function takes all documents and removes punctuation, stop words and do stemming
    """
    for word in doc.iloc[0][index_column].split():
        if word not in inverted_index:
            inverted_index.setdefault(word, [doc_id])
        else:
            inverted_index[word].append(doc_id)
    return inverted_index


#------- 3.1.2 Execute the query -------

def searchQueryConjunctive(inverted_index, query):
    """
    Function: searchQueryConjunctive(inverted_index, query)
    - Input: inverted_index --> (collection) Collection with the inverted index
    - Input: query --> (string) Query to search
    - Description: This function search all documents where have matches with the querys
    """
    print(BOLD + "Query intruduced: " + END + query)
    # Clean query
    query = library.cleanString(query)
    # Look for the words that are in the inverted index
    words = [word for word in query if word in inverted_index]
    print(BOLD + "Cleaned query: " + END + " ".join(words))
    # Save the documents that have coincidences
    docAppearances = [set(inverted_index[word]) for word in words]
    # Count the number of appearances of words of each doc
    return dict(collections.Counter(itertools.chain.from_iterable(docAppearances)))