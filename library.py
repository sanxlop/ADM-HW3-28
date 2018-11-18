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
import math
#nltk.download('punkt')
#nltk.download('stopwords')
import library #library of functions

# To reaload library import
from importlib import reload
reload(library)

# To print results
BOLD = '\033[1m'
END = '\033[0m'

#========================== Step 3: Search Engine ==========================
#------- 3.1 Conjunctive Query -------

def cleanString(data):
    """
    Function: cleanString(data)
    - Input: data --> (string)
    - Description: This function takes all documents and removes punctuation, stop words and do stemming
    """
    if type(data) is str: # If the data is type str do this
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
        listOfWords = [porter.stem(word) for word in wordsFiltered]
        result = ' '.join(listOfWords)
    else:
        result = np.nan #if the data introduced is not str return nan
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
        doc.at[0, 'title']  = cleanString(doc.iloc[0]["title"])
        doc.at[0, 'description'] = cleanString(doc.iloc[0]["description"])
        # Re-write results in the same document
        doc.to_csv('documentsCleaned/doc_'+str(i)+'.tsv', sep='\t', encoding='utf-8', index = False)


#------- 3.1.1 Create your Index -------

def invertedIndexAdd(inverted_index, doc_id, doc, index_column):
    """
    Function: invertedIndexAdd(inverted_index, doc_id, doc, index_column)
    - Input: inverted_index --> (dic) Dictionary to save results
    - Input: doc_id --> (string) Name of the doc
    - Input: doc --> (pandas.df) Document to explore
    - Input: index_column --> (string) Column to explore
    - Description: This function makes an inverted index
    """
    entry = doc.iloc[0][index_column]
    if type(entry) is str: #the data must be a string
        for word in entry.split():
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
    query = library.cleanString(query).split()
    # Look for the words that are in the inverted index
    words = [word for word in query if word in inverted_index]
    print(BOLD + "Cleaned query: " + END + " ".join(words))
    # Save the documents that have coincidences
    docAppearances = [set(inverted_index[word]) for word in words]
    # Count the number of appearances of words of each doc
    return dict(collections.Counter(itertools.chain.from_iterable(docAppearances)))

#------- 3.2 Conjunctive Query and Ranking Score-------
#------- 3.2.1 Inverted index -------

def invertedIndexScoredAdd(inverted_index_scored, doc_id, doc, inverted_index, n_total_docs):
    """
    Function: invertedIndexScoredAdd(inverted_index_scored, doc_id, doc, inverted_index)
    - Input: inverted_index_scored --> (dic) Dictionary to save results
    - Input: doc_id --> (string) Name of the doc
    - Input: doc --> (pandas.df) Document to explore
    - Input: inverted_index --> (dic) Dictionary of the normal inverted index
    - Input: n_total_docs --> (int) number of docs
    - Description: This function makes an inverted index with tf-idf score
    """
    entry1 = doc.iloc[0]['description']
    entry2 = doc.iloc[0]['title']

    if type(entry1) is str and type(entry2) is str: #the data must be string
        textDoc = entry1 + " " + entry2
        nTextWords = len(textDoc)
        setTextDoc = set(textDoc.split()) # In order to not repeat documents.
        for word in setTextDoc:
            # Compute TF
            wordOccur = textDoc.split().count(word) # Number of appearances in the text
            tf = wordOccur/nTextWords # Term frequency
            # Compute IDF
            idf = math.log( n_total_docs / len(inverted_index[word]) ) # Inverse document frequency
            # Compute TF-IDF
            tfIdf = tf*idf
            # Add values to the dictionary
            if word not in inverted_index_scored:
                inverted_index_scored.setdefault(word, [(doc_id, tfIdf)])
            else:
                inverted_index_scored[word].append((doc_id, tfIdf))
                
    return inverted_index_scored

#------- 3.2.2 Execute the query -------
