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
import heapq
import folium
from geopy import distance
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
    Description: This function takes all documents and removes punctuation, stop words and do stemming
    - Return: result (string) cleaned string of data
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
    Description: This function modifies all documents using cleanString to clean "description" and "title"
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
    Description: This function makes an inverted index
    - Return: inverted_index (dictionary)
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
    Description: This function search all documents where have matches with the querys
    - Return: Dictionary of documents and each number of coincidences with the query (dictionary)
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

def listOfConjunctiveMatches(searched_results, number_query_words):
    """
    Function: listOfConjunctiveMatches(searched_results)
    - Input: searched_results --> (collection) Collection with matches of the query with each doc
    Description:Loop through all the serached_results and see if the number of coincidences is equal to the number of words of the query
    - Return: (list) List of matches tsv files string
    """
    list_matches = []
    for result in searched_results.items():
        if result[1] == number_query_words:
            list_matches.append(pd.read_csv('documents/'+result[0]+'.tsv', sep='\t', encoding='utf-8', usecols=['title', 'description', 'city', 'url']))
    return list_matches

#------- 3.2 Conjunctive Query and Ranking Score-------
#------- 3.2.1 Inverted index scored -------

def readTitleAndDesc(doc_id):
    """
    Function: readTitleAndDesc(doc_id)
    - Input: doc_id --> (string) Name of the doc_id
    Description: This function read the doc_id selected and return title and description
    - Return: textDoc (string)
    """
    doc = pd.read_csv('documentsCleaned/'+doc_id+'.tsv', sep='\t', encoding='utf-8')
    entry1 = doc.iloc[0]['description']
    entry2 = doc.iloc[0]['title']
    if type(entry1) is str and type(entry2) is str:
        textDoc = entry1 + " " + entry2
    else:
        textDoc = ""
    return textDoc

def invertedIndexScoredAdd(inverted_index_scored, doc_id, inverted_index, n_total_docs):
    """
    Function: invertedIndexScoredAdd(inverted_index_scored, doc_id, doc, inverted_index)
    - Input: inverted_index_scored --> (dic) Dictionary to save results
    - Input: doc_id --> (string) Name of the doc
    - Input: doc --> (pandas.df) Document to explore
    - Input: inverted_index --> (dic) Dictionary of the normal inverted index
    - Input: n_total_docs --> (int) number of docs
    Description: This function makes an inverted index with tf-idf score (without taking into account the query)
    - Return: inverted_index_scored (dictionary)
    """

    textDoc = readTitleAndDesc(doc_id).split()
    nTextWords = len(textDoc) # Number of words in the doc
    setTextDoc = set(textDoc) # In order to not repeat documents.
    for word in setTextDoc:
        ## Compute TF
        wordOccurInDoc = textDoc.count(word) # Number of appearances in the text
        tf = wordOccurInDoc #/ nTextWords # Term frequency #
        ## Compute IDF
        N = n_total_docs #total number of documents in the corpus
        nDocsInTerm = len(inverted_index[word]) # Number of documents where the term "word" appears
        idf = math.log( N / nDocsInTerm ) # Inverse document frequency
        ## Compute TF-IDF
        tfIdf = tf*idf
        # Add values to the dictionary
        if word not in inverted_index_scored:
            inverted_index_scored.setdefault(word, [(doc_id, tfIdf)])
        else:
            inverted_index_scored[word].append((doc_id, tfIdf))
                
    return inverted_index_scored

#------- 3.2.2 Execute the query -------

def cosineSimilarity(query, doc_id, inverted_index_scored): 
    """
    Function: cosineSimilarity(query, doc_id)
    - Input: query --> (string) Query
    - Input: doc_id --> (string) Name of the doc_id
    Description: Compute summatory(TFIDF) / (|q| * |d|)
    - Return: cosSim (float)
    """
    
    cosSim = (0, doc_id) # Cosine similarity result
    
    ## Computing |d|
    d_ = 0 # Variable |d|
    textDoc = readTitleAndDesc(doc_id).split() # String of words that contains the document
    setTextDoc = set(textDoc) # In order to not repeat documents.
    for word in setTextDoc:
        nWordTimesInDoc = textDoc.count(word) #Number of times that appears the word in the doc
        d_ += nWordTimesInDoc**2
    d_ = d_**(1/2)
    
    ## Computing |q|
    query = library.cleanString(query).split() # String query cleaned
    q_ = 0 #Variable |q|
    setQuery = set(query) # In order to not repeat documents.
    for word in setQuery:
        nWordTimesInQuery = query.count(word) #Number of times that appears the word in the doc
        q_ += nWordTimesInQuery**2
    q_ = q_**(1/2)
    #q_ = len(query)**(1/2) # Square root of query length
    
    ## Computing summatory(tfidf)
    sum_tfidf = float(0) # Variable to store the sum of the tfidf scores
    for word in query:
        if word in inverted_index_scored:
            for item in inverted_index_scored[word]:
                if doc_id in item:
                    #print(item)
                    sum_tfidf += item[1] # Take TFIDF value
                    
    ## Computing cos similarity
    if q_ != 0 and  d_ != 0:
        cosSim = (sum_tfidf / ( (q_) * (d_) ), doc_id)
        
    return cosSim


def getListOfConjunctiveDocIds(inverted_index, query):
    """
    Function: getListOfConjunctiveDocIds(inverted_index, query)
    - Input: inverted_index --> (dic) Dictionary with the inverted index
    - Input: query --> (string) Query
    Description: Compute the string doc_id of the conjunctive matches
    - Return: conjunctiveDocId (list) list of string of the doc_id
    """
    
    # Compute the search with the query and obtain a dictionary with the number of matches
    searchedResultsNMatches = searchQueryConjunctive(inverted_index, query)
    numberOfQueryWords = len(cleanString(query).split())
    print(BOLD + "Number of query words:" + END, numberOfQueryWords)

    # Look for doc_id conjunctive match with the query
    conjunctiveDocId = []
    for resultNMatches in searchedResultsNMatches.items(): # sortedResults contains tuples doc_id and number of querymatches
        if(resultNMatches[1] == numberOfQueryWords):
            conjunctiveDocId.append(resultNMatches[0])

    # Print the number of conjunctive matches
    if(conjunctiveDocId == []):
        print(BOLD + "No conjunctive matches" + END)
    else:
        print(BOLD + "Number of conjunctive matches:" + END,len(conjunctiveDocId))
        
    return conjunctiveDocId

def computeCosineSim(conjunctive_docid, n_data, query, inverted_index_scored):
    """
    Function: computeCosineSim(conjunctive_docid, n_data, query, inverted_index_scored)
    - Input: conjunctive_docid --> (list) List with doc_id in the conjunctive search
    - Input: n_data --> (int) Number of row data of files to compute
    - Input: query --> (string) Query
    - Input: inverted_index_scored --> (Dic) Dictionary with the tfidf score
    Description: Compute cosineSimilarity for conjunctive results or all the rest
    - Return: cos_sim_results (list) list of tuples cos_sim_result and doc_id
    """
    cos_sim_results = []
    if(len(conjunctive_docid) != 0):
        #CONJUNCTIVE SEARCH
        for doc_id in conjunctive_docid:
            cos_sim_results.append(cosineSimilarity(query, doc_id, inverted_index_scored))
    else:
        #NOT CONJUNCTIVE SEARCH (takes more time)
        for i in range(0, n_data):
            cos_sim_results.append(cosineSimilarity(query, "doc_"+str(i), inverted_index_scored))
    return cos_sim_results

def makeAndDisplayCosineSimilarityDataFrame(sorted_cos_sim, conjunctive_docid):
    """
    Function: computeAndDisplayCosineSimilarityDataFrame(sorted_cos_sim, conjunctive_docid)
    - Input: conjunctive_docid --> (List) List with doc_id in the conjunctive search
    - Input: sorted_cos_sim --> (List) List of tuples containing "doc_id" and "cos_sim"
    Description: Compute dataframe with the data to visualize and display it
    - Display: dataframe of resuts
    """
    # List of pandas df
    dfs_cos = []
    # Loop through all the sorted results and add similarity and conjunctive_match info to a df
    for i in range(0, len(sorted_cos_sim)):
        if sorted_cos_sim[i][0] > 0:
            doc_test = pd.read_csv('documents/'+sorted_cos_sim[i][1]+'.tsv', sep='\t', encoding='utf-8', usecols=['title', 'description', 'city', 'url'])
            doc_test["similarity"] = sorted_cos_sim[i][0]
            dfs_cos.append(doc_test)

    # Print if they are conjunctive results or not
    if(len(conjunctive_docid) != 0):
        print(BOLD + "CONJUNCTIVE RESULTS" + END)
    elif(len(conjunctive_docid) != 0):
        print(BOLD + "NOT CONJUNCTIVE RESULTS" + END)

    # Concat all dataframes into one to show the results        
    if(len(dfs_cos) != 0):
        # Concatenate all data into one DataFrame
        big_frame = pd.concat(dfs_cos, ignore_index=True)
        # Reorder columns 
        df = big_frame.loc[:, ['title', 'description', 'city', 'url', 'similarity']]
        # Display dataframe result of the query
        display(df)
    else:
        print("NO RESULTS")


#========================== Step 4: Define a new score! ==========================
#------- 4.1  -------

def returnAndShowDatasetConjunctiveResults(conjunctive_doc_id):
    """
    Function: showDatasetConjunctiveResults(conjunctive_doc_id)
    - Input: conjunctive_docid --> (List) List with doc_id in the conjunctive search
    Description: Show the conjunctive results dataframe
    - Display: dataframe of resuts
    - Return: pandas dataframe with results
    """
    dfs_new = []
    columns = ['average_rate_per_night', 'bedrooms_count', 'city', 'description', 'title', 'url']
    # Loop through all the sorted results and add similarity and conjunctive_match info to a df
    for doc_id in conjunctive_doc_id:
        if len(conjunctive_doc_id) != 0:
            doc_test = pd.read_csv('documents/'+doc_id+'.tsv', sep='\t', encoding='utf-8', usecols=columns)
            doc_test["doc_id"] = doc_id
            doc_test["average_rate_per_night"] = int(library.cleanString(doc_test.iloc[0]["average_rate_per_night"]))
            if doc_test.iloc[0]["bedrooms_count"] == "Studio":
                doc_test["bedrooms_count"] = 1
            dfs_new.append(doc_test)

    # Print if they are conjunctive results or not
    if(len(conjunctive_doc_id) != 0):
        print(BOLD + "CONJUNCTIVE RESULTS" + END)
    elif(len(conjunctive_doc_id) != 0):
        print(BOLD + "NOT CONJUNCTIVE RESULTS" + END)

    # Concat all dataframes into one to show the results        
    if(len(dfs_new) != 0):
        # Concatenate all data into one DataFrame
        df = pd.concat(dfs_new, ignore_index=True)
        # Display dataframe result of the query
        display(df)
    else:
        print("NO RESULTS")

    return df

def dicNormalized(conjunctive_docid, df, column_name):
    """
    Function: dicNormalized(conjunctive_docid, df, column_name)
    - Input: conjunctive_docid --> (List) List of conjunctives docid
    - Input: df --> (pandas DataFrame) Dataframe to compute
    - Input: column_name --> (String) Name of the column to compute
    Description: Compute list of ints normalized by the max value
    - Return: normaliez (Dict) Dictionary of tuples with normalized values and docid
    """
    normalized = {}
    for docid in conjunctive_docid:
        value = list(df.loc[df["doc_id"]==docid, column_name])[0]
        maxValue = df[column_name].max()
        normalized.update({docid: -value/maxValue})
    return normalized

def dicMatchCityQuery(conjunctive_docid, query, df):
    """
    Function: dicMatchCityQuery(conjunctive_docid, query, df)
    - Input: conjunctive_docid --> (List) List of conjunctives docid
    - Input: df --> (pandas DataFrame) Dataframe to compute
    - Input: query --> (String) Query string
    Description: Compute list of ints normalized by the max value if the city appears in the query
    - Return: normaliez (Dict) Dictionary of tuples with normalized values and docid
    """
    found_words = {}
    query = library.cleanString(query).split()
    for docid in conjunctive_docid:
        coincidence = 0
        city = library.cleanString(list(df.loc[df["doc_id"]==docid, "city"])[0]).split()
        for string in city:
            if string in query:
                coincidence = 1
        found_words.update({docid: coincidence})
    return found_words

def listOfComputeScores(conjunctive_docid, *args):
    """
    Function: listOfComputeScores(conjunctive_docid, *args)
    - Input: conjunctive_docid --> (List) List of conjunctives docid
    - Input: args --> (Lists) List of tuples with scores and docid
    Description: Compute the summatory of all the scores in args
    - Return: newScores (List) List of tuples with scores and docid
    """
    newScores = []
    for docid in conjunctive_docid:
        sumScore = 0
        for arg in args:
            sumScore += arg[docid]
        newScores.append((sumScore, docid))
    return newScores


def returnAndShowDatasetResultsOwnScore(sorted_scored):
    """
    Function: returnAndShowDatasetResultsOwnScore(sorted_scored)
    - Input: sorted_scored --> (List) List of tuples with the score and the doc_id
    Description: Show the results ordered by our own score
    - Display: dataframe of resuts
    - Return: pandas dataframe with results
    """
    dfs_new = []
    columns = ['title', 'description', 'city', 'url']
    # Loop through all the sorted results and add similarity and conjunctive_match info to a df
    for doc_id in sorted_scored:
        doc_test = pd.read_csv('documents/'+doc_id[1]+'.tsv', sep='\t', encoding='utf-8', usecols=columns)
        #doc_test["new_score"] = doc_id[0]
        dfs_new.append(doc_test)

    # Concat all dataframes into one to show the results        
    if(len(dfs_new) != 0):
        # Concatenate all data into one DataFrame
        df = pd.concat(dfs_new, ignore_index=True)
        # Display dataframe result of the query
        display(df)
    else:
        print("NO RESULTS")
        
    return df


#========================== Bonus Step: Show a map! ==========================

def serchDocIdHousesInRadio(n_files, start_point, radius):
    """
    Function: serchDocIdHousesInRadio(n_files, start_point)
    - Input: n_files --> (Int) Number of files in the data set
    - Input: start_point --> (Float) Tuple of floats containing latitude and longitude
    - Input: radius --> (Int) Number of meters of the circle radius
    Description: Make a list with the doc_id that are inside the calculated distance
    - Return: (List) List of strings with doc_id
    """
    # Search houses in radio
    docid_houses_in_radio = []
    for i in range(0,n_files):
        # Read one tsv file
        readFile = pd.read_csv('documents/doc_'+str(i)+'.tsv', sep='\t', encoding='utf-8') 
        if(not np.isnan(readFile.latitude[0]) and not np.isnan(readFile.longitude[0])):
            coordinates_house = (readFile.latitude[0], readFile.longitude[0]) # Take coordinates of the house
            computed_distance = distance.distance(start_point, coordinates_house).km # Compute the distance from the start point
            if (computed_distance < radius/1000): # If is in the radio save result
                docid_houses_in_radio.append('doc_'+str(i))
    return docid_houses_in_radio

def addHouseToMap(doc_ids, map_folium):
    """
    Function: addHouseToMap(doc_ids, map_folium)
    - Input: doc_ids --> (List) List of doc_id in the radio
    - Input: map_folium --> (Folium object) Map folium
    Description: Add houses in doc_ids to the map
    """
    #Adding houses
    for docid in doc_ids:
        # Read one tsv file
        readFile = pd.read_csv('documents/'+docid+'.tsv', sep='\t', encoding='utf-8')
        tooltip_val = '<b>'+str(readFile.title[0])+'</b>'
        popup_val = '<b>Price per night: '+str(readFile.average_rate_per_night[0])+'</b>'
        folium.Marker([readFile.latitude[0], readFile.longitude[0]], popup=popup_val, tooltip=tooltip_val, icon=folium.Icon(color='blue')).add_to(map_folium)
