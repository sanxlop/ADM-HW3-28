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
    wordsFiltered = [porter.stem(word) for word in words]
    # Return cleaned string
    return wordsFiltered

def modifyDocs():
    """
    Function: modifyDocs()
    - Input: 
    - Description: This function modifies all documents using cleanString to clean "description" and "title"
    """
    # Read documents to clean "description" and "title" data
    for i in range(0, nRowsOrFiles):
        # Read data frame of the document i
        doc = pd.read_csv('documents/doc_'+str(i)+'.tsv', sep='\t', encoding='utf-8')
        # Edit data frame with cleaned values
        doc.at[0, 'title']  = ' '.join(cleanString(doc.iloc[0]["title"]))
        doc.at[0, 'description']= ' '.join(cleanString(doc.iloc[0]["description"]))
        # Re-write results in the same document
        doc.to_csv('documents/doc_'+str(i)+'.tsv', sep='\t', encoding='utf-8', index = False)


#------- 3.1.1 Create your Index #-------
