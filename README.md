--> [VISUALIZE JUPYTER NOTEBOOK IN NBVIEWER](https://nbviewer.jupyter.org/github/sanxlop/ADM-HW3-28/blob/master/Homework_3.ipynb)
# Homework 3 - Find the perfect place to stay in Texas!
In this assignment we perform an analysis of houses, rooms or apartments in Texas. Specifically, we start from a dataset with different information about rental housing
## Data to analyze
We have used the data of (https://www.kaggle.com/PromptCloudHQ/airbnb-property-data-from-texas)
## Additional data
We used also additional data provided in the [Homework 3 repository](https://github.com/CriMenghini/ADM-2018/tree/master/Homework_3).
## List of files
1. `Homework_3.ipynb`
- This ipython notebook contains the followed steps to analyze and make the searh engine.
2. `library.py` 
- This python file contains all the libraries used to perform the search engine.
## Description of the project
First of all, we have managed the data provided in a csv file to multiple tsv files. Secondly, we have cleaned the data in order to avoid special characters and repeated words.
And finally, we have used inverted index, TF-IDF and cosine similarity to perform the different search engines provided. 
____
## STEPS followed
### STEP 1. Download the data
Just download the csv dataset.
### STEP 2. Create documents
Create a tsv file per each row of the dataset.
### STEP 3. Search Engine
Clean the data of the "title" and "description" with tokenizer (stopwords, special characters, stemming words)
#### STEP 3.1 Search Engine
Make an inverted index and use it to find documents with a conjunctive query
#### STEP 3.2 Search Engine TF-IDF and Cosine Similarity
Make and inverted index with the TF-IDF score and use it to find documents with a conjunctive query ordered by cosine similarity
### STEP 4. Define a new score!
We have defined our new score normalizing values of average price, city and number of bedrooms in order to sort the previous results.
### BONUS STEP: Make a nice visualization
In this section we are taking coordinates and a radio to show in the map the houses inside the area (map_folium)
