from flask import Flask, render_template, Response, request, redirect, url_for
import pandas as pd
import gensim as gs
import networkx as nx
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stops = stopwords.words('english')

def GetSummary(file):
    dataset = pd.read_csv("file.csv",encoding='unicode_escape')
    sentences = []
    for s in dataset['article_text']:
        sentences.append(sent_tokenize(s))
    sentences = [y for x in sentences for y in x] # flatten list
    dataset=dataset['article_text']
    doc_processed=[]
    for i in dataset:
         doc_processed.append(gs.utils.simple_preprocess(i))
    #intilize the model
    w2v = Word2Vec(doc_processed, vector_size=100, window=5, min_count=1, sg=0)
    #build vocabulary
    w2v.build_vocab(doc_processed,progress_per=1000)
    #train word2vec model
    w2v.train(doc_processed, total_examples=w2v.corpus_count, epochs=w2v.epochs)
    punctuations= '''!()-[]{};:'"\,<>./?@#$%^&*_~''' 
    my_str=sentences[:]
    no_punct=""
    for char in my_str:
        if(char not in punctuations):
            no_punct =no_punct+char
    lowercase=""
    for char in no_punct:
        lowercase =lowercase+char.lower()   
    Tokenizedwords=nltk.word_tokenize(no_punct)
    # Sample data (list of strings/documents)
    documents=stops
    # Step 1: Convert text to numerical vectors using CountVectorizer
    vectorizer = CountVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Step 2: Calculate cosine similarity
    cosine_similarities = cosine_similarity(vectors)

    # Output the cosine similarity matrix
    #print("Cosine Similarity Matrix:")
    #print(cosine_similarities)

    nx_graph = nx.from_numpy_array(cosine_similarities)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences[:])), reverse=True)
    finallen=len(ranked_sentences)

    result=""
    for i in range(finallen):
        result=result+ranked_sentences[i][1]
        result=result+"\n"

    return result


def remove_stopwords(sample):
    sen_new = " ".join([i for i in sample if i not in stops])
    return sen_new













