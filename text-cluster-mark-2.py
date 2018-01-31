# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:31:17 2017

@author: Jaskaran
"""

import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


def word_tokenizer(text):
        #tokenizes and stems the text
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=0.9,
                                        min_df=0.1,
                                        lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)


if __name__ == "__main__":
        # sentences = ["Nature is beautiful","I like green apples",
        #         "We should protect the trees","Fruit trees provide fruits",
        #         "Green apples are tasty"]
        
        sentences = ["Merch by Amazon",
"Lecture Search",
"Discover dev",
"Machine Learning Translation and the Google Translate Algorithm",
"Deep Learning — from Prototype to Production – Becoming Human",
"Gilt ♥ Open Source / Visually Similar Recommendations",
"Tutorial – Becoming Human",
"CS224n: Natural Language Processing with Deep Learning",
"Courses | Stanford Artificial Intelligence Laboratory",
"The Man from Earth - Wikipedia",
"Building a Real-time Chat App with Angular 2 and deepstream | Codementor",
"Building An Image Crawler Using Python And Scrapy | Codementor",
"Python Tutorials and Insights | Codementor Community",
"tutorials overview - deepstreamhub",
"FaceApp adds paid selfie styling effects | TechCrunch",
"big dreams by happy raikoti - Google Search",
"faceapp - Google Search",
"hackethereum",
"Fitting to Noise or Nothing At All: Machine Learning in Markets | Zachary David&#39;s",
"GitHub - minimaxir/textgenrnn: Python module to easily generate text using a pretrained character-based recurrent neural network."]

        nclusters= 5
        clusters = cluster_sentences(sentences, nclusters)
        for cluster in range(nclusters):
                print ("cluster ",cluster,":")
                for i,sentence in enumerate(clusters[cluster]):
                        print ("\tsentence ",i,": ",sentences[sentence])