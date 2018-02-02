# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 08:29:59 2018

@author: Jaskaran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Import dataset
dataset = pd.read_csv('E:\courses\Udemy-courses\Machine Learning A-Z Template Folder\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
# cleaning data
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words Model
# tokenization
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() # creates sparse matrix
y = dataset.iloc[:, 1].values

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# fitting the classifier to training set
# Create classifier here
classifier = GaussianNB()
#classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

# preditct test set results
y_pred = classifier.predict(X_test)

# making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

# just normal accuracy
acs = accuracy_score(y_test, y_pred)
print (acs)