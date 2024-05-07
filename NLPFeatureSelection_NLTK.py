#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:58:13 2021

@author: nitinsinghal
"""

# Text mining using NLTK, two types of feature selection methods 

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel,SelectKBest, chi2
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')

# Load the data
customer_data = pd.read_csv('./Customers.csv')
comment_data = pd.read_csv('./Comments.csv')

y = customer_data["TARGET"]
X = customer_data.drop(columns=["TARGET"])

print(X.shape)
print(comment_data.shape)
print(comment_data.head())
print(y)

#Tokenize - Split the sentences to lists of words
comment_data['CommentsTokenized'] = comment_data['Comments'].apply(word_tokenize)

# Porter Stemmer
stemmer = PorterStemmer()
comment_stemdata=pd.DataFrame()
comment_stemdata=comment_data[['ID']]
comment_stemdata['CommentsTokenizedStemmed'] = comment_data['CommentsTokenized'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
comment_stemdata.to_csv('./comment_Porterstemdata.csv')

# Use Snowball English stemmer.
stemmer = SnowballStemmer("english")
comment_stemdata=pd.DataFrame()
comment_stemdata=comment_data[['ID']]
comment_stemdata['CommentsTokenizedStemmed'] = comment_data['CommentsTokenized'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
comment_stemdata.to_csv('./comment_Snowballstemdata.csv')

#Join stemmed strings
comment_stemdata['CommentsTokenizedStemmed'] = comment_stemdata['CommentsTokenizedStemmed'].apply(lambda x: " ".join(x))

# Use CountVectorizer to remove the stop words and Create a Bag-Of-Words model for Term-Document Matrix
# Convert a collection of text documents to a matrix of token counts
# Learn the vocabulary dictionary and return term-document matrix.
count_vect = CountVectorizer(stop_words='english',lowercase=False)
TD_counts = count_vect.fit_transform(comment_stemdata.CommentsTokenizedStemmed)
TermDoc_Matrix = pd.DataFrame(TD_counts.toarray())
TermDoc_Matrix.to_csv('./TermDoc_Matrix.csv')

#Compute TF-IDF Matrix from Term-Document Matrix
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(TD_counts)
TFIDF_Matrix = pd.DataFrame(X_tfidf.toarray())
TFIDF_Matrix.to_csv('./TFIDF_Matrix.csv')

#Merge customer data and comments TFIDF matrix data
combined_data = pd.concat([X, TFIDF_Matrix], axis=1)

#Do one Hot encoding for categorical features in customer data
X_cat = ["Sex","Status","Car_Owner","Paymethod","LocalBilltype","LongDistanceBilltype"]
combined_data_ohe = pd.get_dummies(combined_data,columns=X_cat)

combined_data_ohe.to_csv('./combined_data_ohe.csv')

df_kbselectedfeatures= pd.DataFrame()

#Feature selection - Filter Type - KBest
for i in (25,50,100):
    selector = SelectKBest(score_func=chi2, k=i)
    selectedfeatures = selector.fit_transform(combined_data_ohe,y)
    feature_names_out = selector.get_support(indices=True)
    df_kbselectedfeatures= pd.DataFrame(selectedfeatures)
    df_kbselectedfeatures.to_csv('./Combined_KBest_k'+str(i)+'.csv')
    
    #Construct a Random Forest Classifier on text data
    clf=RandomForestClassifier()
    RF_text = clf.fit(df_kbselectedfeatures,y)
    print('KBest Results k =  ', i)
    print("Accuracy score (training): {0:.6f}".format(clf.score(df_kbselectedfeatures, y)))
    rf_predictions = clf.predict(df_kbselectedfeatures)
    print("Confusion Matrix:")
    print(confusion_matrix(y, rf_predictions))
    print("Classification Report: ")
    print(classification_report(y, rf_predictions))
    
#Feature selection - Wrapper Type - SelectFromModel RandomForestClassifier
selector = SelectFromModel(estimator=RandomForestClassifier())
selectedfeatures = selector.fit_transform(combined_data_ohe,y)
feature_names_out = selector.get_support(indices=True)
df_rfselectedfeatures= pd.DataFrame(selectedfeatures)
df_rfselectedfeatures.to_csv('./Combined_RFModelSelectedFeatures.csv')
#Construct a Random Forest Classifier on text data
clf=RandomForestClassifier()
RF_text = clf.fit(df_rfselectedfeatures,y)
print('SelectFromModel RandomForestClassifier Results: ')
print("Accuracy score (training): {0:.6f}".format(clf.score(df_rfselectedfeatures, y)))
rf_predictions = clf.predict(df_rfselectedfeatures)
print("Confusion Matrix:")
print(confusion_matrix(y, rf_predictions))
print("Classification Report")
print(classification_report(y, rf_predictions))

#Feature selection - Wrapper Type - SelectFromModel GradientBoostingClassifier
selector = SelectFromModel(estimator=GradientBoostingClassifier())
selectedfeatures = selector.fit_transform(combined_data_ohe,y)
feature_names_out = selector.get_support(indices=True)
df_gbselectedfeatures= pd.DataFrame(selectedfeatures)
df_gbselectedfeatures.to_csv('./Combined_GBModelSelectedFeatures.csv')
#Construct a GradientBoostingClassifier  on text data
clf=GradientBoostingClassifier()
gb_text = clf.fit(df_gbselectedfeatures,y)
print('SelectFromModel GradientBoostingClassifier Results: ')
print("Accuracy score (training): {0:.6f}".format(clf.score(df_gbselectedfeatures, y)))
gb_predictions = clf.predict(df_gbselectedfeatures)
print("Confusion Matrix:")
print(confusion_matrix(y, gb_predictions))
print("Classification Report")
print(classification_report(y, gb_predictions))

# Final models For Filter Type - KBest
# Split the combined ohe data with best set of features into training and test set 80-20
X_train, X_test, y_train, y_test = train_test_split(df_kbselectedfeatures, y, test_size = 0.20)

#Construct a Random Forest Classifier on train/test data
clf=RandomForestClassifier()
RF_CombBestFeat_Split = clf.fit(X_train,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train, y_train)))
y_pred = clf.predict(X_test)
print('KBest (K=100) RandomForestClassifier Results: ')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))

#Construct a GradientBoostingClassifier on train/test data
clf=GradientBoostingClassifier()
gb_CombBestFeat_Split = clf.fit(X_train,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train, y_train)))
y_pred = clf.predict(X_test)
print('KBest (K=100) GradientBoostingClassifier Results: ')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))

# FFinal models For Wrapper Type - SelectFromModel
# Split the combined ohe data with best set of features into training and test set 80-20
X_train, X_test, y_train, y_test = train_test_split(df_rfselectedfeatures, y, test_size = 0.20)

#Construct a Random Forest Classifier on train/test data
clf=RandomForestClassifier()
RF_CombBestFeat_Split = clf.fit(X_train,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train, y_train)))
y_pred = clf.predict(X_test)
print('Wrapper Type RandomForestClassifier Results: ')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))

#Construct a GradientBoostingClassifier on train/test data
clf=GradientBoostingClassifier()
gb_CombBestFeat_Split = clf.fit(X_train,y_train)
print('Wrapper Type GradientBoostingClassifier Results: ')
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train, y_train)))
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))

