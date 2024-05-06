#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:50:15 2022

@author: nitinsinghal
"""

# import required packages
import spacy
import numpy as np
import pandas as pd

# load the Spacy language model
nlp = spacy.load("en_core_web_lg")

# a default list of stop words set by the Spacy language model
stopwords = nlp.Defaults.stop_words
print(stopwords)

# variables to store term statistics
num_of_comments = 0
unique_word = set() # using the set-type variable since it does not allow duplicates > able to count the number of unique words
num_of_token_per_comment = [] # using the list-type varailbe since we want to measure corpus-level statistics (e.g., avg, max, min, median, etc.)
total_number_of_tokens = 0
unique_author = set() 
time_stamp_list = [] # able to measure the number of comments by day, week, etc.
reply_count = 0
unique_submission = set() 

data = pd.read_csv('/cafedata.csv')

for i in range(len(data)):
    text = data['MsgBody'][i]
    doc = nlp(text)
    num_of_comments += 1
    num_of_tokens = len(doc)
    token_count = 0
    for token in doc:
        if token.is_stop is True:
            pass
        else:
            unique_word.add(str(token).lower())
            token_count += 1
            total_number_of_tokens += 1
    # staistics regarding authors
    num_of_token_per_comment.append(token_count)
    author_name = str(data['Author'][i])
    unique_author.add(author_name.lower())
    # statistics regarding time_stamp
    time_stamp = data['Timestamp'][i]
    time_stamp_list.append(time_stamp)
    # statistics regarding replies
    reply_to = data['ReplyTo'][i]
    if reply_to == "-":
        pass
    else:
        reply_count += 1
    # statistics regarding submissions
    thread_id = data['ThreadID'][i]
    unique_submission.add(thread_id)

# statistics
print("number of comments:", num_of_comments)
print("number of unique words:", len(unique_word))
print("total number of words in the corpus:", total_number_of_tokens)
print("average number of words in comments:", np.mean(np.asarray(num_of_token_per_comment)))
print("maximum number of words in comments:", np.max(np.asarray(num_of_token_per_comment)))
print("minimum number of words in comments:", np.min(np.asarray(num_of_token_per_comment)))
print("median number of words in comments:", np.median(np.asarray(num_of_token_per_comment)))
print("number of unique authors:", len(unique_author))
print("number of comments replying to other comments:", reply_count)
print("number of submissions:", len(unique_submission))







































