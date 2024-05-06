#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:18:34 2022

@author: nitinsinghal
"""

from gensim import corpora
from gensim import models
import pyLDAvis
import pyLDAvis.gensim_models
import spacy
import pandas as pd
import re

nlp = spacy.load("en_core_web_lg")

data = pd.read_csv('/espressodata.csv')

documents = []
for i in range(len(data)):
    text = data['MsgBody'][i]
    doc = nlp(text)
    if len(doc) <= 5: # exclude comments with five or less than five words
        pass
    else:
        documents.append(re.sub("[^a-zA-Z]+ ", "",text))

processed_documents = [] # list for storing tokenized documents
for document in documents: 
    temp_list = [] # list for storing tokens in a document
    for token in nlp(document):
        if token.is_stop is True: # check whether the provided token is a stop word and decide whether to disregard it
            pass
        elif not nlp.vocab.has_vector(str(token)): # check whether the provided token is in a vocabulary
            pass
        elif len(token) < 3:
            pass
        else:
            temp_list.append(str(token.lemma_)) # lemmatize
    processed_documents.append(temp_list) 

dictionary = corpora.Dictionary(processed_documents) # index words with integers
corpus = [dictionary.doc2bow(sentence) for sentence in processed_documents] # TF representation
tfidf = models.TfidfModel(corpus) # Fit TF-IDF
corpus_tfidf = tfidf[corpus] # Transform "corpus" into TF-IDF

no_topics = 10
lda_model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=no_topics) # train LDA model

for topic in lda_model.show_topics(num_topics=no_topics, num_words=5):# show 5 most important words for each topic
    print(topic) 
for topic_proportion in lda_model[corpus_tfidf[0]]: # show topic distribution for the first document in the corpus
    print(topic_proportion)

lda_model.print_topics(num_topics=no_topics, num_words=5)

for doc, topic_dist in zip(documents, lda_model[corpus_tfidf]): # print out topic distribution for each document in the corpus
    print(doc, ": ", topic_dist)
    break

lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus_tfidf, dictionary)
pyLDAvis.save_html(lda_visualization, '/coffeeroasters_lda_result.html')


# Coherence Measures
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_documents, dictionary=dictionary, coherence='c_v') # by changing the values for the "coherence" parameter, you can use different coherence methods (e.g., 'u_mass', 'c_v', 'c_uci', 'c_npmi')
coherence_lda = coherence_model_lda.get_coherence()
print(coherence_lda)

# Coherence measures over different K
num_of_topics = []
coherence = []
for k in range(no_topics-1):
    lda_model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=k+2) # train LDA model; k+2 because we are only intereseted when K >= 2
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_documents, dictionary=dictionary, coherence='c_v') # calculate coherence measures
    coherence_lda = coherence_model_lda.get_coherence()
    num_of_topics.append(k+2)
    coherence.append(coherence_lda)


# Visualizaing coherence measures over different K
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.plot(num_of_topics, coherence) # x-axis = K, y-axis = coherece measures
plt.xticks(num_of_topics) # To label x-axis with K values
plt.show()

top_topics = lda_model.top_topics(corpus) 
avg_topic_coherence = sum([t[1] for t in top_topics]) / no_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

# Extract words from each top topic for wordlcoud
topic_words = ''
for i in range(no_topics):
    for j in range(5):
        topic_words += top_topics[i][0][j][1] + ' '

# Wordcloud of topics
from wordcloud import WordCloud

wordcloud = WordCloud(background_color ='white').generate(topic_words)                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Sentiment analysis using TextBlob and vaderSentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

words = TextBlob(topic_words)
print(words.sentiment)

analyzer = SentimentIntensityAnalyzer()
print(analyzer.polarity_scores(topic_words))


# data_words = ''
# for i in range(len(processed_documents)):
#     for j in range(len(processed_documents[i])):
#         data_words += processed_documents[i][j] + ' '

# dwords = TextBlob(data_words)
# print(dwords.sentiment)

# danalyzer = SentimentIntensityAnalyzer()
# print(danalyzer.polarity_scores(data_words))





























