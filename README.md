# Common_NLP_Tasks_Libraries
Common NLP tasks (Sentiment, Keyword, Topic, Summary, etc) using main libraries (HuggingFace Transformers, Spacy, Pytorch, Tensorflow, Scikit, etc)

Idea is to show how to perform different NLP tasks using different algorithms and libraries and compare their results. 

Main NLP tasks covered are 
•	Sentiment Analysis
•	Classification
•	Summarization
•	Keyword Extraction
•	Topic Modelling

Libraries used are 
•	HuggingFace Transformers
•	KeyBERT
•	BERTopic
•	Spacy, Gensim
•	TextBlob, VaderSentiment
•	Rake
•	NLTK
•	Scikit-learn
•	WordCloud

Algorithms used are
•	BERT, DistilBERT – Bidirectional Encoder Representations from Transformers by Google. Trained on 3.3 billion words
•	LDA – Latent Dirichlet Allocation. Finds hidden topics, probabilistic neighboring terms
•	TF-IDF – Term Frequency Inverse Document Frequency – word importance 
•	CountVectorizer – word count across documents 
•	POS – Part of Speech tagging (Noun, Verb, etc)

Other algorithms that can be used on the basis of this code are 
•	Using Transformers - Roberta, GPT, etc
•	RNN, LSTM, etc

Generally, the new transformer based algorithms pre-trained on large language models perform better. They are an example of transfer learning, as it is hard to get large word datasets and expensive to train them. 
The older models are useful to in terms of simplicity, flexibility, and speed giving good results for small NLP problems.

The code example below shows how to use different NLP algorithms and libraries to solve different NLP tasks. The code is self-explanatory and links are given for each library and algorithm. 

Data
Data used is from Reddit for two subreddit threads – ‘/Food’ and ‘/Economics’. 
Only results for Food are shown. https://www.reddit.com/r/food/ 

Reddit data can be loaded using PRAW library and they have provided examples on how to do that. https://praw.readthedocs.io/en/stable/getting_started/quick_start.html 
Data from 2021 to current was loaded for upto 10,000 comments.

The data files are provided in github. Each file contains the subreddit’s topics and message (comments) by different users on each row. The comments were used for analysis. To find out general sentiment or feeling of users towards the economy or food as topics.  It can be applied to any text data like Twitter, blog, etc.

The data required some wrangling using regex for cleaning and junk data removal – hyperlinks, special characters, etc. Emoji’s are still there as they don’t impact analysis, but can be removed too using regex.

Warning: Reddit being an open forum, some of the comment words are profane or not appropriate, so if you see them in the results please ignore them. Objective is to show how to perform NLP analysis. There are ways to filter such data but that will be done separately.

Library, Algorithms

Transformers -  Text classification, Sentiment analysis, Summarization
HuggingFace Transformer pipeline was used to perform text classification, sentiment analysis and summarization. It’s a very good library to use as it provides access to most of the widely used large pre-trained models like BERT, RoBerta, GPT etc via an easy pipeline. You can train your models too using it, but that requires more time and resources. You can also directly use tensorflow or pytorch to do the same, but that too requires a lot of effort, but good way to learn how to build your own NLP models.
https://huggingface.co/transformers/v3.0.2/quicktour.html
https://huggingface.co/docs/transformers/task_summary
https://huggingface.co/docs/transformers/model_doc/bert

For some NLP tasks using Transformer BERT models only part of the data was analyzed due to size, but you can change that and check.

TextBlob and VaderSentiment were also used for sentiment analysis. They are quick and easy to use, give decent results. But Transformer based models give better results.

https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
https://github.com/cjhutto/vaderSentiment#code-examples


Keyword, Topic Modelling
Some specific libraries like KeyBERT for keyword extraction and BERTopic for Topic analysis were tried for popularity but the results are unclear. Might need more data to see better results.
https://github.com/MaartenGr/KeyBERT
https://github.com/MaartenGr/BERTopic

For topic modelling using Spacy/Gensim with LDA is the preferred approach and gives decent results. Spacy NLP pipeline with large English model was used.
https://spacy.io/usage/processing-pipelines#processing
https://radimrehurek.com/gensim/models/ldamodel.html

NLTK and scikit-learn can be used for tokenizing, lemmatizing and normalizing the text to get different word measures like countvector, tf-idf, etc.
https://www.nltk.org/book_1ed/
https://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage

Visualization
PylDavis, matplotlib and WordCloud libraies can be used for visualizing the topic terms in different ways.
I have avoided posting the wordcloud as some words are not appropriate, you can see that in the results if you run the notebook.

https://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb
https://amueller.github.io/word_cloud/auto_examples/simple.html#sphx-glr-auto-examples-simple-py

POS tagging
For POS tagging, spacy is quick and easy to use. The results from the model, using only noun or verb words, can then be used for different NLP tasks like keyword, topic modelling etc. This can sometimes be more accurate and efficient than passing the entire text data.
However, one of the limitations in this approach is it loses the context and meaning of the language as it ignores the order and location of words and sentences. That is where models like BERT are very good as they remove the need for normalizing the text to a dictionary of words.

https://spacy.io/usage/linguistic-features#pos-tagging

Customized multi model for keyword, topic extraction
Transformer based BERT or other models generally give better results when compared to Spacy, Gensim, etc. However for Topic modelling or keyword extraction, there is no well defined process or transformer pipeline. It requires combining different libraries and algorithms and is supposed to give better results due to the large pre-trained language model.
One approach would be to get a dictionary of main word tokens or POS category words and create an embedding using the BERT model and then perform cosine similarity to find the main words in the text. But that requires multiple libraries to work in one environment, which is difficult and not working at the moment as given in the Note below.
Partial code has been and is not complete. Will provide it later once it is working.

Note - faced known issues with Tensorflow/Pytorch/Transformers and other libraries on Macbook M1. There are lots of libraries and dependencies, eventually something breaks and very difficult to setup locally using miniforge, etc.
Used Google Colab as it works for most libraries, but when using transformer DistilBERT model to load the word embeddings in memory it fails due to large size. 
This problem requires a fully working local machine setup or a large RAM cloud based service like AWS, etc. 


