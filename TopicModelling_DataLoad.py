#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 18:12:05 2022

@author: nitinsinghal
"""

# Use reddit and PRAW for text extraction
# Topic modelling, Sentiment analysis, summary statistics

import praw 
from psaw import PushshiftAPI
import datetime

#Reddit API Credentials
reddit = praw.Reddit(user_agent='',
                     client_id='', client_secret="",
                     username='', password='$',
                     ratelimit_seconds=600) #default is set to 5 seconds. Set it with a generous number so that your program does not fail                
	           
#subreddit = "espresso" 
#subreddit = "starbucks" 
#subreddit = "coffee" 
#subreddit = "cafe" 
subreddit = "coffee_roasters" 

#File write
f = open('./coffeeroastersdata.csv','w', encoding='utf8')	
#In this next line we print out column headers
f.write("MsgID,Timestamp,Author,ThreadID,ThreadTitle,MsgBody,ReplyTo,Permalink\n")

#Begin streaming user-generated comments from the focal subreddit specified in the 'subreddit' variable earlier in this code
count = 1
for comment in reddit.subreddit(subreddit).stream.comments():
	commentID = str(comment.id) 
	author = str(comment.author).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ") 
	timestamp = str(datetime.datetime.fromtimestamp(comment.created)) 
	replyTo = "" 
	if not comment.is_root: #If it is indeed a reply, this column contains the message ID of the parent message. If it is not a reply, a '-' is written to this column
		replyTo = str(comment.parent().id)
	else:
		replyTo = "-"
	threadID = str(comment.submission.id) 
	threadTitle = str(comment.submission.title).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ") 
	msgBody = str(comment.body).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ") 
	permalink = str(comment.permalink).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ") 
	
	#Print all collected message data to console
	print("-------------------------------------------------------")
	print("Comment ID: " + str(comment.id))
	print("Comment Author: "+ str(comment.author))
	print("Timestamp: "+str(datetime.datetime.fromtimestamp(comment.created)))
	if not comment.is_root:
		print("Comment is a reply to: " + str(comment.parent().id))
	else:
		print("Comment is a reply to: -")
	print("Comment Thread ID: " + str(comment.submission.id))
	print("Comment Thread Title: " + str(comment.submission.title))
	print("Comment Body: " + str(comment.body))
	print("Comment Permalink: " + str(comment.permalink))
	
    #Write everything to file
	f.write(commentID+","+timestamp+","+author+","+threadID+","+threadTitle+","+msgBody+","+replyTo+","+permalink+"'\n")
	print("Total messages collected from /r/"+subreddit+": " + str(count))
	count += 1
    
# To access historic data

#Existing File append historic data
f = open('/coffeeroastersdata.csv','a', encoding='utf8')	

api = PushshiftAPI()
gen = api.search_submissions(
    after=2021,
    filter=['id'], 
    subreddit=subreddit, 
    limit=1000
    )
for submission in gen:
    submission_id = submission.d_['id']
    # once you get submission id, you will use PRAW
    submission_praw = reddit.submission(id=submission_id)
    print(submission_praw.title)
    print(submission_praw.selftext)
    print(submission_praw.url)
    
    submission_praw.comments.replace_more(limit=None)
    for comment in submission_praw.comments.list():
        commentID = str(comment.id)
        if(comment.author):
            author = str(comment.author).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ")
        else:
            author = ""
        timestamp = str(datetime.datetime.fromtimestamp(comment.created))
        replyTo = "" 
        if not comment.is_root:
            replyTo = str(comment.parent().id)
        else:
            replyTo = "-"
        threadID = str(comment.submission.id)
        threadTitle = str(comment.submission.title).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ")
        msgBody = str(comment.body).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ")
        permalink = str(comment.permalink).replace(";", "").replace("'","").replace(",","").replace("\"","").replace("\n", " ").replace("\r"," ")
    
        print(comment.body)
        
        #Append everything to existing data file 
        f.write(commentID+","+timestamp+","+author+","+threadID+","+threadTitle+","+msgBody+","+replyTo+","+permalink+"'\n")
        
        print("***")




















































