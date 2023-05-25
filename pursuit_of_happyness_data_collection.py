# -*- coding: utf-8 -*-
"""Pursuit of Happyness - Data collection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tNypMzP0YYbmFajHto-Y8kzvGrTV1dIc
"""
#  requirements to run this notebook
!apt-get install openjdk-8-jdk-headless -qq > /dev/null # install java
!wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz # download spark
!tar xf spark-3.1.1-bin-hadoop3.2.tgz                    # untar spark
!pip install -q findspark                             # install findspark
!pip install emoji                                  # install emoji

# setting up environment variables
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.1.1-bin-hadoop3.2"

from google.colab import drive
drive.mount('/content/drive')

import requests
import time  


# Get posts from a subreddit and Process them
def processPosts(posts, sub, directory):
    f = open(f'{directory}/{sub}-reddit.csv', "a", encoding="utf-8")
    for post in posts:
       # add a delimiter "DELIM" to separate the features
       s = str(post['created_utc']) + "DELIM" + " ".join(post['title'].splitlines()) + "DELIM" + " ".join(post['selftext'].splitlines())  + "DELIM" + str(post['score']) + "\n"
       f.write(s)
    f.close()
    return posts[-1]['created_utc']

# Get URL for a subreddit
def getURL(sub, before):
    url = "https://api.pushshift.io/reddit/submission/search/?size=1000&sort=created_utc&subreddit=" + sub
    if before:
        url += "&before="+str(before)
    return url

# Fetch posts from a list of subreddits 
def fetchRedditPosts(subreddits, directory):
    for sub in subreddits:
        beforeTime = int(time.time())
        print(f"Fetching posts for {sub} before: ", beforeTime)
        while True:
            response = requests.get(getURL(sub, beforeTime))

            if response.status_code == 200:
                # The request was successful
                posts = response.json()['data']

                if len(posts) == 0:
                    break
                beforeTime = processPosts(posts, sub, directory)
                print(f"\nNow will fetch {sub} posts before: ", beforeTime)
            elif response.status_code == 429:
                print("Ratelimited. Trying again in a minute..")
                time.sleep(60)
            else:
                # The request was unsuccessful
                print('Request failed with status code:', response.status_code)
                #Try again
 
# call another py file to combine all the files into one
from utils import combineFiles

# Get Mental Health data
MHdirectory = "Data/MH"

# List of Mental Health subreddits
MHsubreddits = ["Stress", "Anxiety", "depression", "AnxietyDepression", "mentalhealth", "bipolar", "SuicideWatch", "MentalHealthSupport", "mentalillness", "socialanxiety", "Anxietyhelp", "PanicAttack", "Anger", "rant"]
for sub in MHsubreddits:

    # call the function to fetch posts from a subreddit
    fetchRedditPosts(sub, MHdirectory)

# save all the files into one
MHdatapath = os.path.join(MHdirectory, "mental-health.csv")
files = [sub+"-reddit.csv" for sub in MHsubreddits]
combineFiles(MHdirectory, files, MHdatapath)

with open("univ-reddit-handles", "r") as fi:
    univ_reddits = fi.read().split(",")

for univ in univ_reddits:
    fetchRedditPosts(sub, "Data/University")

# Get NON-Mental Health data
!wget https://huggingface.co/datasets/sentence-transformers/reddit-title-body/resolve/main/reddit_title_text_2021.jsonl.gz
!gunzip reddit_title_text_2021.jsonl.gz


# import libraries
import time
import json
import numpy as np
from numpy.linalg import pinv
import findspark
import sys
findspark.init()
from pyspark.sql import SparkSession
from pyspark import AccumulatorParam
import numpy as np

# create spark session
spark = SparkSession.builder \
		.master("local[*]") \
		.appName("Project") \
		.getOrCreate()  

# read the data
data = spark.sparkContext.textFile("reddit_title_text_2021.jsonl")
print("Total no. of lines", data.count())

# load the data
data = data.map(json.loads)

print("\n\nCheckpoint: ", data.count())
sub_broadcast = spark.sparkContext.broadcast(MHsubreddits + univ_reddits)

# filter out the posts from the mental health subreddits
def filterFunc(rec):
    MHsubs = sub_broadcast.value
    return rec["subreddit"] not in MHsubs

def mapFunc(post):
    # We are using placeholder Timestamp and Scores because it does not matter for training.
    # We are adding them to keep it uniform with the university reddit data.
    placeholderTS = "1683942542"
    placeholderScore = "1"
    return placeholderTS + "DELIM" + " ".join(post['title'].splitlines()) + "DELIM" + " ".join(post['body'].splitlines())  + "DELIM" + placeholderScore


data = data.filter(filterFunc).map(mapFunc)
data.saveAsTextFile("Data/Non-MH")

spark.stop()

# call another py file to preprocess the data
from utils import preprocessData

preMHdir = "/Data/MH/Preprocessed"
# Get preprocessed Mental Health data
preprocessData("/Data/MH/mental-health.csv", preMHdir)

preNMHdir = "/Data/Non-MH/Preprocessed"
# Get preprocessed Non-Mental Health data
preprocessData("/Data/Non-MH", preNMHdir)

# call combineParts to combine all the files into one
def combineParts(directory, output_file):
    # List files in the directory
    files = os.listdir(directory)

    # Filter files with the name pattern "part-<5 digit number>"
    filtered_files = [file for file in files if file.startswith("part-")]
    combineFiles(directory, filtered_files, output_file)

# Combine all the parts into one
combineParts(preMHdir, "Model/train_MH")
combineParts(preNMHdir, "Model/train_NMH")