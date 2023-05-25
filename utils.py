# Description: Utility functions for preprocessing data
# import all the required libraries
import re
import string
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

import findspark
findspark.init()
from pyspark.sql import SparkSession

# function to preprocess text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Replace all punctuation characters except ' with spaces
    translation_table = str.maketrans(string.punctuation.replace("'", ""), " " * (len(string.punctuation) - 1))
    text = text.translate(translation_table)

    # Remove ' from the processed text
    text = text.replace("'", "")
    
    # Remove emojis
    text = emoji.demojize(text)
    
    # Remove emoji placeholders
    text = re.sub(r":[a-zA-Z_]+:", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back to text
    processed_text = " ".join(tokens)
    
    return processed_text

# # Example usage
# input_text = "I can't believe, how stressed I am right now! 😫"
# preprocessed_text = preprocess_text(input_text)
# print(preprocessed_text)

# function to preprocess text
def preprocessRec(rec, eval):
    rec = rec.split("DELIM")
    data = f"{rec[1]} {rec[2]}"
    data = data.replace("[removed]", "")
    data = data.replace("[deleted]", "")
    preprocessed = preprocess_text(data)
    if eval:
        return [rec[0], preprocessed]
    return preprocessed

# spark implementation of preprocessing
def preprocessData(file, eval = False):
    spark = SparkSession.builder \
            .master("local[*]") \
            .getOrCreate()  

    data = spark.sparkContext.textFile(file)
    print("\nCheckpoint 1: ", data.count())

    data = data.map(lambda x: preprocessRec(x, eval))

    data.saveAsTextFile("MH")
    spark.stop()

# function to combine the files
def combineFiles(directory, files, combined_file):
    count = 0

    # Open the combined file in write mode
    with open(combined_file, "a") as outfile:

        # Iterate over the parts of the dataset
        for part_file in files:
            count += 1
            print(f"Concatenating {part_file}...")
            part_file_path = os.path.join(directory, part_file)

            # Open each part file in read mode
            with open(part_file_path, "r") as infile:
                
                # Read the content of the part file and write it to the combined file
                outfile.write(infile.read())

    print(f"Total number of files: {count}")