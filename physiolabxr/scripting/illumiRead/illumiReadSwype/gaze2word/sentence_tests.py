import nltk
from nltk.corpus import reuters, brown, gutenberg, inaugural, nps_chat
import csv
import random
import string
import pandas as pd
import re

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('reuters')
    nltk.download('brown')
    nltk.download('gutenberg')
    nltk.download('inaugural')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('nps_chat')

    # sentences = brown.sents()
    
    # posts = nps_chat.posts()
    
    # # Join each post into a sentence
    # sentences = [' '.join(post) for post in posts]
    
    # # Filter out sentences containing "PART" or "JOIN"
    # filtered_sentences = [sentence for sentence in sentences if "PART" not in sentence and "JOIN" not in sentence]

    # # Convert the filtered sentences to a DataFrame
    # df_filtered = pd.DataFrame(filtered_sentences, columns=['Sentence'])

    # # Convert the sentences to a DataFrame
    # # df = pd.DataFrame(sentences, columns=['Sentence'])

    # # Save the DataFrame to a CSV file
    # output_file_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\nps_chat_sentences.csv'
    # df_filtered.to_csv(output_file_path, index=False)

    # Re-run the filtering and saving process
    
    file_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\brown_sentences.csv'

    df = pd.read_csv(file_path, header=None)

    # Extract sentences from the first column
    sentences = df[0].tolist()

    # Define offensive words, political, and religious terms, and internet slang
    offensive_words = {'damn', 'hell', 'shit', 'fuck', 'bitch', 'bastard', 'asshole', 'ass', 'penis', 'pussy', 'dick'}
    political_religious_words = {'god', 'allah', 'jesus', 'bible', 'quran', 'church', 'mosque', 'temple', 'vote', 'election', 'president', 'minister', 'abortion', 'gays'}
    internet_slang = {'lol', 'hahaha', 'omg', 'brb', 'ttyl', 'rofl', 'lmao', 'wtf', 'btw', 'pm', 'ppl', 'omg', 'wtf'}

    # Function to check if a sentence is valid
    def is_valid_sentence(sentence):
        # Check for numbers
        if any(char.isdigit() for char in sentence):
            return False
        # Check for offensive words
        words = set(re.findall(r'\b\w+\b', sentence.lower()))
        if words & offensive_words:
            return False
        # Check for political or religious words
        if words & political_religious_words:
            return False
        # Check for internet slang
        if words & internet_slang:
            return False
        return True

    # Filter the sentences
    filtered_sentences = [sentence for sentence in sentences if is_valid_sentence(sentence)]

    # Filter sentences by word count
    short_sentences = [sentence for sentence in filtered_sentences if 5 <= len(sentence.split()) <= 12]
    long_sentences = [sentence for sentence in filtered_sentences if 13 <= len(sentence.split()) <= 20]

    # Select 20 sentences from each category
    short_sentences_sample = short_sentences
    long_sentences_sample = long_sentences

    # Combine the samples into one dataframe
    filtered_sample_df = pd.DataFrame(short_sentences_sample + long_sentences_sample, columns=['Sentence'])

    # Save the filtered sample sentences to a new CSV file
    output_file_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\output.csv'
    filtered_sample_df.to_csv(output_file_path, index=False)

    print(f"Filtered sentences saved to {output_file_path}")

    