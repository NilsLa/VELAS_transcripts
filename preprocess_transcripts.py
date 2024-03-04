#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:10:22 2024

@author: nilsl
"""

import os
from striprtf.striprtf import rtf_to_text
import re
import chardet
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords


def split_into_lines(text):
    lines = text.splitlines()
    # Filtering out empty lines and lines containing '.mp3'
    filtered_lines = [line for line in lines if line.strip() and '.mp3' not in line and 'audio' not in line]
    return filtered_lines


def extract_and_remove_hashtags(text):
    # Regular expression pattern to find text between hashtags
    pattern = r'#(.*?)#'
    
    # Find all occurrences of the pattern
    matches = re.findall(pattern, text)
    
    # Remove them from the text
    text_without_hashtags = re.sub(pattern, '', text)

    return text_without_hashtags, matches


def clean_text(text, stopwords_list, remove_numbers = False):
    # Convert text to lowercase
    text = text.lower()
    
    # Replace special unicode characters with spaces
    text = text.replace('\xa0', ' ')
    
    # Remove all types of quotation marks
    text = re.sub(r'[\'"`]', '', text)
    
    # Remove backslashes and forward slashes
    text = text.replace('\\', '').replace('/', '')
    
    
    # Remove numbers and words that start with a number
    if remove_numbers:
        text = re.sub(r'\b\d+\w*|\w*\d+\w*', '', text)
    
    # Remove text within brackets (including the brackets)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)

    # Split the text into sentences
    sentences = sent_tokenize(text)

    cleaned_sentences = []
    for sentence in sentences:
        # Tokenize the sentence into words, filtering out punctuation
        words = [word for word in word_tokenize(sentence) if word.isalnum()]
        # Remove stopwords
        filtered_words = [word for word in words if word not in stopwords_list]
        # Check if the sentence is non-empty and not a single word after cleaning
        if len(filtered_words) > 1:
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)

    # Rejoin the cleaned sentences into a single text string
    cleaned_text = '. '.join(cleaned_sentences)
    cleaned_text += '.'
    
    # Remove unnecessary whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def read_rtf(file_path):
    # Detect the encoding of the file
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read the file with the detected encoding
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        rtf_content = file.read()

    return rtf_to_text(rtf_content)


def process_lines(lines, stopwords_list, remove_numbers = False):
    i_lines = []
    other_lines = []

    for line in lines:
        # Check if the line starts with 'I:' or 'I::'
        if line.startswith("I:") or line.startswith("I::"):
            # Remove 'I:' or 'I::' from the line
            cleaned_line = re.sub(r'^I::? ', '', line)
            cleaned_line = clean_text(cleaned_line, stopwords_list, remove_numbers)
            if len(cleaned_line)>1:
                i_lines.append(cleaned_line)
                
        else:
            # Remove any '*:' or '*::' pattern from other lines
            cleaned_line = re.sub(r'^.*::? ', '', line)
            if sum(char.isdigit() for char in cleaned_line) / len(line) < 0.3:
                cleaned_line = clean_text(cleaned_line, stopwords_list, remove_numbers)
                if len(cleaned_line)>1:
                    other_lines.append(cleaned_line)
                        
    return i_lines, other_lines


def parse_filename(filename):
    parts = filename.split('_')
    patient_id = parts[0]
    # Assuming the task is always the third part and the part information (e.g., "teil1", "teil2") is the fourth part
    task = parts[2] if len(parts) > 2 else 'unknown'
    part_info = parts[3].split('.')[0] if len(parts) > 3 else ''
    full_task = f"{task}_{part_info}" if part_info else task
    return patient_id, full_task


def load_rtf_files(directory):
    transcripts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".rtf"):
            file_path = os.path.join(directory, filename)
            patient_id, task = parse_filename(filename)
            if patient_id not in transcripts:
                transcripts[patient_id] = {}
            transcripts[patient_id][task] = read_rtf(file_path)
    return transcripts

if __name__ == '__main__':
    
    remove_stopwords = False
    remove_numbers = False
    
    # Adjusting the path to the directory containing transcripts
    directory = os.path.join("..", "..", "data", "VELAS", "transcripts")
    transcripts_dict = load_rtf_files(directory)
    
    final_data = {}
    
    question_freq = defaultdict(int)
    answer_freq = defaultdict(int)
    
    if remove_stopwords:
        stopwords_list = list(stopwords.words('german'))
        stop_list = stopwords_list + ['hm', 'mega', 'ja', 'voll', 'aber', 'eben', 'also', 'so', 'ah', 'aha', 'oder', 'bisschen', 'einfach', 'eigentlich', 'vielleicht', 'schon', 'halt', 'irgendwie', 'mal']
        
    else: stop_list = []
    
    for patient_id, tasks in transcripts_dict.items():
        final_data[patient_id] = {}
        for task, transcript in tasks.items():
            cleaned_transcript, hashtag_texts = extract_and_remove_hashtags(transcript)
            lines = split_into_lines(cleaned_transcript)
            i_lines, other_lines = process_lines(lines, stop_list, remove_numbers=False)
    
            for seq in i_lines:
                for word in word_tokenize(seq):
                    question_freq[word] += 1
                    
            for seq in other_lines:
                for word in word_tokenize(seq):
                    answer_freq[word] += 1
    
            # Store in nested dictionary
            final_data[patient_id][task] = {
                'questions': i_lines,
                'answers': other_lines
            }
            
    np.save(f"clean_transcripts_stopwords_{remove_stopwords}_numbers_{remove_numbers}.npy", final_data)