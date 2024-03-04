#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:18:45 2024

@author: nilsl
"""
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import time
import gc


def load_gpt_model(model_name='dbmdz/german-gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("mps")
    return model, tokenizer

def calculate_embeddings_gpt(text, model, tokenizer, max_length=1024, chunk_size = 64, batch_size=5):
    # Tokenize the input text
    total_start_time = time.time()
    tokens = tokenizer(text)
    inputs = []
    prob, entr, cos_sim, max_prob_diff, cond_entr_err = [], [], [], [], []
    n_tok = len(tokens["input_ids"])
    print(f"Num Tokens: {n_tok}")

    for i in range(chunk_size, len(tokens["input_ids"]), chunk_size): 
        start_index = max(0, i - 1024)
        end_index = i 
        inputs.append(tokens["input_ids"][start_index:end_index])
    # n_inp = len(inputs)
    # Initialize variable to store the last hidden state of the previous batch
    # print(f"Num Inputs: {n_inp}")
    
    for i in range(0, len(inputs), batch_size):
        # start_time = time.time()
        # print(i)
        input_ids_batch = pad_sequence([torch.tensor(seq) for seq in inputs[i:i+batch_size]], batch_first=True, padding_value= -100).to("mps")
        attention_masks_batch = torch.where(input_ids_batch != -100, 1, 0).to("mps")
        
        with torch.no_grad():
            outputs = model(input_ids_batch, attention_mask=attention_masks_batch, output_hidden_states=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]

            token_probs_batch, max_prob_diffs_batch, entropies_batch, conditional_entropy_error_batch, cosine_similarities_batch = calculate_metrics(
                logits, hidden_states, input_ids_batch, attention_masks_batch, chunk_size)
            prob += token_probs_batch
            entr += entropies_batch
            cos_sim += cosine_similarities_batch
            max_prob_diff += max_prob_diffs_batch
            cond_entr_err += conditional_entropy_error_batch

        # Clear memory
        del input_ids_batch, attention_masks_batch, outputs, logits, hidden_states
        gc.collect()
        torch.mps.empty_cache()
        # end_time = time.time()
        # elapsed = (end_time-start_time)/(batch_size * chunk_size)
        # print(f"Time per token iteration: {elapsed}")
    total_end_time = time.time()
    elapsed = (total_end_time-total_start_time)/n_tok
    print(f"Total time per token: {elapsed}")
    
    print(n_tok)
    print(len(prob))
    
    return prob, entr, cos_sim, max_prob_diff, cond_entr_err


def calculate_metrics(logits, hidden_states, input_ids, attention_mask, chunk_size):
    batch_size, seq_len, _ = logits.size()
    
    # Containers for the metrics
    token_probs = []
    max_prob_diffs = []
    entropies = []
    cosine_similarities = []
    conditional_entropy_error = []

    normalized_logits = torch.nn.functional.log_softmax(logits, dim=-1)
  
    probabilities = torch.exp(normalized_logits)
    
    for i in range(batch_size):
        actual_length = attention_mask[i].sum().item()  # Number of real tokens in the sequence
        start_index = max(actual_length - chunk_size, 0)  # Start from the first real token in the chunk
        for j in range(start_index, actual_length):  # Iterate over real tokens in the last chunk
            if j == 0:    
                print("aua")
                continue
            else:
                actual_current_token_id = input_ids[i, j].unsqueeze(-1)  # Get the actual next token ID
                token_prob = probabilities[i, j-1].gather(-1, actual_current_token_id).item()
                
                max_prob = torch.max(probabilities[i, j-1]).item()
                prob_diff = max_prob - token_prob

                entropy = -torch.sum(probabilities[i, j-1] * normalized_logits[i, j-1]).item()

                cos_sim = cosine_similarity(hidden_states[i, j-1], hidden_states[i, j], dim=-1).item()
         
                cond_entr_error = -np.log(token_prob) - entropy

            token_probs.append(token_prob)
            max_prob_diffs.append(prob_diff)
            entropies.append(entropy)
            cosine_similarities.append(cos_sim)
            conditional_entropy_error.append(cond_entr_error)

    return token_probs, max_prob_diffs, entropies, conditional_entropy_error, cosine_similarities



def combine_interview_parts(df):
    # Copy of the DataFrame to avoid modifying the original during iteration
    df_copy = df.copy()
    
    # Identifying unique study_ids and their items
    unique_study_ids = df['study_id'].unique()
    unique_metrics = df["metric"].unique()
    
    for study_id in unique_study_ids:
        for metric in unique_metrics:
        # Filtering the DataFrame for the current study_id
            df_study = df_copy[np.logical_and(df_copy['study_id'] == study_id, df_copy["metric"] == metric)]
            
            # Finding study items base names (without part1/part2)
            study_items_base = df_study['study_item'].str.replace('_teil[12]', '', regex=True).unique()
            
            for item_base in study_items_base:
                # Extract rows for part 1 and part 2
                part1_row = df_study[df_study['study_item'].str.contains(f'{item_base}_teil1')]
                part2_row = df_study[df_study['study_item'].str.contains(f'{item_base}_teil2')]
                
                # Ensure both parts exist before proceeding
                if not part1_row.empty and not part2_row.empty:
                    # Combining the lists from part 1 and part 2
                    combined_values = np.concatenate((part1_row['values'].values[0], part2_row['values'].values[0]))
                    
                    # Update the DataFrame: Assign combined values to the part 1 row and drop the part 2 row
                    df_copy.at[part1_row.index[0], 'values']  = combined_values
                    df_copy = df_copy.drop(part2_row.index)
                    
                    # Update the study_item to no longer reflect part1/part2
                    df_copy.at[part1_row.index[0], 'study_item'] = item_base
                    
    return df_copy

if __name__ == '__main__':
    
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    

    remove_stopwords = False
    remove_numbers = False
    subsample = False
    n = 5
    
    path = os.path.join("..", "..", "data", "VELAS", "velas_key.csv")
    key = pd.read_csv(path, dtype = str)[["study_id", "group"]]
    key.index = key["study_id"]
    key = key[key["group"]!="ls_alt"].drop(['2213', '2203', '1403', '1347'])
    
    if subsample:
        print(n)
        sub_sample = key.groupby("group").sample(n)
    else:
        print(key)
        n = np.inf
        sub_sample = key
    
    transcripts_dict = np.load(f"clean_transcripts_stopwords_{remove_stopwords}_numbers_{remove_numbers}.npy", allow_pickle=True).item()
    embeddings_dict = {}
    
    # # Load the model
    gpt_model, gpt_tokenizer = load_gpt_model()

    for study_id in sub_sample.index:
        print(study_id)
        embeddings_dict[study_id] = {} 
        for item in transcripts_dict[study_id].keys():
            print(item)
            data = ' '.join(transcripts_dict[study_id][item]["answers"])
            prob, entr, sim, max_prob_diff, cond_entr_err = calculate_embeddings_gpt(data, gpt_model, gpt_tokenizer)
            embeddings_dict[study_id][item] = {
                                    "probability": prob,
                                    "entropy": entr,
                                    "similarity": sim,
                                    "max_probability_difference": max_prob_diff,
                                    "conditional_entropy_error": cond_entr_err
                }
        
    # Flatten the nested dictionary
    flat_list = []
    for study_id, study_items in embeddings_dict.items():
        for study_item, metrics in study_items.items():
            for metric, values in metrics.items():

                flat_list.append({
                    "group": key.loc[str(study_id)]["group"],
                    "study_id": study_id,
                    "study_item": study_item,
                    "metric": metric,
                    "values": values
            })

    # Create DataFrame
    df_embeddings = pd.DataFrame(flat_list)

    # Apply the corrected combination function
    combined_df_metrics = combine_interview_parts(df_embeddings)
    combined_df_metrics.to_csv(f"velas_subsample_{n}_stopwords_{remove_stopwords}_numbers_{remove_numbers}.csv")
    combined_df_metrics.to_pickle(f"velas_subsample_{n}_stopwords_{remove_stopwords}_numbers_{remove_numbers}.pkl")
    
    