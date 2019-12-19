import json
import numpy as np
import random
import pandas as pd
import time
random.seed(41)
def count_word_overlap(sent_1, sent_2):
    sent_1_split = sent_1.split()
    sent_2_split = sent_2.split()
    n = 0 
    n1 = len(sent_1_split)
    n2 = len(sent_2_split)
    if n1 < n2:
        shortest_sent = sent_1_split
        longuest_sent = sent_2_split
    else:
        shortest_sent = sent_2_split
        longuest_sent = sent_1_split
    for word in shortest_sent:
        if word in longuest_sent:
            n += 1
    return n

def generate_eval_data(n, val_set):
    new_test_data = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2','True_Complex_Sentence',  'Sampled_Complex_Sentences'])
    for i in range(len(val_set)):
        data_without_complex_sentence_i = val_set[val_set['Complex_sentence_index']!= i+1]
        data_without_duplicate_complex_sentences = data_without_complex_sentence_i.drop_duplicates(subset='Complex_sentence_index', keep='last')
        #sampled_index = randomly_sample_n_index_without_replacement(n, data_without_duplicate_complex_sentences)
        word_overlap = {}
        for j in range(len(data_without_duplicate_complex_sentences)):
            word_overlap[data_without_duplicate_complex_sentences['Complex_sentence'].iloc[j]] = count_word_overlap(
                val_set['Complex_sentence'].iloc[i],
                data_without_duplicate_complex_sentences['Complex_sentence'].iloc[j])
        print(word_overlap.values())
        word_overlap = sorted(word_overlap.items(), key=lambda kv: kv[1], reverse=True)
        sampled_complex_sentences = word_overlap[:n]

        new_test_data = new_test_data.append({'Complex_sentence_index':i+1, 'Simple_sentence_1': val_set['Simple_sentence_1'].iloc[i], 'Simple_sentence_2': val_set['Simple_sentence_2'].iloc[i],
                                                 'True_Complex_Sentence': val_set['Complex_sentence'].iloc[i],  'Sampled_Complex_Sentences': [sampled_complex_sentences[k][0] for k in range(len(sampled_complex_sentences))]}, ignore_index=True)
    return new_test_data
if __name__ == "__main__":
    testdata = pd.read_csv('val_data.csv')
    n = 100
    new_test_data = generate_eval_data(n, testdata)
    new_test_data.to_csv('new_val_data_hits100_by_word_overlap.csv')
   