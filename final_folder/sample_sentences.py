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
    new_test_data = pd.DataFrame(columns=['Simple_sentence_1', 'Simple_sentence_2','True_Complex_Sentence',  'Paraphrase_Sentences', 'Sampled_Sentences'])
    index = np.arange(len(val_set))
    
    for i in range(len(val_set)):
        index1 = np.delete(index, i)
        sampled_sentences = []
        nb_overlapp = []
        comp_sent = val_set['complex_sentence'].iloc[i]
        for j in index1:
            sent = val_set['complex_sentence'].iloc[j]
            nb_overlapp.append((count_word_overlap(comp_sent, sent), j))
        sorted_nb = sorted(nb_overlapp, key=lambda tup: tup[0], reverse=True)
        for k in range(n):
            sampled_sentences.append(val_set['complex_sentence'].iloc[sorted_nb[k][1]])
        new_test_data = new_test_data.append({'Simple_sentence_1': val_set['simple_sentences1'].iloc[i], 'Simple_sentence_2': val_set['simple_sentences2'].iloc[i],'True_Complex_Sentence': val_set['complex_sentence'].iloc[i],  'Paraphrase_Sentences': val_set['paraphrase'].iloc[i], 'Sampled_Sentences': sampled_sentences}, ignore_index=True)
    return new_test_data
if __name__ == "__main__":
    testdata = pd.read_csv('paraphrase_dataset.csv')
    n = 5
    new_test_data = generate_eval_data(n, testdata)
    new_test_data.to_csv('paraphrase_word_overlapping.csv')
    