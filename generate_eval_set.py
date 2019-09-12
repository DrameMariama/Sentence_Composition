import json
import numpy as np
import random
import pandas as pd
random.seed(41)
def randomly_sample_n_index_without_replacement(n, testset):
    sampled_index = random.sample(range(len(testset)), n)
    return sampled_index

def generate_eval_data(n, testset):
    new_test_data = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2','True_Complex_Sentence',  'Sampled_Complex_Sentences'])
    for i in range(len(testset)):
        data_without_complex_sentence_i = testset[testset['Complex_sentence_index']!= i+1]
        data_without_duplicate_complex_sentences = data_without_complex_sentence_i.drop_duplicates(subset='Complex_sentence_index', keep='last')
        sampled_index = randomly_sample_n_index_without_replacement(n, data_without_duplicate_complex_sentences)
        sampled_complex_sentences = []
        for j in sampled_index:
            sampled_complex_sentences.append(data_without_duplicate_complex_sentences['Complex_sentence'].iloc[j])
        new_test_data = new_test_data.append({'Complex_sentence_index':i+1, 'Simple_sentence_1': testset['Simple_sentence_1'].iloc[i], 'Simple_sentence_2': testset['Simple_sentence_2'].iloc[i],
                                                 'True_Complex_Sentence': testset['Complex_sentence'].iloc[i],  'Sampled_Complex_Sentences': sampled_complex_sentences}, ignore_index=True)
        
    return new_test_data
if __name__ == "__main__":
    testdata = pd.read_csv('test_data.csv')
    n = 5
    new_test_data = generate_eval_data(n, testdata)
    new_test_data.to_csv('new_test_data.csv')
    print(new_test_data)