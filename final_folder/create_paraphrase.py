import os
import pandas as pd
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
import nltk
import glob
from collections import defaultdict
import xml.etree.ElementTree as ET 
import json
import csv

import pandas as pd
import json
def get_sentence_pairs(filename, train_dev_test_file):
    with open(train_dev_test_file, 'r') as f_split: #train_dev_test_file correspond to the Websplit file which contains the training valid and test split
        data_split = json.loads(f_split.read())
    val = data_split['VALIDATION']
    train = data_split['TRAIN']
    n_train_examples = 0
    n_test_examples = 0
    n_val_examples = 0
    with open(filename, 'r') as f:  
        with open('training_paraphrase1', 'w') as f1:
                f = iter(f)  #transform f to an iterable object
                i = 1     #to trck the number of complex sentence
                line = f.readline()
                while line:
                    s = line.replace('\n', '')
                    if s == 'COMPLEX-'+str(i):  # check if the current line is a new complex sentence
                        #print(s)
                        next_line = f.readline()
                        complex_sentence = next_line.replace('\n', '')
                        
                        j = 1 #to track the number of the simple sentence
                        while True:
                            #print(next_line)
                            if next_line.replace('\n', '') == 'COMPLEX-'+str(i)+':MR-1:SIMPLE-'+str(j):  #check if the current line is defining simple sentences generated from the current complex sentence
                                sentences = []
                                next_line1 = f.readline()
                                k = 0

                                while not next_line1.startswith('COMPLEX-'+str(i)+':MR-1:SIMPLE-'+str(j)+':SPTYPE-'):
                                    #print(next_line1.replace('\n', ''))
                                    sentences.append(next_line1.replace('\n', ''))
                                    k += 1
                                    next_line1 = f.readline()
                                #print(j, k)
                                
                                
                                if k == 2:  # to consider  complex sentences that are splitted only into two simple sentences
                                    
                                    
                                #   print(next_line1.replace('\n', ''))
                                    if i in train:
                                        pos = f.tell()
                                        categories = []
                                        lin = f.readline()
                                        n = 1
                                        while 'COMPLEX' not in lin:
                                            
                                            if 'category' in lin:    ## the category is used to define paraphrases
                                                categories.append(lin)
                                        
                                            lin = f.readline()
                                            n += 1
                                        f.seek(pos)
                                    
                                        n_train_examples += 1
                                        f1.write(complex_sentence+'\n')
                        
                                j += 1

                            next_line = f.readline()
                           
                            if next_line.replace('\n', '') == 'COMPLEX-'+str(i+1) or next_line == '\n' or next_line == '' or next_line == ' ':
                                break
                        line = next_line
                        i += 1
                    print('Line:  ', i)
                    if line == '' or line == '\n' or line == ' ':
                        break
def get_categories_file(datafile):
    data = pd.read_csv(datafile, converters={'category2': eval})
    df = pd.DataFrame(columns = ['id', 'category1', 'category2'])
    #with open('categories.txt', 'w') as f:
    for i in range(len(data)):
        j = 0
        print(data['category2'].iloc[i][0])
        s1 = data['category2'].iloc[i][0].replace('\n', '')
        if len(data['category2'].iloc[i]) > 1:  
            s2 = data['category2'].iloc[i][1].replace('\n', '')
        else:
            s2 = ''
        ind = data['Sentence_id'].iloc[i]
        df = df.append({'id': ind, 'category1': s1, 'category2': s2}, ignore_index=True)
    df.to_csv('training_categories.csv')
        
def get_paraphrase(datafile):
    df = pd.read_csv(datafile)
    arr = []
    for i in range(len(df)):
        ind = df['id'].iloc[i]
        c1 = (df['category1'].iloc[i].split())
        c2 = str(df['category2'].iloc[i]).split()
        j = 0
        while j < len(df):
            if df['id'].iloc[j] != ind:
                ind_prime = df['id'].iloc[j]
                c1_prime = df['category1'].iloc[j].split()
                c2_prime = str(df['category2'].iloc[j]).split()
                if c1[:2] == c1_prime[:2] and c2[:2] == c2_prime[:2]:  #test if their simple sentences are from the same triple id(category)
                    if (ind, ind_prime) not in arr:
                        arr.append((ind, ind_prime))
                        print(ind, ind_prime)
            j += 1
    return arr

if __name__ == "__main__":
    # FILE_NAME = 'output.txt'
    # SPLITTED_FILE = 'benchmark-v1.0/Split-train-dev-test.benchmark-v1.0.json'
    # get_sentence_pairs(FILE_NAME, SPLITTED_FILE)
    # get_categories_file('training_paraphrase1.csv')
    arr = get_paraphrase('training_categories.csv')
    arr = np.array(arr)
    np.savetxt('paraphrase_training_index',arr)
    # paraphrase = pd.read_csv('paraphrase1.csv')
    # paraphrase = paraphrase.drop_duplicates(subset='Sentence_id', keep='first')
    # new_paraphrase = pd.DataFrame(columns=['simple_sentences1', 'simple_sentences2', 'complex_sentence', 'paraphrase'])
    # a = np.loadtxt('paraphrase_index1')
    # #with open('paraphrase1.txt', 'w') as f:
    # for i in range(len(paraphrase)):
    #     sent_id = paraphrase['Sentence_id'].iloc[i]
    #     j = 0
    #     comp_sent = paraphrase['Complex_sentence'].iloc[i]
    #     while j < len(a) and sent_id != int(a[j][0]):
    #         j += 1
    #     if j < len(a):
    #         ind_prime = int(a[j][1])
    #         para_sent = paraphrase[paraphrase['Sentence_id']==ind_prime]
    #         p = para_sent['Complex_sentence'].iloc[0]
    #         new_paraphrase = new_paraphrase.append({'simple_sentences1': paraphrase['Simple_sentence_1'].iloc[i], 'simple_sentences2': paraphrase['Simple_sentence_2'].iloc[i], 'complex_sentence': comp_sent, 'paraphrase': p}, ignore_index=True)
    #         print(sent_id,ind_prime)
    # new_paraphrase.to_csv('paraphrase_dataset.csv')
                #f.write(comp_sent)
                # while k < len(a) and sent_id == int(a[k][0]):
                #     ind_prime = int(a[k][1])
                #     para_sent = paraphrase[paraphrase['Sentence_id']==ind_prime]
                    
                #     f.write('\t')
                #     p = para_sent['Complex_sentence'].iloc[0]
                #     f.write(p)
                #     k += 1
                #     print(sent_id,ind_prime)
                # f.write('\n')
                    