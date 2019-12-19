import pandas as pd
import json
def get_sentence_pairs(filename, train_dev_test_file):
    #df = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2', 'Complex_sentence']) #will contain our new dataset
    with open(train_dev_test_file, 'r') as f_split:
        data_split = json.loads(f_split.read())
    train = data_split['TRAIN']
    val = data_split['VALIDATION']
    test = data_split['TEST']
    n_train_examples = 0
    n_test_examples = 0
    n_val_examples = 0
    train_data  = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2', 'Complex_sentence'])
    val_data = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2', 'Complex_sentence'])
    test_data = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2', 'Complex_sentence'])
    with open(filename, 'r') as f:
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
                                print('in Train')
                                n_train_examples += 1
                                train_data = train_data.append({'Complex_sentence_index':i, 'Simple_sentence_1': sentences[0], 'Simple_sentence_2': sentences[1], 'Complex_sentence': complex_sentence}, ignore_index=True)
                            elif i in val:
                                print('in validation')
                                n_val_examples += 1
                                val_data = val_data.append({'Complex_sentence_index':i, 'Simple_sentence_1': sentences[0], 'Simple_sentence_2': sentences[1], 'Complex_sentence': complex_sentence}, ignore_index=True)
                            else:
                                print('in test')
                                n_test_examples += 1
                                test_data = test_data.append({'Complex_sentence_index':i, 'Simple_sentence_1': sentences[0], 'Simple_sentence_2': sentences[1], 'Complex_sentence': complex_sentence}, ignore_index=True)
                        j += 1
                    next_line = f.readline()
                    #print(next_line)
                    if next_line.replace('\n', '') == 'COMPLEX-'+str(i+1) or next_line == '\n' or next_line == '' or next_line == ' ':
                        break
                line = next_line
                i += 1
            if line == '' or line == '\n' or line == ' ':
                break
    return train_data, val_data, test_data, n_train_examples, n_val_examples ,n_test_examples

if __name__ == '__main__':
    FILE_NAME = 'output.txt'
    SPLITTED_FILE = 'Split-train-dev-test.benchmark-v1.0.json'
    train_data, val_data, test_data, n_train_examples, n_val_examples ,n_test_examples = get_sentence_pairs(FILE_NAME, SPLITTED_FILE)
    train_data.to_csv('train_data.csv')
    test_data.to_csv('test_data.csv')
    val_data.to_csv('val_data.csv')
    print(n_train_examples, '\n', n_val_examples , '\n', n_test_examples)
    