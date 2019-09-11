import pandas as pd
import json
def get_sentence_pairs(filename):
    df = pd.DataFrame(columns=['Complex_sentence_index' ,'Simple_sentence_1', 'Simple_sentence_2', 'Complex_sentence']) #will contain our new dataset
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
                            df = df.append({'Complex_sentence_index':i, 'Simple_sentence_1': sentences[0], 'Simple_sentence_2': sentences[1], 'Complex_sentence': complex_sentence}, ignore_index=True)
                        j += 1
                    next_line = f.readline()
                    #print(next_line)
                    if next_line.replace('\n', '') == 'COMPLEX-'+str(i+1) or next_line == '\n' or next_line == '' or next_line == ' ':
                        break
                line = next_line
                i += 1
            if line == '' or line == '\n' or line == ' ':
                break
    return df

def get_all_sentences(filename):
    with open(filename, 'r') as f:
        f = iter(f)  #transform f to an iterable object
        i = 1     #to trck the number of complex sentence
        line = f.readline()
        json_list = []
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
                        #k = 0

                        while not next_line1.startswith('COMPLEX-'+str(i)+':MR-1:SIMPLE-'+str(j)+':SPTYPE-'):
                            #print(next_line1.replace('\n', ''))
                            sentences.append(next_line1.replace('\n', ''))
                           # k += 1
                            next_line1 = f.readline()
                        data = {                               #create new entry in the json file
                                "Sentences": {
                                "Complex_sentence": complex_sentence,
                                 "Simple_sentences": sentences

                            }
                        }
                        json_list.append(data)
                        j += 1
                    next_line = f.readline()
                    #print(next_line)
                    if next_line.replace('\n', '') == 'COMPLEX-'+str(i+1) or next_line == '\n' or next_line == '' or next_line == ' ':
                        break

                line = next_line
                i += 1
                if line == '' or line == '\n' or line == ' ':
                    break
        #the file complete_data.json is opened in appending mode and if it does not exist it will be created
        #make sure to delete its contains if you run this function multiple times otherwise it will just append new lines
        with open('complete_data.json', 'a', encoding='utf-8') as f1:
            json.dump(json_list, f1, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    filename = 'output.txt'
    df = get_sentence_pairs(filename)
    df.to_csv('two_sentences_composition.csv')
    get_all_sentences(filename)
    with open('complete_data.json') as data_file:
        data = json.loads(data_file.read())
    print(len(data))