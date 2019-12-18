import spacy
import random
import csv
nlp = spacy.load("en")
nlp.max_length = 15000000
def split_text_into_sentence():
    with open('validation.tsv') as tsvfile:
        with open('Data/first_data/eval_files/simple1_sent', 'w') as f:
            with open('Data/first_data/eval_files/simple1_questions', 'w') as f1:
                with open('Data/first_data/eval_files/complex_sent', 'w') as f2:
                    with open('Data/first_data/eval_files/complex_questions', 'w') as f3:
           
                        reader = csv.reader(tsvfile, delimiter='\t')
                        simple_answers = []
                        complex_answers = []
                        i=0
                        for line in reader:
                            comp = line[0]
                        
                            sent = comp
                            sent2 = list(sent)
                            ner_sent = []
                            token_sent = []
                            sent = nlp(sent)
                            for ent in sent.ents:
                                ner_sent.append((ent.text, ent.start_char, ent.end_char))

                            comp2 = line[1].split('<::::>')[0]
                            sent_1 = comp2
                            sent_2 = list(sent_1)
                            ner_sent_1 = []
                            token_sent_1 = []
                            sent_1 = nlp(sent_1)
                            for ent in sent_1.ents:
                                ner_sent_1.append((ent.text, ent.start_char, ent.end_char))
                                
                            if len(ner_sent) != 0 and len(ner_sent_1) != 0:
                                random_index = random.choice(ner_sent)
                                answer = ''.join(sent2[random_index[1]:random_index[2]+1])
                                sent2[random_index[1]:random_index[2]+1] = '[MASK] '
                                sent2 = ''.join(sent2)
                                sent2 = sent2.strip()
                                f3.write(sent2+'\n')
                                f2.write(line[0]+'\n')
                                complex_answers.append(answer)

                                random_index = random.choice(ner_sent_1)
                                answer1 = ''.join(sent_2[random_index[1]:random_index[2]+1])
                                sent_2[random_index[1]:random_index[2]+1] = '[MASK] '
                                sent_2 = ''.join(sent_2)
                                sent_2 = sent_2.strip()
                                f1.write(sent_2+'\n')
                                f.write(comp2+'\n')
                                simple_answers.append(answer1)

                             
                            
            with open("Data/first_data/eval_files/complex_answers", 'w') as f1:
                for answer in complex_answers:
                    f1.write(answer +'\n')
            with open("Data/first_data/eval_files/simple_answers", 'w') as f1:
                for answer in simple_answers:
                    f1.write(answer +'\n')
            
if __name__ == "__main__":
    split_text_into_sentence()
    # #for sents in data.sents
    # with open('valid_complex_sent', 'r') as f:
    #     n = 0
    #     for line in f:
    #         n += 1
    #     print(n)