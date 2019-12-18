import torch
import numpy as np

def encode_sentence(model, sentence):
    tokens = model.encode(sentence)
    last_layers_tokens = model.extract_features(tokens)
    tensor = torch.mean(last_layers_tokens, dim=1)
    return tensor

def embed_sentence_roberta(datafile, model):
    with open(datafile, 'r') as f:
        with open('/private/home/mariama20/devfair_sentence_composition/web-split/Roberta/Data/valid_data/paraphrase_sampled_overlapp', 'w') as f1:
                i = 0
                
                for line in f: 
                    
                    line = line.replace('\n', '')
                    vect = encode_sentence(model, line).detach().numpy()
                    np.savetxt(f1, vect)
                    i += 1
                    if i % 100 == 0:
                        print('write sentence number ', i)
def main():
    datafile = '/private/home/mariama20/devfair_sentence_composition/web-split/paraphrase_sampled_overlapp' 
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()
    embed_sentence_roberta(datafile, roberta)
if __name__ == "__main__":
    main()