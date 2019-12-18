import nltk
import numpy as np
import torch
from models import InferSent
nltk.download('punkt')
def embed_sent(datafile):
    sentences = []
    with open(datafile, 'r') as f:
        i = 0
        for line in f:
            line = line.replace('\n', '')
            sentences.append(line)
            i += 1
            if i == 455820:
                        break
    V = 1
    MODEL_PATH = 'infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = 'GloVe/glove.840B.300d.txt'
    infersent.set_w2v_path(W2V_PATH)

    infersent.build_vocab(sentences, tokenize=True)

    embeddings = infersent.encode(sentences, tokenize=True)

    np.savetxt("../../wiki-split/Data/Infersent_vectors/complex_sent", embeddings)

def cosine_similarities(vect1, vect2):
    return np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2))
def compute_similarities(comp_sent, eval_sent):
    acc = 0
    misclassified = []
    for i in range(len(comp_sent)):
        sim = []
        true_comp = comp_sent[i]
        for j in range(len(eval_sent[i])):
            sim.append((cosine_similarities(true_comp,eval_sent[i][j]), j))
        sim1 = sim.copy()
        sim1.sort(key=lambda tup: tup[0], reverse=True)
        if sim1[0][1] == sim[0][1]:
            acc += 1
        else:
            ind_misclassified = i
            misclassified.append((ind_misclassified, sim1[0][1]))
    return acc / len(comp_sent), misclassified
if __name__ == "__main__":
    datafile = '../../wiki-split/complex_sent'

    embed_sent(datafile)
    # a = np.loadtxt('Data/valid_data/true_complex')
    # print(a.shape)

    # b = np.loadtxt('Data/valid_data/paraphrase_10')
    # # b = b.reshape(b.shape[0]//11, 11, b.shape[1])
    # # print(b.shape)
    # b.resize(b.shape[0] // 11, 11, b.shape[1])
    # print(b.shape)
    # # acc, misclassified = compute_similarities(a, b) 
    # # print(acc)  
    # # print(misclassified)
