import numpy as np

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
    dim = 1024
    paraphrase  = np.fromfile('/private/home/mariama20/devfair_sentence_composition/web-split/paraphrase_sampled_overlapp.raw',  dtype=np.float32, count=-1)                                                                          
    paraphrase.resize((paraphrase.shape[0] // dim) // 10, 10,  dim)  
    print(paraphrase.shape) 
    true_complex = np.fromfile('Data/valid_data/true_complex.raw',  dtype=np.float32, count=-1)                                                                          
    true_complex.resize(paraphrase.shape[0] // dim,  dim)     

    print(true_complex.shape) 
    acc, misclassified = compute_similarities(true_complex, paraphrase) 
    print(acc)  
                      