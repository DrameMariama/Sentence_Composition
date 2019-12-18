import numpy as np
def cosine_similarities(vect1, vect2):
    return np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2))
def compute_similarities(simple1, simple2, eval_sent):
    acc = 0
    misclassified = []
    for i in range(len(simple1)):
        vect_sum = simple1[i] * simple2[i]
        sim = []

        for j in range(len(eval_sent[i])):
            sim.append((cosine_similarities(vect_sum,eval_sent[i][j]), j))
        sim1 = sim.copy()
        sim1.sort(key=lambda tup: tup[0], reverse=True)
        if sim1[0][1] == sim[0][1]:
            acc += 1
        else:
            ind_misclassified = i
            misclassified.append((ind_misclassified, sim1[0][1]))
    return acc / len(simple2)


if __name__ == "__main__":
    dim = 1024
    a = np.fromfile('/private/home/mariama20/devfair_sentence_composition/web-split/paraphrase_sampled_overlapp.raw',  dtype=np.float32, count=-1)                                                                          
    a.resize((a.shape[0] // dim) // 6, 6,  dim)  
    print(a.shape) 

    b = np.fromfile('Data/valid_data/simple1.raw',  dtype=np.float32, count=-1)  
    b.resize(b.shape[0] // dim, dim)

    print(b.shape)

    c = np.fromfile('Data/valid_data/simple2.raw',  dtype=np.float32, count=-1)  
    c.resize(c.shape[0] // dim, dim)
    print(c.shape)

    print(compute_similarities(b, c, a))
