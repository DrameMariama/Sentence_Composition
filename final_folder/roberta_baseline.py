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
    # dim = 1024
    a = np.loadtxt('/private/home/mariama20/devfair_sentence_composition/web-split/Roberta/Data/valid_data/paraphrase_sampled_overlapp')
    a = np.resize(a, (a.shape[0] // 6, 6, 1024))
    print(a.shape)
     
    b = np.loadtxt('Data/valid_data/true_complex')
    print(b.shape)
    acc, misclassified = compute_similarities(b, a) 
    print(acc)  
    # print(misclassified)                       
    # a = np.loadtxt('Data/valid_data/paraphrase_sampled')
    # a.resize(a.shape[0] // 6, 6, 1024)
    # print(a.shape)