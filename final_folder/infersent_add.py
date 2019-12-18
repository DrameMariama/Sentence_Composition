import numpy as np
def cosine_similarities(vect1, vect2):
    return np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2))
def compute_similarities(simple1, simple2, eval_sent):
    acc = 0
    misclassified = []
    for i in range(len(simple1)):
        vect_sum = simple1[i] + simple2[i]
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
    dim = 4096
    a = np.loadtxt('Data/valid_data/paraphrase')                                                                         
    a = np.resize(a, (a.shape[0] // 10, 10,  dim)) 
    print(a.shape) 

    b = np.loadtxt('Data/valid_data/simple1')  
   

    print(b.shape)

    c = np.loadtxt('Data/valid_data/simple2')  

    print(c.shape)

    print(compute_similarities(b, c, a))
