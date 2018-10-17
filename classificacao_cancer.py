from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from scipy.spatial import distance
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import DistanceMetric

def PegaDados():
    dados = np.loadtxt("cancer.data", delimiter=",") # pega o dataset
    label_bruto = open("cancer-label.data", 'r')

    label = np.zeros(569).reshape((569))
    c = 0
    for l in label_bruto:
        #print(l)
        if(l == "M\n"):
            #print("entrou M")
            label[c] = 1
        elif(l == "B\n"):
            #print("entrou B")
            label[c] = 0
        c = c + 1
    label = label.astype(int)
    
    return dados, label

def k_means(dados,n):
    kmeans = KMeans(n_clusters = n, random_state = 0).fit(dados)

    return kmeans

def separa_normais(dados, labels):
    benignos = []
    t = 0
    for i in range(0, dados.shape[0]):
        if(labels[i] == 0):
            benignos.append(dados[i])
            t += 1
        
    return benignos, t

def main():
    dados, labels = PegaDados()
    benignos, t = separa_normais(dados, labels)
    kmeans = k_means(benignos,1)
    print(kmeans.labels_)
    centro = kmeans.cluster_centers_
    print("\ncentro:\n",centro)
    benignos = np.asarray(benignos)
    #linhas = benignos
    #print("linhas: ",linhas)
    dist = DistanceMetric.get_metric('euclidean')
    print("\n\n\n",sum(dist.pairwise(benignos,centro))/benignos.shape[0])



    
    


main()