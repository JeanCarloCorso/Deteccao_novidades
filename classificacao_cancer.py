from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from scipy.spatial import distance
from sklearn.cluster import KMeans
import numpy as np

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

def k_means(dados):
    kmeans = KMeans(n_clusters = 1, random_state = 0).fit(dados)

    return kmeans

def separa_normais(dados, labels):
    benignos = []

    for i in range(0, dados.shape[0]):
        if(labels[i] == 0):
            benignos.append(dados[i])
        
    return benignos

def main():
    dados, labels = PegaDados()
    benignos = separa_normais(dados, labels)
    kmeans = k_means(benignos)
    print(kmeans.labels_)

    


main()