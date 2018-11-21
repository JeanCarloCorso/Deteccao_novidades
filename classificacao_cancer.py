from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from scipy.spatial import distance
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix

def distancia(dados, ponto):
    modelo = DistanceMetric.get_metric('euclidean')
    dist = modelo.pairwise(dados, ponto)
    return dist

def pertence_class(dados, ponto, k=3):

    dist = distancia(dados, ponto)
    posicao = np.zeros(dist.shape[0]*dist.shape[1]).reshape(dist.shape[0],dist.shape[1])
    for i in range(0, posicao.shape[0]):
        posicao[i] = int(i)
    dist_pos = np.concatenate((dist, posicao), axis=1)
    dist_pont = [900000,0]
    for i in range(0, dist_pos.shape[0]):
        if(dist_pont[0] > dist_pos[i][0]):
            dist_pont = dist_pos[i]
    dist = distancia(dados, dados[int(dist_pont[1])].reshape(1,30))
    dist = np.sort(dist,axis=0)
    soma = 0
    for i in range(1, k+1):
        soma += dist[i]
    dist_media = soma/k
    if dist_pont[0] <= dist_media:
        return True
    return False


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
    centro = kmeans.cluster_centers_
    benignos = np.asarray(benignos)
    #linhas = benignos
    #print("linhas: ",linhas)

    dist = DistanceMetric.get_metric('euclidean')
    dist_media = sum(dist.pairwise(benignos,centro))/benignos.shape[0]

    distancia = dist.pairwise(dados,centro)
    #print(centro)
    rotulos = distancia
    for i in range(np.asarray(distancia).shape[0]):
        if(distancia[i] > dist_media):
            rotulos[i] = 1
        else:
            rotulos[i] = 0

    rotulos = np.asarray(rotulos).reshape((rotulos.shape[0])).astype(int)
    
    acuracia = np.sum(labels == rotulos)/labels.shape[0]
    confusao = confusion_matrix(labels.ravel(), rotulos)

    print("\n------MÉDIA-TOTAL------\n")

    print("acuracia: ",acuracia, "\nConfusao: \n", confusao,"\nBenignos: ",confusao[0][0]/(confusao[0][0]+confusao[0][1]),"\nMalignos: ",confusao[1][1]/(confusao[1][0]+confusao[1][1]))
    rotulo2 = []
    for i in range(0,dados.shape[0]):
        if(pertence_class(benignos, dados[i].reshape(1,30), 3)):
            rotulo2.append(0)
        else:
            rotulo2.append(1)
    
    print("\n------K-VIZINHOS------\n")

    rotulo2 = np.asarray(rotulo2).reshape((labels.shape[0])).astype(int)
    
    acuracia = np.sum(labels == rotulo2)/labels.shape[0]
    confusao = confusion_matrix(labels.ravel(), rotulo2)

    print("acuracia: ",acuracia, "\nConfusao: \n", confusao,"\nBenignos: ",confusao[0][0]/(confusao[0][0]+confusao[0][1]),"\nMalignos: ",confusao[1][1]/(confusao[1][0]+confusao[1][1]))

main()