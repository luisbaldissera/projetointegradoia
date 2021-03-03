import math
from rna import SVM
import statistics

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from base import euclidean, ClassificationResults

def knn(xtrain, ytrain, xtest, ytest, k = 1):
    ypred = []
    for i in range(len(xtest)):
        ki = 0
        # Heap mínimo de distâncias (mínimo sempre em [0])
        heap = []
        # X relativo a distância calculada
        xij = []
        # Y relativo a distância calculada
        yij = []
        # Maior valor do heap
        heapmax = -math.inf
        # Índice do maior valor do heap
        maxindex = 0
        # Percorre conjunto de treino e preenche heap
        for j in range(len(xtrain)):
            # Calcula distância euclidiana
            distance = euclidean(xtest[i], xtrain[j])
            # Preenche os k primeiros elementos
            # A partir o k-ésimo elemento, o heap não pode mais crescer
            if ki < k:
                heap.append(distance)
                xij.append(xtrain[j])
                yij.append(ytrain[j])
                if distance > heapmax:
                    heapmax = distance
                    maxindex = ki
                ki += 1
                if distance < heap[0]:
                    heap[0], heap[-1] = heap[-1], heap[0]
                    xij[0], xij[-1] = xij[-1], xij[0]
                    yij[0], yij[-1] = yij[-1], yij[0]
            # Se a distância é menor que o heapmax, ela deve ser inserida no heap
            elif distance < heapmax:
                newelement, newxij, newyij = distance, xtrain[j], ytrain[j]
                # Se a distância é menor do que o heap mínimo, ela deve ficar no começo
                if distance < heap[0]:
                    newelement, newxij, newyij = heap[0], xij[0], yij[0]
                    heap[0], xij[0], yij[0] = distance, xtrain[j], ytrain[j]
                if k > 1:
                    heap[maxindex], xij[maxindex], yij[maxindex] = newelement, newxij, newyij
                # Atualiza 'maxindex'
                for l in range(k):
                    if heap[l] > heap[maxindex]:
                        maxindex = l
                        heapmax = heap[maxindex]
        # Prediz y utilizando a moda
        yi = statistics.mode(yij)
        # Adiciona y preditivo em 'ypred'
        ypred.append(yi)
    # Calcula acurácia
    accurracy = sum([ yi == yt for yi,yt in zip(ypred, ytest) ]) / len(ytest)
    # Retorna os resultados
    return ClassificationResults(
        x = xtest,
        y = ypred,
        accurracy = accurracy
    )


def nb(xtrain, ytrain, xtest, ytest):
    classifier = GaussianNB()
    classifier.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    accurracy = metrics.accuracy_score(ytest, ypred)
    return ClassificationResults(
        x = xtest,
        y = ypred,
        accurracy = accurracy
    )


def dt(xtrain, ytrain, xtest, ytest):
    classifier = DecisionTreeClassifier()
    classifier.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    accurracy = metrics.accuracy_score(ytest, ypred)
    return ClassificationResults(
        x = xtest,
        y = ypred,
        accurracy = accurracy
    )

def svm(xtrain, ytrain, xtest, ytest):
    classifier = SVM(xtrain, ytrain)
    ypred = classifier(xtest)
    accurracy = metrics.accuracy_score(ytest, ypred)
    return ClassificationResults(
        x = xtest,
        y = ypred,
        accurracy = accurracy
    )

