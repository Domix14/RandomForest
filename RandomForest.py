from DecisionTree import DecisionTree
from random import randrange

class RandomForest:
    def __init__(self):
        self.trees = list()
    
    # Zwraca losowy podzbiór danych
    # Umożliwia to stworzenie losowych, niezależnych
    # drzewek decyzyjnych używając tego samego zbioru danych
    # Argumenty:
    # dataset - pełny zbiór danych
    # minSimpleSize - minimalna ilość danych w podzbiorze
    def subsample(self, dataset, minSampleSize):
        sample = list()
        while len(sample) < minSampleSize:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    # Tworzy las losowy na podstawie danych wejściowych
    # Argumenty:
    # dataset - pełny zbiór danych
    # samplesSize - rozmiar podzbiorów
    # treeCount - ilość drzewek do stworzenia
    # max_depth - maksymalna głebia drzewka
    # minimalny rozmiar drzewka
    def fit(self, dataset, samplesSize, treeCount, max_depth=5, min_size=5):
        for treeIndex in range(treeCount):
            sample = self.subsample(dataset, samplesSize)
            tree = DecisionTree()
            tree.fit(sample, max_depth, min_size)
            self.trees.append(tree)
    
    # Przewiduje wynik na wszystkich drzewkach i zwraca najczęściej występujący wynik
    # test - dane do przetestowania
    def predict(self, test):
        sum_prediction = list()
        for tree in self.trees:
            prediction = tree.predict(test)
            sum_prediction.append(prediction)
        
        final_prediction = list()
        for i in range(len(test)):
            prediction_count = dict()
            for prediction in sum_prediction:
                if prediction[i] in prediction_count:
                    prediction_count[prediction[i]] += 1
                else:
                    prediction_count[prediction[i]] = 1
            final_prediction.append(max(prediction_count))
        return final_prediction
        

