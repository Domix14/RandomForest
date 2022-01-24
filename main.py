# CART on the Bank Note dataset

from RandomForest import RandomForest
from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from csv import reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Ładuje plit csv
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset
 
# Konwertuje dane string na float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Zamienia klasy tekstowe na liczbowe
def create_classes(dataset):
    Y = []
    count = 0
    classesNames = dict()
    for row in dataset:
        if row[-1] not in classesNames:
            classesNames[row[-1]] = count
            count += 1
        Y.append(classesNames[row[-1]])
        row[-1] = str(classesNames[row[-1]])
    return dataset

# Oblicza dokładność
def calculate_accuracy(dataset, prediction):
    prediction_accuracy = list()
    for i in range(len(dataset)):
        if dataset[i][-1] == prediction[i]:
            prediction_accuracy.append(1)
        else:
            prediction_accuracy.append(0)
    return sum(prediction_accuracy) / len(prediction)


# Dzieli dane oraz wyniki
def split_labels(dataset):
    X = []
    Y = []
    for row in dataset:
        X.append(row[0:-1])
        Y.append(row[-1])
    return [X, Y]


def split_data(dataset, ratio):
    learn_data_size = int(len(dataset) * ratio)
    learn_data = dataset[0:learn_data_size]
    test_data = dataset[learn_data_size:len(dataset)]
    return [learn_data, test_data]

# Porównuje dokładność dla różnej ilości drzewek
def tree_count_test(dataset):
    [learn_data, test_data] = split_data(dataset, 0.75)
    acurracies = []
    treeCount = []
    for i in range(1, 20):
        treeCount.append(i)
        forest = RandomForest()
        forest.fit(dataset, len(dataset), i, 5, 5)
        predictions = forest.predict(test_data)
        accuracy = calculate_accuracy(test_data, predictions)
        acurracies.append(accuracy)
    print(acurracies)
    print(treeCount)
    plt.plot([i for i in treeCount], [i for i in acurracies], 'o', color='black')
    plt.xlabel('Tree count')
    plt.ylabel('Accuracy')
    plt.title("Accuracy vs Tree count")
    plt.show()

# Porównuje nasz algorytm z zbudowanym algorytmem   
def sklearn_test(dataset):
    [learn_data, test_data] = split_data(dataset, 0.75)
    [X, Y] = split_labels(dataset)
    [X_t, Y_t] = split_labels(test_data)
    our_forest_accurancies = []
    sklearn_forest_accurancies = []
    treeCount = []
    for i in range(1, 50):
        treeCount.append(i)
        model = RandomForestClassifier(max_depth=5, min_samples_split=5, n_estimators=i)
        model.fit(X, Y)
        sklearn_forest_predictions = model.predict(X_t)
        sklearn_forest_accurancy = calculate_accuracy(test_data, sklearn_forest_predictions)
        sklearn_forest_accurancies.append(sklearn_forest_accurancy)

        forest = RandomForest()
        forest.fit(dataset, len(dataset)/2, i, 5, 5)
        our_forest_predictions = forest.predict(test_data)
        our_forest_accuracy = calculate_accuracy(test_data, our_forest_predictions)
        our_forest_accurancies.append(our_forest_accuracy)

    plt.plot([i for i in treeCount], [i for i in sklearn_forest_accurancies], 'o', color='red')
    plt.plot([i for i in treeCount], [i for i in our_forest_accurancies], 'o', color='blue')
    plt.xlabel('Tree count')
    plt.ylabel('Accuracy')
    plt.title("Our algorithm(blue) vs Sklearn algorithm(red)")
    plt.show()



filename = 'iris.csv'
dataset = load_csv(filename)

dataset = create_classes(dataset)

# zamiana danych na float
for i in range(len(dataset[0])):
     str_column_to_float(dataset, i)

#tree_count_test(dataset)

#sklearn_test(dataset)

max_depth = 5
min_size = 10

# # Nasz decision tree
# tree = DecisionTree()
# tree.fit(dataset)
# predictions = tree.predict(dataset)
# print("Our decision tree accuracy: " + str(calculate_accuracy(dataset, predictions)))

# # Nasz random forest
# forest = RandomForest()
# forest.fit(dataset, len(dataset)/2, 6)
# predictions = forest.predict(dataset)
# print("Out random forest accuracy: " + str(calculate_accuracy(dataset, predictions)))




