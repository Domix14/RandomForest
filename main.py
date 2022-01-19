# CART on the Bank Note dataset

from RandomForest import RandomForest
from DecisionTree import DecisionTree
from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
def create_classes(data):
    Y = []
    count = 0
    classesNames = dict()
    for row in data:
        if row[-1] not in classesNames:
            classesNames[row[-1]] = count
            count += 1
        Y.append(classesNames[row[-1]])
        row[-1] = classesNames[row[-1]]
    return [X, Y, dataset]

def calculate_accuracy(dataset, prediction):
    prediction_accuracy = list()
    for i in range(len(dataset)):
        if dataset[i][-1] == prediction[i]:
            prediction_accuracy.append(1)
        else:
            prediction_accuracy.append(0)
    return sum(prediction_accuracy) / len(prediction)




filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)


for i in range(len(dataset[0])):
     str_column_to_float(dataset, i)

max_depth = 5
min_size = 10

# Nasz decision tree
tree = DecisionTree()
tree.fit(dataset)
predictions = tree.predict(dataset)
print("Our decision tree accuracy: " + str(calculate_accuracy(dataset, predictions)))

# Nasz random forest
forest = RandomForest()
forest.fit(dataset, len(dataset)/2, 6)
predictions = forest.predict(dataset)
print("Out random forest accuracy: " + str(calculate_accuracy(dataset, predictions)))




