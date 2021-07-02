import csv
import numpy
import pandas
import random
import warnings
import prettytable
import sklearn.metrics
import sklearn.model_selection
warnings.filterwarnings("ignore")



class Evaluate:

    def __init__(self, filename, k=7):
        self.k = k
        self.folds = []
        self.accuracy_metric = []
        self.f1_metric = []
        self.precision_metric = []
        self.recall_metric = []
        self.filename = filename
        self.dataset = []
        self.import_dataset()
        self.cross_validate()
        self.evaluate()


    def import_dataset(self):
        with open(self.filename, "rt") as dataset_csvfile:
            dataset_reader = csv.reader(dataset_csvfile, delimiter=",")
            self.dataset = list(dataset_reader)


    def cross_validate(self):
        fold_size = int(len(self.dataset)/self.k)
        for i in range(self.k):
            fold = []
            while len(fold) < fold_size:
                index = random.randrange(len(self.dataset))
                fold.append(self.dataset.pop(index))
            self.folds.append(fold)


    def average(self, array):
        return round(sum(array)/len(array), 3)


    def accuracy(self, originalset, prediction):
        hits = 0
        for a, b in zip(originalset, prediction):
            if a == b:
                hits += 1
        return hits/len(originalset)


    def evaluate(self):
        for fold in list(self.folds):
            dataset_array = numpy.array(fold)
            dataset = pandas.DataFrame(dataset_array, columns=["ClassName", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"])
            NAIVEBAYES = Naivebayes()
            NAIVEBAYES.dataset = dataset
            Y = dataset['ClassName']
            X = dataset.iloc[:,1:]
            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)
            NAIVEBAYES.train_model(X_train, Y_train)
            prediction = NAIVEBAYES.predict(X_test, Y_test) 
            self.accuracy_metric.append(self.accuracy(Y_test, prediction))
            self.f1_metric.append(sklearn.metrics.f1_score(Y_test, prediction, average='macro'))
            self.precision_metric.append(sklearn.metrics.precision_score(Y_test, prediction, average='macro'))
            self.recall_metric.append(sklearn.metrics.recall_score(Y_test, prediction, average='macro'))


    def display(self):
        t = prettytable.PrettyTable(["Fold #", "Accuracy", "F1", "Precision", "Recall"])
        print(" [+] Evaluation Metric Scores: \n")
        for index in range(self.k):
            accuracy = round(self.accuracy_metric[index], 3)
            f1 = round(self.f1_metric[index], 3)
            precision = round(self.precision_metric[index], 3)
            recall = round(self.recall_metric[index], 3)
            t.add_row([index+1, accuracy, f1, precision, recall])
        t.add_row(["AVERAGE", self.average(self.accuracy_metric), self.average(self.f1_metric), self.average(self.precision_metric), self.average(self.recall_metric)])
        print(t)


class Naivebayes:

    def __init__(self):
        self.model = {}


    def summarize_class(self):
        self.class_prior = dict(zip(self.labels.value_counts().index, self.labels.value_counts().values))
        self.data_points = sum(self.class_prior.values())
        self.class_dict = {}
        for class_ in self.class_prior.keys():
            self.class_dict[class_] = self.class_prior[class_]
            self.class_prior[class_] = self.class_prior[class_]/ self.data_points


    def summarize_feature(self):
        self.feature_columns = list(self.columns_details.keys())
        feature_prior = {}
        for feature in self.feature_columns:
            values = self.columns_details[feature]
            values = values.astype('float')
            for value in values.keys():
                values[value] = values[value]/self.data_points
            feature_prior[feature] = dict(values)
        
        dataframe_temporary = pandas.concat([self.features, self.labels], axis = 1)
        for column in self.feature_columns:
            column_details = pandas.DataFrame(self.columns_details[column])
            for class_ in self.class_prior.keys():
                number_of_value_feature = []
                for value in feature_prior[column]:
                    index_of_class = dataframe_temporary[dataframe_temporary[self.labels.name] == class_]
                    count_of_class = index_of_class[self.dataset[column] == value].shape[0]
                    number_of_value_feature.append(count_of_class)
                column_details[class_] = value = number_of_value_feature
            self.columns_details[column] = column_details


    def probability(self):
        denominator = list(self.class_dict.values())
        for column in self.feature_columns:
            feature = self.columns_details[column]
            feature_dicts = {}
            for index, row in feature.iterrows():
                value = row[1:]
                if 0 in list(row):
                    value = [((value[i]+1)/denominator[i]) for i in range(len(value))]
                else:
                    value = [(value[i]/denominator[i]) for i in range(len(value))]
                classes = list(self.class_prior.keys())
                value_dicts = dict(zip(classes, value))
                feature_dicts[index] = value_dicts
            self.model[column] = feature_dicts
        self.model[self.labels.name] = self.class_prior


    def train_model(self, features, labels):
        self.columns_details = {}
        self.labels = labels
        self.features = features
        for column in self.features.columns:
            self.columns_details[column] = self.features[column].value_counts()
        del column

        self.summarize_class()
        self.summarize_feature()
        self.probability()


    def predict(self, features, Y_test):
        classes = list(self.model[Y_test.name].keys())
        class_predictions = []
        for i, row in features.iterrows():
            class_prob = []
            for class_ in classes:
                probabilities = 1
                for index, value in row.iteritems():
                    try:
                        probabilities = probabilities * self.model[index][value][class_]
                    except:
                        probabilities = probabilities
                probabilities = probabilities * self.model[Y_test.name][class_]
                class_prob.append(probabilities)
            index_max = numpy.argmax(class_prob)
            class_predictions.append(classes[index_max])
        return class_predictions


def main():
    print("\n" + "="*70)
    print(" Discrete Naive Bayes Algrithm")
    print("="*70 + "\n")
    random.seed(1)
    EVALUATE = Evaluate("balance-scale.csv")
    EVALUATE.display()


main()