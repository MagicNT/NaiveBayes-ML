# Naive Bayes (Continuous / Discrete) Algorithm

## **Implementation (Continuous)**

### Scripting Language

- Python 3

### Dataset

- We will use the Iris Flower Species Dataset (Continuous) and we will attempt to predict the flower species given measurements of iris flowers. The dataset contains 150 observations with 4 input variables (Sepal length in cm, Sepal width in cm, Petal length in cm, Petal width in cm) and 1 output variable representing the class, and the total number of observations for each class is balanced.

- Dataset Link: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

- Dataset excerpt:

```
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor

```

### Import Libraries

- We import the csv Python library to read in the dataset saved inside a CSV file
- We import the math Python library to utilize its exponential and square root functions
- We import the random Python library to generate random values
- We import the prettytable Python library to present the output results in a nice tabular layout
- We import the sklearn Python library to utilize its metric calculation functions


```python
import csv
import math
import random
import prettytable
import sklearn.metrics
```

### Algorithm Evaluation

We will evaluate the algorithm via cross-validation by splitting the data into 7 folds (k=7) and computing the accuracy of predictions.

The general procedure for this evaluation is:
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
        - Take group as a test dataset
        - Take remaining groups as a training dataset
        - Fit a model on the training dataset and evaluate it on the test dataset
        - Retain the evaluation accuracy score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores


```python
class Evaluate:

	def __init__(self, dataset, func, k=7):
		self.k = k
		self.folds = []
		self.accuracy_metric = []
		self.f1_metric = []
		self.precision_metric = []
		self.recall_metric = []
		self.cross_validate(dataset)
		self.evaluate(dataset, func)


	def cross_validate(self, dataset):
		fold_size = int(len(dataset)/self.k)
		for i in range(self.k):
			fold = []
			while len(fold) < fold_size:
				index = random.randrange(len(dataset))
				fold.append(dataset.pop(index))
			self.folds.append(fold)


	def average(self, array):
		return round(sum(array)/len(array), 3)


	def accuracy(self, originalset, prediction):
		hits = 0
		for a, b in zip(originalset, prediction):
			if a == b:
				hits += 1
		return hits/len(originalset)


	def evaluate(self, dataset, func):
		for fold in list(self.folds):
			trainset = list(self.folds)
			trainset.remove(fold)
			trainset = sum(trainset, [])
			testset = [row for row in fold]
			originalset = [row[-1] for row in fold]
			prediction = func(trainset, testset)
			self.accuracy_metric.append(self.accuracy(originalset, prediction))
			self.f1_metric.append(sklearn.metrics.f1_score(originalset, prediction, average='macro'))
			self.precision_metric.append(sklearn.metrics.precision_score(originalset, prediction, average='macro'))
			self.recall_metric.append(sklearn.metrics.recall_score(originalset, prediction, average='macro'))


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
```

### Statistics Tools Class For Data Summary

To describe the given dataset we require to calculate the **mean** and the **standard deviation**. For sake of organization we implemented into one class below all the needed routines for us to call and utilize throughout the algorithm processing stages. We also implemented into this class the probability and class probability routines.

**Probability Routine:** Finding the likelihood of a real-value can be done by assuming that this value is taken from a Gaussian distribution which is summarized in terms of the mean and the standard deviation. We can estimate the probability of a certain value using a Gaussian Probability Distribution Function as in: *f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))*.


**Class Probability Routine:** Probabilities are calculated for each class; we calculate the probability that a new data belongs to the first class, then to the second class, and so on. The probability that a data belongs to a class is calculated as: *P(class|data) = P(X|class) * P(class)*. Note that the division part has been removed from the original formula to simplify the calculation, as we are more interested in the class prediction than the probability itself. The calculation for the class with highest value will be the prediction. Since this is a naive technique, the input variables are treated separately. For two input variables, the probability that a row belongs to the first class is: *P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0)*. The implemented routine for this procedure, takes a set of summaries and a new row as input arguments. First the total number of training data rows is obtained from the counts stored in the summary statistics, then used in the calculation of class probability (i.e., ratio of rows of a class to all rows in the training data). Probabilities are computed for each input value in the row, for the column and class using the Gaussian probability density function, then they are multiplied together as they accumulate. This procedure is repeated for each class in the dataset, and as a result a dictionary of probabilities is generated with one entry for each class.


```python
class Statistics:

	def __init__(self):
		pass


	def mean(self, numbers):
		return sum(numbers)/len(numbers)


	def standard_deviation(self, numbers):
		avg = self.mean(numbers)
		variance = sum([(x-avg)**2 for x in numbers])/(len(numbers)-1)
		return math.sqrt(variance)


	def probability(self, x, mean, stdev):
		math.exponent = math.exp(-((x-mean)**2/(2*stdev**2 )))
		return (1/(math.sqrt(2*math.pi)*stdev))*math.exponent


	def class_probabilities(self, summaries, row):
		total_rows = sum([summaries[label][0][2] for label in summaries])
		probabilities = dict()
		for class_value, class_summaries in summaries.items():
			probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
			for i in range(len(class_summaries)):
				mean, stdev, _ = class_summaries[i]
				probabilities[class_value] *= self.probability(row[i], mean, stdev)
		return probabilities
```

### Naive Bayes Algorithm Class

We implemented the algorithm in a modular form for better data management as well as better organization. In the initialization function of the class, we specify the required local variables (CSV dataset file name) and trigger the dataset importation process.


```python
class Naivebayes:

	def __init__(self, filename):
		self.dataset = list()
		self.filename = filename
		self.ST = Statistics()
		self.import_dataset()
```

### Importing/Pre-Processing Dataset

In the below routine we import the corresponding dataset file, convert all the found class strings into numeric indexes, then convert the originally interpreted strings into numeric values.


```python
	def import_dataset(self):
		with open(self.filename, "rt") as dataset_csvfile:
			dataset_reader = csv.reader(dataset_csvfile, delimiter=",")
			dataset_tmp = list(dataset_reader)

		class_values = [row[len(dataset_tmp[0])-1] for row in dataset_tmp if row]
		unique = set(class_values)
		lookup = dict()
		for i, value in enumerate(unique):
			lookup[value] = i
		
		for row in dataset_tmp[0:-2]:
			for x in range(len(dataset_tmp[0])):
				if x == len(dataset_tmp[0])-1 and row:
					row[x] = lookup[row[x]]
				elif row:
					row[x] = float(row[x].strip())
				self.dataset.append(row)
```

## Class Seperation

We calculate the "base rate" which is the probability of data by thei corresponding class. First we separate the training data by class. We create dictionary where each key is the class value and add a list of all the records as entry in the dictionary. The below routine impementation executes this procedure.


```python
	def separate_by_class(self, dataset):
		separated = dict()
		for i in range(len(dataset)):
			vector = dataset[i]
			class_value = vector[-1]
			if class_value not in separated:
				separated[class_value] = list()
			separated[class_value].append(vector)
		return separated
```

We require the mean and standard deviation statistics for each input attribute or each column in the data, by gathering all of the values for each column into list and calculating the statistics. We repeat this for each column in the dataset and return a list of tuples of statistics. Below is the routine implementation of this approach.


```python
	def summarize_dataset(self, dataset):
		summaries = [(self.ST.mean(column), self.ST.standard_deviation(column), len(column)) for column in zip(*dataset)]
		del(summaries[-1])
		return summaries
```

### Dataset Summary (Per Class)

The dataset is split per class, then statistics are calculated on each subset. The results in the form of a list of tuples of statistics are  stored in a dictionary by class. Below is the routine implementation to separate a dataset into rows per class. By using the previous summary function we calculate summary statistics for each column, thus together, we can summarize the columns in the dataset organized per class.


```python
	def summarize_by_class(self, dataset):
		separated = self.separate_by_class(dataset)
		summaries = dict()
		for class_value, rows in separated.items():
			summaries[class_value] = self.summarize_dataset(rows)
		return summaries
```

### Make Prediction

Below implemented routine manages the calculation of the probabilities of a new row for each class, then selects the one with the largest probability


```python
	def predict(self, summaries, row):
		probabilities = self.ST.class_probabilities(summaries, row)
		best_label, best_prob = None, -1
		for class_value, probability in probabilities.items():
			if best_label is None or probability > best_prob:
				best_prob = probability
				best_label = class_value
		return best_label
```

### Naive Bayes Algorithm Trigger Routine

The below implemented routine initiates the Naive Bayes algorithm, to learn the statistics from a training dataset then use them to make predictions for a test dataset


```python
	def naive_bayes(self, train, test):
		summarize = self.summarize_by_class(train)
		predictions = list()
		for row in test:
			output = self.predict(summarize, row)
			predictions.append(output)
		return predictions
```

### Driver Routine

Below is the main driver routine which creates a Naivebayes object that will import the dataset, then passes the dataset as well as the trigger decision tree routine over to the evaluation class to initiate the algorithm and measure its performance. In return, the Evaluate object will provide the performance score results from this performance runtime, which we print in a nice way and provide their representative overall average score.


```python

def main():
	print("\n" + "="*70)
	print(" Continuous Naive Bayes Algrithm")
	print("="*70 + "\n")
	random.seed(1)
	NAIVEBAYES = Naivebayes("iris.data")
	EVALUATE = Evaluate(NAIVEBAYES.dataset, NAIVEBAYES.naive_bayes)
	EVALUATE.display()
```

### Output

    
    ======================================================================
     Continuous Naive Bayes Algrithm
    ======================================================================
    
     [+] Evaluation Metric Scores: 
    
    +---------+----------+-------+-----------+--------+
    |  Fold # | Accuracy |   F1  | Precision | Recall |
    +---------+----------+-------+-----------+--------+
    |    1    |  0.981   | 0.982 |   0.983   | 0.981  |
    |    2    |  0.934   | 0.933 |   0.932   | 0.936  |
    |    3    |  0.962   |  0.96 |   0.963   | 0.959  |
    |    4    |  0.943   | 0.939 |   0.939   | 0.939  |
    |    5    |  0.972   |  0.97 |   0.971   | 0.969  |
    |    6    |  0.943   | 0.944 |   0.947   | 0.942  |
    |    7    |  0.943   | 0.946 |   0.949   | 0.946  |
    | AVERAGE |  0.954   | 0.953 |   0.955   | 0.953  |
    +---------+----------+-------+-----------+--------+




## **Implementation (Discrete)**

### Scripting Language

- Python 3

### Dataset

- We will use the Balance Scale Dataset (Discrete) and we will attempt to predict the ClassName given the measurements: Left-Weight, Left-Distance, Right-Weight, Right-Distance. The dataset contains 625 observations (49 balanced, 288 left, 288 right) with 4 input variables (Left-Weight, Left-Distance, Right-Weight, Right-Distance) and 1 output variable representing the class.

- Dataset Link: https://archive.ics.uci.edu/ml/datasets/Balance+Scale

- Dataset excerpt:

```
R,1,2,5,3
R,1,2,5,4
R,1,2,5,5
L,1,3,1,1
L,1,3,1,2
B,1,3,1,3
R,1,3,1,4
R,1,3,1,5
L,1,3,2,1
R,1,3,2,2
R,1,3,2,3
R,1,3,2,4
```

### Import Libraries

- We import the csv & pandas Python library to read in the dataset saved inside a CSV file
- We import the numpy Python library to utilize its vast libraries for creating and manipulating multi-dimensional arrays and matrices as well as high-level math functions to operate on these arrays
- We import the math Python library to utilize its exponential and square root functions
- We import the random Python library to generate random values
- We import the warnings Python library to suppress some annoying warnings
- We import the prettytable Python library to present the output results in a nice tabular layout
- We import the sklearn Python library to utilize its metric calculation functions


```python
import csv
import numpy
import pandas
import random
import warnings
import prettytable
import sklearn.metrics
import sklearn.model_selection
warnings.filterwarnings("ignore")
```

### Algorithm Evaluation

We will evaluate the algorithm via cross-validation by splitting the data into 7 folds (k=7) and computing the accuracy of predictions.

The general procedure for this evaluation is:
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:
        - Take group as a test dataset
        - Take remaining groups as a training dataset
        - Fit a model on the training dataset and evaluate it on the test dataset
        - Retain the evaluation accuracy score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores


```python
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
```

### Naive Bayes Algorithm Class

We implemented the algorithm in a modular form for better data management as well as better organization. In the initialization function of the class, we specify the required local variables (the model variable where we will save the trained model).


```python
class Naivebayes:

    def __init__(self):
        self.model = {}
```

### Train Model

In the below implemented routines from the overall NaiveBayes class, we train our model and save it inisde a class local variable for use along with test data inside the prediction function. For sake of clearness, we divided this model training process into several functions where each has its own job. As a first step in this routine, we summarize the classes that have been passed through the parameter variable "labels". Then we summarize  the features which are taken from the training dataset passed through "features" paramater. Lastly we calculate the probabilities. 


```python
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
```

### Make Prediction

Below implemented routine manages the calculation of the probabilities of a new row for each class, then selects the one with the largest probability


```python
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
```

### Driver Routine

Below is the main driver routine which creates a Naivebayes object that will import the dataset, then passes the dataset as well as the trigger decision tree routine over to the evaluation class to initiate the algorithm and measure its performance. In return, the Evaluate object will provide the performance score results from this performance runtime, which we print in a nice way and provide their representative overall average score.


```python
def main():
    print("\n" + "="*70)
    print(" Discrete Naive Bayes Algrithm")
    print("="*70 + "\n")
    random.seed(1)
    EVALUATE = Evaluate("balance-scale.csv")
    EVALUATE.display()
```

### Output

    
    ======================================================================
     Discrete Naive Bayes Algrithm
    ======================================================================
    
     [+] Evaluation Metric Scores: 
    
    +---------+----------+-------+-----------+--------+
    |  Fold # | Accuracy |   F1  | Precision | Recall |
    +---------+----------+-------+-----------+--------+
    |    1    |  0.852   | 0.841 |   0.832   | 0.861  |
    |    2    |  0.704   | 0.517 |   0.506   | 0.528  |
    |    3    |  0.444   | 0.447 |   0.706   | 0.617  |
    |    4    |  0.556   | 0.413 |    0.45   | 0.382  |
    |    5    |  0.593   | 0.557 |   0.591   | 0.576  |
    |    6    |  0.741   | 0.487 |   0.524   | 0.457  |
    |    7    |  0.556   | 0.336 |   0.329   | 0.343  |
    | AVERAGE |  0.635   | 0.514 |   0.563   | 0.538  |
    +---------+----------+-------+-----------+--------+



