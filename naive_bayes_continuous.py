import csv
import math
import random
import prettytable
import sklearn.metrics



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


class Naivebayes:

	def __init__(self, filename):
		self.dataset = list()
		self.filename = filename
		self.ST = Statistics()
		self.import_dataset()


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


	def separate_by_class(self, dataset):
		separated = dict()
		for i in range(len(dataset)):
			vector = dataset[i]
			class_value = vector[-1]
			if class_value not in separated:
				separated[class_value] = list()
			separated[class_value].append(vector)
		return separated


	def summarize_dataset(self, dataset):
		summaries = [(self.ST.mean(column), self.ST.standard_deviation(column), len(column)) for column in zip(*dataset)]
		del(summaries[-1])
		return summaries


	def summarize_by_class(self, dataset):
		separated = self.separate_by_class(dataset)
		summaries = dict()
		for class_value, rows in separated.items():
			summaries[class_value] = self.summarize_dataset(rows)
		return summaries


	def predict(self, summaries, row):
		probabilities = self.ST.class_probabilities(summaries, row)
		best_label, best_prob = None, -1
		for class_value, probability in probabilities.items():
			if best_label is None or probability > best_prob:
				best_prob = probability
				best_label = class_value
		return best_label


	def naive_bayes(self, train, test):
		summarize = self.summarize_by_class(train)
		predictions = list()
		for row in test:
			output = self.predict(summarize, row)
			predictions.append(output)
		return predictions


def main():
	print("\n" + "="*70)
	print(" Continuous Naive Bayes Algrithm")
	print("="*70 + "\n")
	random.seed(1)
	NAIVEBAYES = Naivebayes("iris.data")
	EVALUATE = Evaluate(NAIVEBAYES.dataset, NAIVEBAYES.naive_bayes)
	EVALUATE.display()


main()





