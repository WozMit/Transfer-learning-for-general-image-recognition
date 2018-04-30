import argparse
import time as tm;
import numpy as np;
from sklearn.linear_model import LogisticRegression;
from sklearn.svm import LinearSVC, SVC;
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.neural_network import MLPClassifier;
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score;
from sklearn.model_selection import GridSearchCV;
from sklearn.externals import joblib;


def svm_grid_search(dataset, labels):
	C_s = 10.0 ** np.arange(-1, 3);
	gammas = 10.0 ** np.arange(-1, 3);
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': gammas, 'C': C_s}];
	model = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=3);
	model.fit(dataset, labels);
	return model.best_params_['C'], model.best_params_['gamma'];

def linearSVM_grid_search(dataset, labels):
	C_s = 10.0 ** np.arange(-1, 3);
	tuned_parameters = [{'C': C_s}];
	model = GridSearchCV(svm.LinearSVC(C=1), tuned_parameters, cv=3);
	model.fit(dataset, labels);
	return model.best_params_['C'];

def J(v):
	v.sort();
	return v[-1] - np.mean(v[:-1]) - np.std(v[:-1]);

def Prop(v):
	v.sort();
	return v[-1] - v[-2];

start_time = tm.time();

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_train_file", type=str, help="Dataset train file name (*.npy)")
parser.add_argument("cls_train_file", type=str, help="Label train filename (*.npy)")
parser.add_argument("data_test_file", type=str, help="Dataset test file name (*.npy)")
parser.add_argument("cls_test_file", type=str, help="Label test filename (*.npy)")
parser.add_argument("model", type=str, help="Classifier", choices=['logistic', 'linear_svc', 'gaussian', 'bernoulli', 'multinomial', 'svc', 'knn', 'rf', 'nn'])
parser.add_argument("rejection", type=str, help="Classify with rejection", choices=['yes', 'no']);
parser.add_argument("output_filename", type=str, help="Predicted output filename");
args = parser.parse_args()

# Take all data
train_data = np.load(args.data_train_file);
train_labels = np.load(args.cls_train_file);
test_data = np.load(args.data_test_file);
test_labels = np.load(args.cls_test_file);
model_name = args.model;
rejection = (args.rejection == "yes");
if(model_name == "linear_svc" or model_name == "svc"): rejection = False;
output_filename = args.output_filename;
print("\nDataset read\n");

# Select model
if model_name == "logistic":
	model = LogisticRegression();
elif model_name == "linear_svc":
	#C = linearSVM_grid_search(train_data, train_labels);
	#C = 0.1;
	model = LinearSVC();
elif model_name == "gaussian":
	model = GaussianNB();
elif model_name == "bernoulli":
	model = BernoulliNB();
elif model_name == "multinomial":
	model = MultinomialNB();
elif model_name == "svc":
	#C, gamma = svm_grid_search(train_data, train_labels)
	#C, gamma = 10, 0.1;
	# Inception, Resnet -> SVC(C=30, gamma=0.0005, tol=0.1);
	model = SVC(C=40, gamma=0.00001);
elif model_name == "knn":
	model = KNeighborsClassifier(1);
elif model_name == "rf":
	model = RandomForestClassifier(max_depth=50, n_estimators=500);
else:
	model = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', alpha=0.1, max_iter=500);
	#model = LogisticRegression();

# Train model
model.fit(train_data, train_labels);
print("\nModel trained");

# Predict test data

predicted_labels = model.predict(test_data);

def print_accuracy(prediction):
	# Accuracy, recall, precision, f1
	accuracy = accuracy_score(test_labels, prediction);
	recall = recall_score(test_labels, prediction, average="macro");
	precision = precision_score(test_labels, prediction, average="macro");
	f1 = f1_score(test_labels, prediction, average="macro");
	print("\nOverall accuracy: %12.5f %%" %(accuracy*100));
	print("Overall recall: %12.2f %%" %(recall*100));
	#print("Overall precision: %9.3f %%" %(precision*100));
	print("Overall f1: %16.2f %%" %(f1*100));

if(rejection):
	pre = model.predict_proba(test_data);
	#good, bad = 0, 0;
	#for t in range(1000):
	#	if(test_labels[t] == predicted_labels[t]):
	#		good += J(pre[t]*100);
	#	else:
	#		bad += J(pre[t]*100);
	#print(good/1000, bad/1000);
	# inception + logistic -> 45
	# inception + multinomial -> 45
	# inception + rf -> 5
	technique_1 = np.copy(predicted_labels);
	lim = 45;
	for t in range(len(test_labels)):
		if(Prop(pre[t]*100) < lim): technique_1[t] = 14;
	print("\nPast proposed technique results:");
	print_accuracy(technique_1);

	technique_2 = np.copy(predicted_labels);
	lim = 45;
	for t in range(len(test_labels)):
		if(J(pre[t]*100) < lim): technique_2[t] = 14;
	print("\nNew proposed technique results:");
	print_accuracy(technique_2);
else:
	print_accuracy(predicted_labels);

#print("\nActual labels:");
#print(test_labels);
#print("\nPredicted labels:");
#print(predicted_labels);

# Save the model
joblib.dump(model, output_filename);
print("\nModel saved as", output_filename);

print("\nTotal time = "+str(tm.time()-start_time)+" s");