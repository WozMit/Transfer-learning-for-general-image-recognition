import argparse
import time as tm;
import numpy as np;
from sklearn import svm;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import precision_score;
from sklearn.metrics import f1_score;
from sklearn.externals import joblib


def J(v):
	v = [3, 7, 10, 80];
	v.append(100 - np.sum(v));
	v.sort();
	return v[-1] - np.mean(v[:-1]) - np.std(v[:-1]);

x = [[0, 0], [1, 1]];
y = [0, 1];
#model = svm.SVC(probability=True);
#model.fit(x, y);
#pre = model.predict_proba([[1, 1]]);
#print(pre);

start_time = tm.time();

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_train_file", type=str, help="Dataset train file name (*.npy)")
parser.add_argument("cls_train_file", type=str, help="Label train filename (*.npy)")
parser.add_argument("data_test_file", type=str, help="Dataset test file name (*.npy)")
parser.add_argument("cls_test_file", type=str, help="Label test filename (*.npy)")
parser.add_argument("model", type=str, help="Classifier", choices=['svm', 'linear_svm', 'rf', 'knn'])
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
output_filename = args.output_filename;

# Select model
model = None;
if model_name == "svm":
	model = svm.SVC();
elif model_name == "linear_svm":
	model = svm.LinearSVC();

# Train model
model.fit(train_data, train_labels);
print("\nModel trained");

# Predict test data
predicted_labels = model.predict(test_data);

# Accuracy
accuracy = accuracy_score(test_labels, predicted_labels);
print("\nOverall accuracy: %.3f %%" %(accuracy*100));
recall = recall_score(test_labels, predicted_labels, average="macro");
print("Overall recall: %.3f %%" %(recall*100));
#precision = precision_score(test_labels, predicted_labels, average="macro")
#print("Overall precision: %.3f %%" %(precision*100));
#f1 = f1_score(test_labels, predicted_labels, average="macro");
#print("Overall f1: %.3f %%" %(f1*100));

# Save the model
joblib.dump(model, output_filename);
print("\nModel saved as", output_filename);

print("\nTotal time = "+str(tm.time()-start_time)+" s");