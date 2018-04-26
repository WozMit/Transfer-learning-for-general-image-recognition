import argparse
import numpy as np;

v = [3, 7, 10, 80];
v.append(100 - np.sum(v));

v.sort();
print(v[-1] - np.mean(v[:-1]) - np.std(v[:-1]));

from sklearn import svm;
x = [[0, 0], [1, 1]];
y = [0, 1];
#model = svm.SVC(probability=True);
#model.fit(x, y);
#pre = model.predict_proba([[1, 1]]);
#print(pre);

parser = argparse.ArgumentParser()
parser.add_argument("data_train_file", type=str, help="Dataset train file name (*.npy)")
parser.add_argument("cls_train_file", type=str, help="Label train filename (*.npy)")
parser.add_argument("data_test_file", type=str, help="Dataset test file name (*.npy)")
parser.add_argument("cls_test_file", type=str, help="Label test filename (*.npy)")
parser.add_argument("model", type=str, help="Classifier", choices=['svm', 'linear_svm', 'rf', 'knn'])
parser.add_argument("rejection", type=str, help="Classify with rejection", choices=['yes', 'no']);
parser.add_argument("output_filename", type=str, help="Predicted output filename");
args = parser.parse_args()

train_data = np.load(args.data_train_file);
train_labels = np.load(args.cls_train_file);
test_data = np.load(args.data_test_file);
test_labels = np.load(args.cls_test_file);
model_name = args.model;
rejection = args.rejection;
output_filename = args.output_filename;