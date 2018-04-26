# Transfer learning for general image recognition
Different pre-trained CNN are used, all provided by the framework [Keras](https://github.com/fchollet/deep-learning-models) ([VGG16, VGG19](https://arxiv.org/abs/1409.1556), [Xception](https://arxiv.org/abs/1610.02357), [ResNet](https://arxiv.org/abs/1512.03385), [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261)) and [Magnus Erik Hvass Pedersen](https://github.com/Hvass-Labs/TensorFlow-Tutorials) ([Inception-v3](https://arxiv.org/abs/1512.00567)).

## Dataset
The [dataset](https://drive.google.com/drive/folders/0BzMfOBUQtl7dMHJfSGgtVTRZRDQ?usp=sharing) consists of 2000 images with 14 classes representing historical buildings of the city of Cusco-Perú. The class label, name and number of images is presented:

| Class label    | Building name   | Number of images|
| :---:         |     :---:       | :---:|
| 01    |  Casa del Inca Garcilaso de la Vega     |  108    |
| 02      | Catedral del Cusco       |159    |
| 03      | La Compañía de Jesús       |176    |
| 04    | Coricancha     |  147|
| 05      | Cristo Blanco       |146|
| 06      | Templo de la Merced       |142|
| 07    | Mural de la Historia Inca    |  137|
| 08    | Paccha de Pumaqchupan   |  114    |
| 09      |  Pileta de San Blas      |139    |
| 10      | Inca Pachacutec      |129    |
| 11    | Sacsayhuaman     |  190|
| 12      | Iglesia de San Francisco       |135|
| 13      |  Iglesia de San Pedro     |146|
| 14      |Iglesia de Santo Domingo    |  132|

## Requirements
* [Python](https://www.python.org/) 3.x
* [Numpy](http://www.numpy.org/) 1.14.2
* [Scipy](https://www.scipy.org/) 1.0.1
* [Scikit-learn](http://scikit-learn.org/stable/) 0.19.1
* [Matplotlib](https://matplotlib.org/) 2.2.2
* [Pillow](https://pillow.readthedocs.io/en/5.1.x/) 5.1.0
* [TensorFlow-GPU](https://www.tensorflow.org/) 1.7.0 / TensorFlow 1.5.1
* [Keras](https://keras.io/) 2.1.6

### VGG16, VGG19, Xception, ResNet, Inception-ResNet-v2
Pre-trained weights can be automatically loaded upon execution. Weights are automatically downloaded if necessary, and cached locally in ~/.keras/models/.

### Inception-v3
You must download the pre-trained model from [here](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) and unzip the model in a folder. Then modify the model's path in the `inception.py` file (line 73 `data_dir` variable).

## Compute transfer values
The main activity is to compute the transfer values wich are given by one of the previously saw CNN when provided several images. This information is saved as two `.npy` files: one containing the vector of features of the last layer of the CNN and the other containing the classes of the dataset. After that, these files can be used to train a new model for fitting the dataset.

As the training must be done with just the training set, it is necessary to compute the transfer values two times: one for the training set and one for the test set. Thus, the transfer values of a particular CNN model can be fully defined by four `.npy` files: `output_data_train.npy`, `output_cls_train.npy`, `output_data_test.npy`, `output_cls_test.npy`.

The command for computing transfer values:

`python compute_transfer_values.py < path > < type > < model > < data_augmentation > < features > < classes >`

Where
* < path > : Path to directory of input images for training or testing
* < type > : Dataset type (train, test)
* < model > : Model type (vgg16, vgg19, resnet, inception, exception, inceptresv2)
* < data_augmentation > : Data augmentation (yes, no)
* < features > : Name of the .npy file to be created containing the vector of features. E.g. `output_data_train.npy`
* < classes > : Name of the .npy file to be created containing the classes. E.g. `output_cls_train.npy`

## Classification
Once obtained the transfer values, our goal is to fit some model with the training data to accurately predict the test data.

The command for making classification:

`python classify.py < train_data > < train_labels > < test_data > < test_labels > < model > < rejection > < output_file >`

Where
* < train_data > : Name of the .npy file containing the training data. E.g. `output_data_train.npy`
* < train_labels > : Name of the .npy file containing the training labels. E.g. `output_cls_train.npy`
* < test_data > : Name of the .npy file containing the test data. E.g. `output_data_test.npy`
* < test_labels > : Name of the .npy file containing the test labels. E.g. `output_cls_test.npy`
* < model > : Model to predict the data (svm, knn, rf)
* < rejection > : Use rejection for prediction (yes, no)
* < output_file > : Name of the file to be created containing the output of the process