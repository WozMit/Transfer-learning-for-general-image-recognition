# Transfer learning for general image recognition

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

## Compute transfer values
Different pre-trained CNN are used, all provided by the framework [Keras](https://github.com/fchollet/deep-learning-models) ([VGG16, VGG19](https://arxiv.org/abs/1409.1556), [Xception](https://arxiv.org/abs/1610.02357), [ResNet](https://arxiv.org/abs/1512.03385), [Inception-Resnet-v2](https://arxiv.org/abs/1602.07261)) and Magnus Erik Hvass Pedersen ([Inception V-3](https://arxiv.org/abs/1512.00567)).

### VGG16, VGG19, Xception, Resnet, Inception-Resnet-v2
Pre-trained weights can be automatically loaded upon execution. Weights are automatically downloaded if necessary, and cached locally in ~/.keras/models/.

### Inception-V3
You must download the pre-trained model from [here](download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) and unzip the model in a folder. Then modify the model's path in the ´inception.py´ file (line 73 ´data_dir´ variable).