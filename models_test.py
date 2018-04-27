from vgg16 import VGG16
from vgg19 import VGG19
from resnet50 import ResNet50
from xception_model import Xception
from inception_resnet_v2 import InceptionResNetV2
from keras import backend as K

from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np

# ********************************** Transfer Values ************************************
def vgg16_transfer_values(image):
	base_model = VGG16(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	#img_path = '01_0016.jpg'
	#img = image.load_img(img_path, target_size=(224, 224))
	#x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#x = preprocess_input(x)
	image = image.reshape((1, 224, 224, 3))
	transferva = model.predict(image)
	nsamples, nx =  transferva.shape
	d2_train_dataset = transferva.reshape((nsamples * nx))
	K.clear_session()
	return d2_train_dataset

def vgg19_transfer_values(image):
	base_model = VGG19(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	#img_path = '01_0016.jpg'
	#img = image.load_img(img_path, target_size=(224, 224))
	#x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#x = preprocess_input(x)
	image = image.reshape((1, 224, 224, 3))
	transferva = model.predict(image)
	nsamples, nx =  transferva.shape
	d2_train_dataset = transferva.reshape((nsamples * nx))
	K.clear_session()
	return d2_train_dataset

def resnet50_transfer_values(image):
	base_model = ResNet50(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	#img_path = '01_0016.jpg'
	#img = image.load_img(img_path, target_size=(224, 224))
	#x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#x = preprocess_input(x)
	image = image.reshape((1, 224, 224, 3))
	transferva = model.predict(image)
	nsamples, nx, ny , npoints =  transferva.shape
	d2_train_dataset = transferva.reshape((npoints * nsamples * nx * ny))
	K.clear_session()
	return d2_train_dataset

def xception_transfer_values(image):
	base_model = Xception()
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	#img = image.load_img(img_path, target_size=(224, 224))
	#x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#image = preprocess_input(image)
	image = image.reshape((1, 299, 299, 3))
	transferva = model.predict(image)
	nsamples, npoints =  transferva.shape
	d2_train_dataset = transferva.reshape((npoints * nsamples))
	K.clear_session()
	return d2_train_dataset

def inception_resnet_v2_transfer_values(image):
	base_model = InceptionResNetV2(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	#img = image.load_img(img_path, target_size=(224, 224))
	#x = image.img_to_array(img)
	#x = np.expand_dims(x, axis=0)
	#image = preprocess_input(image)
	image = image.reshape((1, 299, 299, 3))
	transferva = model.predict(image)
	nsamples, npoints =  transferva.shape
	d2_train_dataset = transferva.reshape((npoints * nsamples))
	K.clear_session()
	return d2_train_dataset
	
# ******************************** Tansfer Len************************************
def vgg16_transfer_len():
	base_model = VGG16(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	img_path = '1.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	transferva = model.predict(x)
	nsamples, nx =  transferva.shape
	d2_train_dataset = transferva.reshape((nsamples * nx))
	K.clear_session()
	return d2_train_dataset.shape[0]

def vgg19_transfer_len():
	base_model = VGG19(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	img_path = '1.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	transferva = model.predict(x)
	nsamples, nx =  transferva.shape
	d2_train_dataset = transferva.reshape((nsamples * nx))
	K.clear_session()
	return d2_train_dataset.shape[0]

def resnet50_transfer_len():
	base_model = ResNet50(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	img_path = '1.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	transferva = model.predict(x)
	nsamples, nx, ny , npoints =  transferva.shape
	d2_train_dataset = transferva.reshape((npoints * nsamples * nx * ny))
	K.clear_session()
	return d2_train_dataset.shape[0]

def xception_transfer_len():
	base_model = Xception()
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	img_path = '1.jpg'
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	block4_pool_features = model.predict(x)
	block = block4_pool_features.shape[1]
	K.clear_session()
	return block

def inception_resnet_v2_transfer_len():
	base_model = InceptionResNetV2(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	img_path = '1.jpg'
	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	block4_pool_features = model.predict(x)
	block = block4_pool_features.shape[1]
	K.clear_session()
	return block