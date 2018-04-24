import tensorflow as tf
from vgg16 import VGG16
from vgg19 import VGG19
from resnet50 import ResNet50
from xception import Xception
from keras import backend as K
from sklearn.externals import joblib

from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
import pickle
from PIL import Image
import models_test
from datetime import timedelta
from utils import plot_2D_images, new_fc_layer
import inception
from inception import transfer_values_cache
import os

def vgg16_transfer_len():
	base_model = VGG16(weights='imagenet')
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	img_path = '3.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	transferva = model.predict(x)
	nsamples, nx =  transferva.shape
	d2_train_dataset = transferva.reshape((1,nsamples * nx))
	K.clear_session()
	return d2_train_dataset

def inception_transfer_len():
	image_path = '12.jpg'	
	# Load the Inception model so it is ready for classifying images.
	model = inception.Inception()
	
	# Path for a jpeg-image that is included in the downloaded data.
	#image_path = os.path.join('validation/', x)

	# Use the Inception model to classify the image.
	pred = model.transfer_values(image_path=image_path)
	pred = np.reshape(pred,(1, 2048))
	return pred

def main():
	model_clone = joblib.load('logreg_joblib.pkl')
	feature_vector = vgg16_transfer_len()
	#feature_vector = inception_transfer_len()
	result = model_clone.predict(feature_vector)
	print(result)
	#prediction
	if [0]==result:
		print "Casa del Inca Garcilaso de la Vega"
	if [1]==result:
		print "La Catedral del Cusco"
	if [2]==result:
		print "La Companhia de Jesus"
	if [3]==result:
		print "Coricancha"
	if [4]==result:
		print "Cristo Blanco"
	if [5]==result:
		print "La Merced"
	if [6]==result:
		print "Mural de Historia Inca del Cusco"
	if [7]==result:
		print "La Paccha de Pumaqchupan"
	if [8]==result:
		print "Pileta de San Blas"
	if [9]==result:
		print "Monumento al Inca de Pachacutec"
	if [10]==result:
		print "Sacsayhuaman"
	if [11]==result:
		print "Templo de San Francisco"
	if [12]==result:
		print "Templo de San Pedro"
	if [13]==result:
		print "Templo de Santo Domingo"

if __name__ == "__main__":
	main()
