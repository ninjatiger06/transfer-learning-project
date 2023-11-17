import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

def main():

	train = utils.image_dataset_from_directory(
		'train',
		label_mode = 'categorical',
		image_size = (224, 224)
	)

	validation = utils.image_dataset_from_directory(
		'valid',
		label_mode = 'categorical',
		image_size = (224, 224)
	)

	print(train)
	print(validation)

	train = train.map(lambda x, y: (resnet50.preprocess_input(x), y))
	validation = validation.map(lambda x, y: (resnet50.preprocess_input(x), y))

	resnet = resnet50.ResNet50(
		include_top = True,
		weights = 'imagenet',
		classifier_activation = 'softmax',
	)

	print(f"resnet = {resnet}")

	resnet.trainable = False

	inputs = keras.Input(shape = (224, 224, 3))
	
	outputs = resnet(inputs)
	outputs = layers.Dense(53, activation = 'softmax')(outputs)

	optimizer = optimizers.legacy.Adam(learning_rate = 0.00001)
	loss = losses.CategoricalCrossentropy()

	model = keras.Model(inputs, outputs)

	model.compile(
		optimizer = optimizer,
		loss = loss,
		metrics = ['accuracy'],
	)

	model.fit(
		train,
		batch_size = 32,
		epochs = 10,
		verbose = 1,
		validation_data = validation,
		validation_batch_size = 32,
	)



main()