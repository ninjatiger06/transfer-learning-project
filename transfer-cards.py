import tensorflow as tf
import tensorflow.keras.utils as utils
# import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
from tensorflow.train import Checkpoint
import json
import os

# def validateInput(prompt):
# 	while True:
# 		userInput = input(prompt)
# 		try:
# 			int(userInput)
# 			if userInput in [1, 2, 3]:
# 				return userInput
# 			else:
# 				print("Input must be an int between 1 and 3")
# 		except ValueError:
# 			print("Input must be an int between 1 and 3")

def createModel():
	netModel = vgg16.VGG16(
		include_top = True,
		weights = 'imagenet',
		classifier_activation = 'softmax',
	)

	print(f"netModel = {netModel}")

	netModel.trainable = False

	inputs = keras.Input(shape = (224, 224, 3))
	
	outputs = netModel(inputs)
	outputs = layers.Dense(53, activation = 'softmax')(outputs)

	optimizer = optimizers.legacy.Adam(learning_rate = 0.0001)
	loss = losses.CategoricalCrossentropy()

	model = keras.Model(inputs, outputs)

	model.compile(
		optimizer = optimizer,
		loss = loss,
		metrics = ['accuracy'],
	)

	return model

def evaluate(model, train, validation, checkpointPath):
	cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointPath,
													save_weights_only = True,
													verbose = 1)

	history = model.fit(
		train,
		batch_size = 32,
		epochs = 66,
		verbose = 1,
		validation_data = validation,
		validation_batch_size = 32,
	)

	model.save(checkpointPath)
	
	pastHist = {
		"accuracy": [],
		"loss": [],
		"val_accuracy": [],
		"val_loss": []
	}

	try:
		with open("modelInfo.json", 'r') as modelInfo:
			pastHist = json.load(modelInfo)
	except (FileNotFoundError, json.decoder.JSONDecodeError):
		pass
	
	if pastHist is not None:
		pastHist["accuracy"] += history.history["accuracy"]
		pastHist["loss"] += history.history["loss"]
		pastHist["val_accuracy"] += history.history["val_accuracy"]
		pastHist["val_loss"] += history.history["val_loss"]
	else:
		pastHist["accuracy"] = history.history["accuracy"]
		pastHist["loss"] = history.history["loss"]
		pastHist["val_accuracy"] = history.history["val_accuracy"]
		pastHist["val_loss"] = history.history["val_loss"]

	with open("modelInfo.json", 'w') as modelInfo:
		json.dump(pastHist, modelInfo, indent=4)

	return history

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

	train = train.map(lambda x, y: (vgg16.preprocess_input(x), y))
	validation = validation.map(lambda x, y: (vgg16.preprocess_input(x), y))

	checkpointPath = "C:/Users/Jonas/OneDrive/Desktop/2023-2024/Advanced-Honors-Comp-Sci/transfer-learning-project/checkpoints"
	checkpointDir = os.path.dirname(checkpointPath)

	os.listdir(checkpointDir)

	model = createModel()

	# print("\n\nBlank Model")
	# evaluate(model, train, validation, checkpointPath)
	checkpoint = Checkpoint(model)

	print("\n\n\nLoaded Weights")
	checkpoint.restore(checkpointPath)
	model.load_weights(checkpointPath)
	history = evaluate(model, train, validation, checkpointPath)


main()