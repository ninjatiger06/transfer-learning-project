import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.applications.resnet50 as resnet50
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

	return model

def evaluate(model, train, validation, checkpointPath):
	cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointPath,
													save_weights_only = True,
													verbose = 1)

	history = model.fit(
		train,
		batch_size = 32,
		epochs = 117,
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

	with open("modelInfo.json", 'r') as modelInfo:
		pasHist = json.load(modelInfo)
	
	if pastHist is not None:
		pastHist["accuracy"].append(history.history["accuracy"])
		pastHist["loss"].append(history.history["loss"])
		pastHist["val_accuracy"].append(history.history["val_accuracy"])
		pastHist["val_loss"].append(history.history["val_loss"])

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

	train = train.map(lambda x, y: (resnet50.preprocess_input(x), y))
	validation = validation.map(lambda x, y: (resnet50.preprocess_input(x), y))

	checkpointPath = "C:/Users/Jonas/OneDrive/Desktop/2023-2024/Advanced-Honors-Comp-Sci/transfer-learning-project/checkpoints"
	checkpointDir = os.path.dirname(checkpointPath)

	os.listdir(checkpointDir)

	model = createModel()

	# print("\n\n\nBlank Model")
	# evaluate(model, train, validation, checkpointPath)
	checkpoint = Checkpoint(model)

	# print("\n\n\nLoaded Weights")
	checkpoint.restore(checkpointPath)
	model.load_weights(checkpointPath)
	history = evaluate(model, train, validation, checkpointPath)


main()