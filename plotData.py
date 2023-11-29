import json
import matplotlib.pyplot as plt

with open("modelInfo.json", 'r') as modelInfo:
    info = json.load(modelInfo)

plt.plot(range(1, len(info["loss"])+1), info["loss"], 'b', label="training loss")
plt.plot(range(1, len(info["val_loss"])+1), info["val_loss"], 'r', label="val_loss")
plt.show()

plt.plot(range(1, len(info["accuracy"])+1), info["accuracy"], 'b', label='training accuracy')
plt.plot(range(1, len(info["val_accuracy"])+1), info["val_accuracy"], 'r', label="val_accuracy")
plt.show()