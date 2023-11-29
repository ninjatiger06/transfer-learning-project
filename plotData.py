import json
import matplotlib.pyplot as plt

with open("modelInfo.json", 'r') as modelInfo:
    info = json.load(modelInfo)

plt.plot(range(1, len(info["loss"][0])+1), info["loss"][0], 'b', label="training loss")
plt.plot(range(1, len(info["val_loss"][0])+1), info["val_loss"][0], 'r', label="val_loss")
plt.show()

plt.plot(range(1, len(info["accuracy"][0])+1), info["accuracy"][0], 'b', label='training accuracy')
plt.plot(range(1, len(info["val_accuracy"][0])+1), info["val_accuracy"][0], 'r', label="val_accuracy")
plt.show()