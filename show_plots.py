import numpy as np
import matplotlib.pyplot as plt
import os

prefix = "D:\Data\wildcamdata"
DATASET = prefix
CROPED_DATA = os.path.join(DATASET, "orig_crop")

TRAIN_CROPED_DATA = os.path.join(CROPED_DATA, "cropped_images_train")
TEST_CROPED_DATA = os.path.join(CROPED_DATA, "cropped_images_test")
MODEL_PATH = "./resnet50_orig_size"

with open(os.path.join(MODEL_PATH, f"train_losses.npy"), 'rb') as f:
    train_losses = np.load(f)
with open(os.path.join(MODEL_PATH, f"train_accs.npy"), 'rb') as f:
    train_accs = np.load(f)
with open(os.path.join(MODEL_PATH, f"val_losses.npy"), 'rb') as f:
    val_losses = np.load(f)
with open(os.path.join(MODEL_PATH, f"val_accs.npy"), 'rb') as f:
    val_accs = np.load(f)


plt.figure()
plt.loglog()
plt.plot(train_losses)
plt.plot(val_losses)
plt.title("Resnet50 Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Validation Loss")
plt.legend(["Train loss", "Val loss"])
plt.show()

plt.figure()
plt.loglog()
plt.plot(train_accs)
plt.plot(val_accs)
plt.title("Resnet50 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Val"])
plt.figure()
plt.show()
