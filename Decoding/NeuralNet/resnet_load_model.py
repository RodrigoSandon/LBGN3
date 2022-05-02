import torch
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Confusion Matrix
from sklearn.metrics import confusion_matrix
import itertools
from torch.utils.data import Dataset
from ResNet18 import ResNet18, FeatureDataset

correct_count, all_count = 0, 0

correct = 0.00
total = 0.00

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

val_set = FeatureDataset("/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net_2/RDT D1/Reward Size_Choice Time (s)/val")
valloader = torch.utils.data.DataLoader(val_set, batch_size=1)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("cuda available")
else:
    device = torch.device('cpu')
    print("cpu")

model = ResNet18()
model.load_state_dict(torch.load("/media/rory/Padlock_DT/Rodrigo/Decoding/NeuralNet/trained_model_1.pt"))
#model.eval()
categories = ["Large", "Small"]

with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device)
        images = images.unsqueeze(1)
        #standard for image is 4 dimensional
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network test images: %d %%' % (
    100 * correct / total))

torch.save(model.state_dict(), "model1")
plt.show()

cm = confusion_matrix(y_true=categories, y_pred=np.argmax(correct/total, axis=-1))
#valloader.class_indices

plot_confusion_matrix(cm, classes=categories, title='Confusion Matrix')