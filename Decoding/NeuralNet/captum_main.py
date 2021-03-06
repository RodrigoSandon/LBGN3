import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from time import time

import resNetCifar10Model
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from torch.utils.data import Dataset
from captum.attr import Occlusion
import torch.nn.functional as F
import torch.nn as nn
import os

class FeatureDataset(Dataset):
  
  def __init__(self, file_name):
    self.x_train = []
    self.y_train = []
    self.X_train = []
    self.Y_train = []

    count = 0
    
    name = {
        'Large': 0,
        'Small': 1,
    }
    """for root,dirs,files in os.walk(file_name):
      print(dirs)
      y = name[dirs]
      for name in files:"""
    #read csv file and load row data into variables
    for subdir in os.listdir(file_name):
      for fname in os.listdir(os.path.join(file_name, subdir)):
        full_path = os.path.join(file_name, subdir, fname)
        y = name[subdir]
        #filename = os.path.join(root,name)
        file_out = pd.read_csv(full_path, header = None) #if don't say header is none, first row will be used as header
        file_out = file_out.iloc[:,1:]
        x = file_out.iloc[0:len(file_out), 0:len(file_out)].values
        x = x[1:].astype(np.float32) # ommitting the column headers (cell names)
        #print(x[0])
        self.x_train.append(x)
        #print("x_train:", x_train)
        # y = file_out.iloc[0:32, 0:32].values
        self.y_train.append(y)
        #Converting to tensors
        X = torch.tensor(x, dtype=torch.float32) #converting to tensors (used to be self.X_train)
        self.X_train.append(X)
        #print("X_train size: ",X_train.shape)
        Y = torch.tensor(y)
        #print("Y_train size: ",Y_train.shape)
        self.Y_train.append(Y)
        count += 1
        #print(count)

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.X_train[idx], self.Y_train[idx]


"""transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])"""
print(torch.cuda.memory_summary(device=None, abbreviated=False))

classes = ("Large", "Small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_set = FeatureDataset("/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Input_Datasets/Neural_Net_2/RDT D1/Reward Size_Choice Time (s)/test")
valloader = torch.utils.data.DataLoader(val_set, batch_size=1)

model = resNetCifar10Model.ResNet18()
model.load_state_dict(torch.load("trained_model_1.pt"))
model.eval()
model.to(device)

images, labels = next(iter(valloader))

for ind in range(len(images)):
    img = images[ind].to(device)
    img = img[None]
    labels = labels.to(device)
    
    
    input = img[:,None,:,:]
    input.requires_grad = True
    input = input.to(device)
    
    print(type(img))
    img = img[:,None,:,:]
    print(img.size())
    img_shape = list(img.size())
    output = model(img)
        
    _, predicted = torch.max(output, 1)
        
        
    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                     target=labels[ind],
                                                     **kwargs
                                                    )
            
        return tensor_attributions     
        
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=labels[ind].item())
        
    grads = grads.view(1, img_shape[2], img_shape[3])
    grads = np.transpose(grads.squeeze().cpu().detach().numpy())
        
        
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
        
  
    attr_ig = attr_ig.view(1, img_shape[2], img_shape[3])
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy())
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print('Approximation delta: ', abs(delta))
        
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                              nt_samples=100, stdevs=0.2)
    attr_ig_nt = attr_ig_nt.view(1, img_shape[2], img_shape[3])
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy())
        
    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = attr_dl.view(1, img_shape[2], img_shape[3])
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy())
        
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(input,
                                               strides = 2,
                                               target=labels[ind].item(),
                                               sliding_window_shapes= (3, 10, 10),
                                               baselines=0)
    attributions_occ = attributions_occ.view(1, img_shape[2], img_shape[3])
    attributions_occ = np.transpose(attributions_occ.squeeze(0).cpu().detach().numpy())
        

    print('Original Image')
    print('Predicted:', classes[predicted[0]], 'Actual:', labels[ind].cpu(),
          ' Probability:', torch.max(F.softmax(output, 1)).item())
        
    original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5)
        
    fig1, _ = viz.visualize_image_attr(None, original_image, 
                              method="original_image", title="Original Image, Actual: " + str(labels[ind].cpu()) + " Predicted: " + str(classes[predicted[0]]))
    
        
    fig2, _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                  show_colorbar=True, title="Overlayed Gradient Magnitudes (Saliency)")
        
    fig3, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                                  show_colorbar=True, title="Overlayed Integrated Gradients")
        
    fig4, _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
                                     outlier_perc=10, show_colorbar=True, 
                                     title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
        
    fig5, _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                                  title="Overlayed DeepLift")
        
    fig6, _  = viz.visualize_image_attr(attributions_occ,
                                              original_image,
                                              method="blended_heat_map",
                                              title="occlusion",
                                              sign="positive",
                                              show_colorbar=True,
                                              outlier_perc=2,
                                             )
    
    
    
    path = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Captum/results_1/" + str(ind)
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig1.savefig(path + "/OriginalImage.png")
    fig2.savefig(path + "/OverlayedGradientMagnitudes.png")
    fig3.savefig(path + "/OverlayedIntegratedGradients.png")
    fig4.savefig(path + "/OverlayedIntegratedGradientsWithSmoothGradSquared.png")
    fig5.savefig(path + "/OverlayedDeepLift.png")
    fig6.savefig(path + "/Occlusion.png")
    
"""
Created on Sat Jan 23 10:02:59 2021

@author: Matthew Chen
"""