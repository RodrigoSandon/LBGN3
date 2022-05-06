# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:02:59 2021

@author: Matthew Chen
"""

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import convModel
from torch import optim 
from time import time

import resNetCifar10Model
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import Occlusion
import torch.nn.functional as F
import torch.nn as nn
import os


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ("visal", "visam", "visl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_set = datasets.ImageFolder("AllRegionsImages/allregions_300_onlyaug/test", transform=transform)
valloader = torch.utils.data.DataLoader(val_set, batch_size=val_set.__len__(), shuffle=True)

model = resNetCifar10Model.ResNet18()
model.load_state_dict(torch.load("trained_model"))
model.eval()
model.to(device)

images, labels = next(iter(valloader))

for ind in range(len(images)):
    img = images[ind].to(device)
    img = img[None]
    labels = labels.to(device)
    
    
    input = img
    input.requires_grad = True
    input = input.to(device)
    
    
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
        
    grads = grads.view(3, 32, 32)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
        
  
    attr_ig = attr_ig.view(3, 32, 32)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))
        
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                              nt_samples=100, stdevs=0.2)
    attr_ig_nt = attr_ig_nt.view(3, 32, 32)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = attr_dl.view(3, 32, 32)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(input,
                                               strides = 2,
                                               target=labels[ind].item(),
                                               sliding_window_shapes= (3, 10, 10),
                                               baselines=0)
    attributions_occ = attributions_occ.view(3, 32, 32)
    attributions_occ = np.transpose(attributions_occ.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
        
    print('Original Image')
    print('Predicted:', classes[predicted[0]], 'Actual:', labels[ind].cpu(),
          ' Probability:', torch.max(F.softmax(output, 1)).item())
        
    original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        
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
    
    
    
    path = "analysisOnlyAug/" + str(ind)
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig1.savefig(path + "/OriginalImage.png")
    fig2.savefig(path + "/OverlayedGradientMagnitudes.png")
    fig3.savefig(path + "/OverlayedIntegratedGradients.png")
    fig4.savefig(path + "/OverlayedIntegratedGradientsWithSmoothGradSquared.png")
    fig5.savefig(path + "/OverlayedDeepLift.png")
    fig6.savefig(path + "/Occlusion.png")
    
    