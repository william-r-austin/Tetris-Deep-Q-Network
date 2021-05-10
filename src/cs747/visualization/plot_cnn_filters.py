'''
Created on May 9, 2021

@author: William
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

model_path = "C:/Users/William/Documents/GMU Grad School/10 - Spring 2021 - CS 747/Semester Project/sample_model_2.tar"

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

if __name__ == '__main__':
    model = torch.load(model_path, map_location=torch.device('cpu'))
    #j = x.parameters()
    print("Loaded!!!")
    
    layer = 1
    #filter = model.features[layer].weight.data.clone()
    filter = model["model_state_dict"]["convolutionalLayer1.weight"]
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()
