# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:02:42 2015
@author: priyanka

This code adds random noise to image mrf_sample.png and the performs de-noising using Markov Random Field
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import scipy

def read_image(name):
    image = scipy.misc.imread(name,flatten=True)
    image = scipy.misc.imresize(image,(140,120),'bilinear')
    image = np.where(image > 175, 1, -1)
    return image
    
def show_image(image):
    image = np.where(image > 0, 255, 0)
    plt.imshow(image,cmap=cm.gray)
    plt.show() 
    
def add_noise(image, noise_percent=0.10):
    rows=image.shape[0]    
    cols=image.shape[1]    
    area=rows*cols
    loc_flip=np.random.randint(0,area,int(area*noise_percent))
    for ii in loc_flip:
        locr=int(ii/cols)
        locc=(ii%cols)
        image[locr,locc]=-1*image[locr,locc]
    return image

def ICM(noisy_image):
    h=-1
    beta=1.0
    eta=2.1
    image=noisy_image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            e_plus = compute_energy(noisy_image,image,row,col,1,h,beta,eta) 
            e_minus = compute_energy(noisy_image,image,row,col,-1,h,beta,eta)
            if e_plus < e_minus:
                image[row,col]=1
            else:
                image[row,col]=-1
    return image
        
def compute_energy(noisy_image,image,row,col,label,h,beta,eta):
    energy=h*label
    if row > 0:
        energy -= beta*label*image[row-1,col]
    if col > 0:
        energy -= beta*label*image[row,col-1]
    if col < image.shape[1]-1:
        energy -= beta*label*image[row,col+1]
    if row < image.shape[0]-1:
        energy -= beta*label*image[row+1,col]
    energy -= eta*label*noisy_image[row,col]
    return energy

    
if __name__=="__main__":
    name="../image/mrf_sample.png"
    image=read_image(name)
    show_image(image)
    noisy_image=add_noise(image,noise_percent=0.10)
    show_image(noisy_image)
    denoised_image=ICM(noisy_image)
    show_image(denoised_image)
    
