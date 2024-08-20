import os
import dask
import cmocean
import statsmodels
import numpy as np
import xarray as xr
import pandas as pd
from xrpatcher import XRDAPatcher
import colorcet as cc
#%matplotlib inline
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from scipy import integrate
import matplotlib
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy import stats
import matplotlib.pyplot as plt
import typing as tp
from dataclasses import dataclass
import itertools
import numpy as np
import xarray as xr
from tqdm import tqdm
from matplotlib import colors
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import torch.nn as nn
import functools
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from pickle import dump, load
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
import torch
from torch.cuda.amp import autocast, GradScaler
from eof import deseason, eof_chl, proj_chl
from models_DPM import ScoreNet, marginal_prob_std, diffusion_coeff
from typing import List, Dict, Tuple, NamedTuple, Optional, Iterable, OrderedDict
import xarray as xr
import numpy as np
from functools import reduce
#from models_new import LpLoss
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
import dask
import colorcet as cc
import cmocean




def prior_likelihood(z, sigma):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x, y, score_model, marginal_prob_std, 
                   diffusion_coeff,
                   batch_size=64, 
                   device='cuda',
                   eps=1e-5):
    epsilon = torch.randn_like(y)
    def divergence_eval(sample, time_steps, epsilon):
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    
  
    shape = x.shape
    shape1 = y.shape
    def score_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
    def divergence_eval_wrapper(sample, time_steps):
        with torch.no_grad():
      # Obtain x(t) by solving the probability flow ODE.
          sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
          time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
      # Compute likelihood.
          div = divergence_eval(sample, time_steps, epsilon)
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)
  
    def ode_func(t, y):
        time_steps = np.ones((shape[0],)) * t    
        sample = y[:-shape[0]]
        logp = y[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = -0.5 * g**2 * score_eval_wrapper(x, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(x, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)
    init = np.concatenate([y.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
  # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape1)
    delta_logp = zp[-shape[0]:].reshape(shape1[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return z, bpd
def loss_fn2(model, x, mask, marginal_prob_std, eps=1e-5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_t = (torch.rand(x.shape[0]) * (1. - eps) + eps).to(device) 
    z = torch.randn_like(x).to(device)
    std = marginal_prob_std(random_t).to(device)
    perturbed_x = (x + z * std[:, None, None, None]).to(device) #####perturbation
    #combined_perturbed_x =  torch.cat((x,y),axis=1)
    score = model(perturbed_x, random_t)
    #score = torch.masked_select(score, mask)
    print(score.shape)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss



def train_network(model,dataloaders, criterion, optimizer ,marginal_prob_std_fn, n_epochs, path_log, time_step,num_timesteps):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float)
    train_losses_save, valid_losses_save = [], [] 
    valid_loss_min = np.Inf # track change in validation loss
    ##############Params
    
    #marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25)
    #diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=25)
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        num_items = 0
        model.train()
        for data, target in dataloaders['train']:
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.float)
        
            optimizer.zero_grad()
            data1=data
            target1=target
            target1=target1.nan_to_num()
            mask = np.isnan(target1.cpu())
            mask = mask.bool()
            mask = ~mask
            mask = mask.to(device)
                
            loss = loss_fn(model, target1, data1, mask, marginal_prob_std_fn)
            loss += loss.item() * target1.shape[0]
            num_items += data1.shape[0]
   
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
               
                    
        model.eval()
        for data, target in dataloaders['val']:
            data, target = data.to(device,dtype=torch.float), target.to(device,dtype=torch.float)
            
            #prevtarget= target.dropna()
            prevtarget=target[:,0,:,:,:]
            prevtarget=prevtarget.nan_to_num()
 
            
            data1=data[:,i,:,:,:]
            target1=target[:,i,:,:,:]
            target1=target1.nan_to_num()
   
            mask = np.isnan(target1.cpu())
            mask = mask.bool()
            mask = ~mask
            mask = mask.to(device)
                
                
            loss = loss_fn(model, target1, data1, mask, marginal_prob_std_fn)
            loss += loss.item() * target1.shape[0]
            num_items += data1.shape[0]   
             
            valid_loss += loss.item()
        
        # calculate average losses and store
        valid_loss = valid_loss/(len(dataloaders['val'].sampler))
            
        train_loss = train_loss/(len(dataloaders['train'].sampler))
        train_losses_save.append(train_loss)
        valid_losses_save.append(valid_loss)

        # print training/validation statistics 
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, n_epochs, train_loss, valid_loss))

    # Plot the loss curve
    fig = plt.subplots(1, 1, figsize=(5, 5),sharey = True, tight_layout=True)
    plt.plot(valid_losses_save, label='Validation loss',color='orange')
    plt.plot(train_losses_save, label='Training loss',color='blue', linestyle='dashed')
    plt.legend(frameon=False)
    plt.savefig(path_log +'Loss', dpi= 100,bbox_inches = "tight")
    
    # Save loss values
    np.save(path_log +'train_loss.npy',train_losses_save)
    np.save(path_log +'valid_loss.npy',valid_losses_save)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model.'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), path_log + 'model.pt')   
        valid_loss_min = valid_loss




def train_val_dataset(X_scaled, Y_scaled, time, val_split, batch_size, log, time_step):
    """
    Split between train and validation function (20% of validation data by default)
    
    Parameters
    ----------
    X : input data scaled
    Y : output data scaled
    time : vector of time to use for the plot of train/validation distribution
    val_split : percentage of data to take in the validation split
    batch_size : size of the batch
    log : path of the working directory to save plots and files
    
    Returns
    -------
    dataloaders : data object containing a split "train" and "val"
    """ 
    transform = transforms.Compose([transforms.ToTensor()])
    dataloaders=[]
    #for i in range(0,6):
    dataset = Dataset_XY(X = X_scaled, Y = Y_scaled, time_step=time_step, transform=transform)
    print(len(dataset))
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    dataloaders = {x:DataLoader(datasets[x],batch_size, shuffle=True, num_workers=1) for x in ['train','val']}
    next(iter(dataloaders['train']))
    x,y = next(iter(dataloaders['train']))

        #dataloaders.append(dataloaderss)
    
    #print('lenghth', len(dataloaders['train']))

    #Plot the train and validation separation in time
    time_train = time[train_idx]
    time_valid = time[val_idx]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5),sharey = True, tight_layout=True)
    ax[0].hist([time_train,time_valid],label = ['train (' + str(len(time_train)) + ')','val (' + str(len(time_valid)) + ')'])
    ax[0].legend()
    ax[1].hist([time_train.dt.month,time_valid.dt.month],bins = 12)
    plt.savefig(log + 'Train_valid_time', dpi= 100,bbox_inches = "tight")
    plt.close()

    return dataloaders



class Dataset_XY(Dataset):
    
    def __init__(self, X, Y, time_step, transform=None):
        """
        Args:
            root_folder (String): path to input and output files
            input_file (String): npy array of the input data to be used by the NN
            output_file (String): npy array of the output data to be use by the NN
            transform (callable, Optional): Optional transform to be applied on
            a sample
            The size of input are by default : time x lat x lon x variable
            The size of output are by default : time x lat x lon
        """
        self.input_arr = X
        self.output_arr = Y
        self.transform = transform
        self.time_step =  time_step
        
    def __len__(self):
        return self.input_arr.shape[0]
    
    def __getitem__(self, idx):
        X = self.input_arr[idx,...]
        fig = plt.figure()
        plt.imshow(X[:,:,0,0])
        plt.title('INPUT')
        plt.show()
        fig.savefig('X without view.png')
      
        X = X.reshape(*X.shape[:-2], -1)
        #print(X.shape)
        Y = self.output_arr[idx,...]
        Y = Y.reshape(*Y.shape[:-2], -1)
        if self.transform:
            X =  self.transform(X)
            Y =  self.transform(Y)
            X = X.view(self.time_step, 8, X.shape[1], X.shape[2])
            fig = plt.figure()
            plt.imshow(X[0,0,:,:])
            plt.title('INPUT')
            plt.show()
            fig.savefig('X after view.png')
      
            Y = Y.view(self.time_step, 1, Y.shape[1], Y.shape[2])
        
        return X,Y
    
class Dataset_X(Dataset):
    def __init__(self, X , transform=None):
        """
        Args:
            root_folder (String): path to input and output files
            input_file (String): npy array of the input data to be used by the NN
            output_file (String): npy array of the output data to be use by the NN
            transform (callable, Optional): Optional transform to be applied on
            a sample
            The size of input are by default : time x lat x lon x variable
            The size of output are by default : time x lat x lon
        """
        self.input_arr = X
        self.transform = transform
        
    def __len__(self):
        return self.input_arr.shape[0]
    
    def __getitem__(self, idx):
        X = self.input_arr[idx,...]
        
        if self.transform:
            X =  self.transform(X)
        
        return X