import PyTorchDiscreteFlows.discrete_flows.disc_utils as disc_utils
from PyTorchDiscreteFlows.discrete_flows.made import MADE
from PyTorchDiscreteFlows.discrete_flows.mlp import MLP
from PyTorchDiscreteFlows.discrete_flows.embed import EmbeddingLayer
from PyTorchDiscreteFlows.discrete_flows.disc_models import *
from preprocessing import *
from flow_functions import *


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import itertools
import math
import random
import time
import os
import tensorflow_datasets as tfds

print(preprocess_binary_mnist())

def Train_MNIST(digit, disc_layer_type, batch_size, epoch, hidden_layer=0, temp_decay=0.83, lr_decay=0.97, path='',
                save=False, image_process=False, CNN=False, test_per_epoch=True, dim=None, sample_size=None, af=None,
                alpha=1, beta=1, id_init=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = Mai_create_X_train_test(preprocess_binary_mnist(), 4 / 5, 1, 1)
    print(train_data.shape)
    mnist = DATA(train_data)
    MNIST_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    sequence_length, vocab_size = 784, 2
    num_flows = 6  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper
    disc_layer_type = disc_layer_type  # 'autoreg' #'bipartite'
    batch_size = batch_size

    train_store_time = []
    train_store_min_loss = []

    # Training
    if id_init:
        train_data = torch.from_numpy(train_data)
        base_log_probs = init_prior(train_data, vocab_size, sequence_length)
        model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                                hidden_layer, CNN=CNN, af=af, alpha=alpha, beta=beta, id=True)
    else:
        base_log_probs = create_base_prior(sequence_length, vocab_size)
        model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                                hidden_layer, CNN=CNN, af=af, alpha=alpha, beta=beta, id=False)
    model = model.to(device)
    print(model)

    loss, final_time, load_path = train_disc_flow(device=device, model=model, data=MNIST_loader,
                                                  base_log_probs=base_log_probs, vocab_size=vocab_size,
                                                  test_per_epoch=test_per_epoch, temp_decay=temp_decay,
                                                  lr_decay=lr_decay, save_path=str(path) + 'MNIST',
                                                  k_fold=disc_layer_type, k_fold_idx=digit, batch_size=batch_size,
                                                  epochs=epoch, learning_rate=0.01, dataloader=True, CNN=CNN, dim=dim,
                                                  save=save, test_data=test_data, update_temp=False,
                                                  data_size=train_data.shape[0])
    train_store_min_loss.append(loss.cpu().clone().detach().numpy())
    train_store_time.append(final_time)

    print("Training Minimum Loss:")
    print(train_store_min_loss)
    print("Training Time:")
    print(train_store_time)

    if image_process:
        # Sample Pior
        prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
        base = prior.sample([sample_size]).to(device)
        # Inverse model
        model.eval()
        if CNN:
            base = base.view((base.shape[0], 4, 14, 14, vocab_size)).to(device)
            # base = F.one_hot(squeeze(torch.argmax(base, dim=-1), 4, 14, 14), -1)
        data = model.reverse(base)
        print(data.shape)
        # Removes one hot
        sample = torch.argmax(data, dim=-1)
        if CNN:
            sample = unsqueeze(sample, 1, 28, 28)
        print(sample.shape)

        sample = sample.cpu().detach().numpy()

        for i in range(sample_size):
            if i == 0:
                im = sample[i].reshape(28, 28)
            else:
                im = np.hstack((im, sample[i].reshape(28, 28)))

        np.save(str(path) + 'MNIST_' + str(disc_layer_type) + '_' + str(digit) + '.npy', im)
        plt.imshow(im)
        plt.show()