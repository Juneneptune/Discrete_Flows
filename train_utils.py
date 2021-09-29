import PyTorchDiscreteFlows.discrete_flows.disc_utils as disc_utils
from PyTorchDiscreteFlows.discrete_flows.made import MADE
from PyTorchDiscreteFlows.discrete_flows.mlp import MLP
from PyTorchDiscreteFlows.discrete_flows.embed import EmbeddingLayer
from PyTorchDiscreteFlows.discrete_flows.disc_models import *
from preprocessing import *

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

def create_base_prior(sequence_length, vocab_size):
    # Pior Distribution
    base_log_probs = torch.tensor(torch.randn(sequence_length, vocab_size), requires_grad = True)
    base = torch.distributions.OneHotCategorical(logits = base_log_probs)
    return base_log_probs

def train_disc_flow(device, model, data, base_log_probs, vocab_size, save_path, k_fold=None, k_fold_idx=None,
                    batch_size=1024, epochs=1500, learning_rate=0.01, dataloader=False, save=True, CNN=False, dim=None):
    print_loss_every = epochs // 10
    if print_loss_every == 0:
        print_loss_every = 1
    total_time = 0
    min_loss = 1e10

    losses = []
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': learning_rate},
            {'params': base_log_probs, 'lr': learning_rate}
        ])

    model.train()
    for e in range(epochs):
        start = time.time()
        if dataloader:
            for x in data:
                x = x.to(device)
                if x.shape[0] < batch_size:
                    continue
                x = F.one_hot(x, num_classes=vocab_size).float()
                # print(x.shape)
                if CNN:
                    x = x.view((x.shape[0], -1, dim[0], dim[1], vocab_size))
                    # x = x.view((x.shape[0], x.shape[1], x.shape[2], -1))
                # print(x.shape)
                optimizer.zero_grad()
                zs = model.forward(x)

                if CNN:
                    zs = zs.view((zs.shape[0], -1, vocab_size))

                base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1).to(device)
                # print(zs.shape, base_log_probs_sm.shape)
                logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
                loss = -torch.sum(logprob) / batch_size

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                end = time.time()

                if loss < min_loss:
                    min_loss = loss
                    path_name = save_path + 'k' + str(k_fold) + '_' + str(k_fold_idx) + '.pt'
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'prior': base_log_probs,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, path_name)

                if e % print_loss_every == 0:
                    print('epoch:', e, 'loss:', loss.item(), 'min loss:', min_loss)
                total_time = total_time + end - start

        else:
            x = torch.from_numpy(sample_batch_size_data(data, batch_size))
            x.to(device)
            x = F.one_hot(x, num_classes=vocab_size).float()

            if CNN:
                x.view((x.shape[0], -1, dim[0], dim[1]))

            optimizer.zero_grad()
            zs = model.forward(x)

            base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
            # print(zs.shape, base_log_probs_sm.shape)
            logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
            loss = -torch.sum(logprob) / batch_size

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            end = time.time()

            if loss < min_loss:
                min_loss = loss
                path_name = save_path + 'k' + str(k_fold) + '_' + str(k_fold_idx) + '.pt'
                if save:
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'prior': base_log_probs,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, path_name)

            if e % print_loss_every == 0:
                print('epoch:', e, 'loss:', loss.item(), 'min loss:', min_loss)
            total_time = total_time + end - start
    plt.plot(losses)
    plt.show()

    return min_loss.clone().detach(), total_time, path_name


def disc_flow_param(num_flows, temp, vocab_size, sequence_length, batch_size, disc_layer_type, hid_lay=0, CNN=False,
                    channel=1):
    '''
    batch_size, sequence_length, vocab_size = 1024, 2, 2

    num_flows = 1 # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1 # used for the straight-through gradient estimator. Value taken from the paper
    disc_layer_type = 'autoreg' #'autoreg' #'bipartite'

    # This setting was previously used for the MLP and MADE networks.
    nh = 64 # number of hidden units per layer
    vector_length = sequence_length*vocab_size
    '''

    flows = []
    for i in range(num_flows):
        if disc_layer_type == 'autoreg':

            # layer = EmbeddingLayer([batch_size, sequence_length, vocab_size], output_size=vocab_size)
            # MADE network is much more powerful.
            layer = MADE([batch_size, sequence_length, vocab_size], vocab_size, [hid_lay, hid_lay, hid_lay])

            disc_layer = DiscreteAutoregressiveFlow(layer, temp,
                                                    vocab_size)

        elif disc_layer_type == 'bipartite':
            # MLP will learn the factorized distribution and not perform well.
            # layer = MLP(vector_length//2, vector_length//2, nh)

            # layer = torch.nn.Embedding(vector_length//2, vector_length//2)
            if i % 2:
                dim = math.ceil(sequence_length / 2)
                dim_ = sequence_length - dim  # Dim of other half of the bipartite
                vector_length = dim * vocab_size
                vector_length_ = dim_ * vocab_size  # Vector length of other half of the bipartite
            else:
                dim = sequence_length // 2
                dim_ = sequence_length - dim  # Dim of other half of the bipartite
                vector_length = dim * vocab_size
                vector_length_ = dim_ * vocab_size  # Vector length of other half of the bipartite
            if CNN:
                layer = nn.Sequential(nn.Conv2d(channel, 8, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 8, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 2 * channel, 3, padding=1))
            else:
                layer = nn.Sequential(nn.Linear(vector_length, math.ceil(vector_length / 4)),
                                      nn.ReLU(),
                                      nn.Linear(math.ceil(vector_length / 4), math.ceil(vector_length / 4)),
                                      nn.ReLU(),
                                      nn.Linear(math.ceil(vector_length / 4), vector_length_ * 2)
                                      # vector_length * 2 allow for both loc and scale
                                      )
            disc_layer = DiscreteBipartiteFlow(layer, i % 2, temp,
                                               vocab_size, dim, isimage=CNN)
            # i%2 flips the parity of the masking. It splits the vector in half and alternates
            # each flow between changing the first half or the second.
        flows.append(disc_layer)

    model = DiscreteAutoFlowModel(flows)
    return model


def test_disc_flow(device, model, data, load_path, vocab_size):
    batch_size = data.shape[0]
    print(batch_size)
    open_path = torch.load(load_path)
    model.load_state_dict(open_path['model_state_dict'])
    base_log_probs = open_path['prior']

    model.eval()

    print(open_path['loss'])

    start = time.time()
    x = torch.from_numpy(data).to(device)
    x = F.one_hot(x, num_classes=vocab_size).float()

    zs = model.forward(x)

    base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1).to(device)
    # print(zs.shape, base_log_probs_sm.shape)
    logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
    loss = -torch.sum(logprob) / batch_size
    end = time.time()
    final_time = end - start
    return loss, final_time

def kfold_splitter(data,n,k,fold_num):
    fold_size = n//k
    test_data = data[fold_num*fold_size:(fold_num+1)*fold_size]
    train_data = np.vstack((data[0:fold_num*fold_size], data[(fold_num+1)*fold_size:n]))
    return train_data, test_data


def Train_Digit(digit, disc_layer_type, batch_size, epoch, hidden_layer=0, path='', save=False, image_process=False,
                sample_size=None):
    Digit = Digits(digit)
    digit_loader = torch.utils.data.DataLoader(dataset=Digit,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    sequence_length, vocab_size = 64, 2
    num_flows = 4  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper
    disc_layer_type = disc_layer_type  # 'autoreg' #'bipartite'
    batch_size = batch_size

    train_store_time = []
    train_store_min_loss = []

    # Training
    base_log_probs = create_base_prior(sequence_length, vocab_size)
    model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                            hidden_layer)

    loss, final_time, load_path = train_disc_flow(model=model, data=digit_loader, base_log_probs=base_log_probs,
                                                  vocab_size=vocab_size, save_path=str(path) + 'Digits',
                                                  k_fold=disc_layer_type, k_fold_idx=digit, batch_size=batch_size,
                                                  epochs=epoch, learning_rate=0.01, dataloader=True, save=save)
    train_store_min_loss.append(loss.clone().detach().numpy())
    train_store_time.append(final_time)

    print("Training Minimum Loss:")
    print(train_store_min_loss)
    print("Training Time:")
    print(train_store_time)

    if image_process:
        # Sample Pior
        prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
        base = prior.sample([sample_size])
        # Inverse model
        data = model.reverse(base)
        # Removes one hot
        sample = torch.argmax(data, dim=2)

        sample = sample.detach().numpy()

        for i in range(sample_size):
            if i == 0:
                im = sample[i].reshape(8, 8)
            else:
                im = np.hstack((im, sample[i].reshape(8, 8)))

        np.save(str(path) + 'Digits_' + str(disc_layer_type) + '_' + str(digit) + '.npy', im)
        plt.imshow(im)
        plt.show()


def Train_MNIST(digit, disc_layer_type, batch_size, epoch, hidden_layer=0, path='', save=False, image_process=False,
                CNN=False, dim=None, sample_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mnist = MNIST(digit)
    MNIST_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    sequence_length, vocab_size = 784, 2
    num_flows = 4  # number of flow steps. This is different to the number of layers used inside each flow
    temperature = 1  # used for the straight-through gradient estimator. Value taken from the paper
    disc_layer_type = disc_layer_type  # 'autoreg' #'bipartite'
    batch_size = batch_size

    train_store_time = []
    train_store_min_loss = []

    # Training
    base_log_probs = create_base_prior(sequence_length, vocab_size)
    model = disc_flow_param(num_flows, temperature, vocab_size, sequence_length, batch_size, disc_layer_type,
                            hidden_layer, CNN=CNN)

    model = model.to(device)
    print(model)

    loss, final_time, load_path = train_disc_flow(device=device, model=model, data=MNIST_loader,
                                                  base_log_probs=base_log_probs, vocab_size=vocab_size,
                                                  save_path=str(path) + 'MNIST', k_fold=disc_layer_type,
                                                  k_fold_idx=digit, batch_size=batch_size, epochs=epoch,
                                                  learning_rate=0.01, dataloader=True, CNN=CNN, dim=dim, save=save)
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
        if CNN:
            base = base.view((base.shape[0], -1, dim[0], dim[1], vocab_size)).to(device)
        data = model.reverse(base)
        print(data.shape)
        # Removes one hot
        sample = torch.argmax(data, dim=-1)
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