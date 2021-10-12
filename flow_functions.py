import PyTorchDiscreteFlows.discrete_flows.disc_utils as disc_utils
from PyTorchDiscreteFlows.discrete_flows.made import MADE
from PyTorchDiscreteFlows.discrete_flows.mlp import MLP
from PyTorchDiscreteFlows.discrete_flows.embed import EmbeddingLayer
from PyTorchDiscreteFlows.discrete_flows.disc_models import *

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


# Squeeze functions
def squeeze(x, c, h, w):
    batch_size, c_, _, _ = x.size()
    factor = int((c / c_) ** 0.5)

    x = x.view(batch_size, c_, h, factor, w, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(batch_size, c, h, w)
    return x


def unsqueeze(x, c, h, w):
    batch_size, c_, h_, w_ = x.size()
    factor = int((c_ / c) ** 0.5)

    x = x.view(batch_size, c, factor, factor, h_, w_)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(batch_size, c, h, w)
    return x


def create_base_prior(sequence_length, vocab_size):
    # Pior Distribution
    base_log_probs = torch.tensor(torch.randn(sequence_length, vocab_size), requires_grad=True)
    base = torch.distributions.OneHotCategorical(logits=base_log_probs)
    return base_log_probs


def init_prior(data, vocab_size, sequence_length):
    data = F.one_hot(data, num_classes=vocab_size)
    count = data.sum(dim=0)
    log_prob = torch.clamp(count / data.shape[0], min=1e-30)
    log_prob = torch.log(log_prob)
    log_prob = torch.tensor(log_prob, requires_grad=True)
    log_prob = log_prob.type('torch.FloatTensor')
    return log_prob


def disc_flow_param(num_flows, temp, vocab_size, sequence_length, batch_size, disc_layer_type, hid_lay=0, CNN=False,
                    channel=2, af='relu', alpha=1, beta=1, id=True):
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
                layer = nn.Sequential(
                    Net(ch_in=channel, ch_out=8, lin_in=vector_length, lin_out=vector_length_, af=af, alpha=alpha,
                        beta=beta, vocab_size=0, id=id))
                '''
                layer = nn.Sequential(nn.Conv2d(channel, 8, 3, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(8, 8, 3, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(8, 2*channel, 3, padding=1))
                '''
            else:
                layer = nn.Sequential(
                    Net(ch_in=channel, ch_out=8, lin_in=vector_length, lin_out=vector_length_, af=af, alpha=alpha,
                        beta=beta, vocab_size=vocab_size, id=id))
            disc_layer = DiscreteBipartiteFlow(layer, i % 2, temp,
                                               vocab_size, dim, isimage=CNN)
            # i%2 flips the parity of the masking. It splits the vector in half and alternates
            # each flow between changing the first half or the second.
        flows.append(disc_layer)

    model = DiscreteAutoFlowModel(flows)
    return model


def train_disc_flow(device, model, data, base_log_probs, vocab_size, temp_decay, lr_decay, save_path, k_fold=None,
                    k_fold_idx=None, batch_size=1024, epochs=1500, learning_rate=0.01, dataloader=False, save=True,
                    CNN=False, dim=None, update_temp=False, test_per_epoch=True, test_data=None, temperature=0.1,
                    data_size=48000):
    torch.set_printoptions(edgeitems=50)
    print_loss_every = epochs // 100
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    model.train()
    # The number of batches in each epoch
    batch_num = data_size // batch_size
    start_bool = True
    epoch_test_loss = np.zeros((epochs + 1,))
    avg_epoch_train_loss = np.zeros((epochs + 1,))
    epoch_train_time = np.zeros((epochs + 1,))
    epoch_test_time = np.zeros((epochs + 1,))
    for e in range(epochs):
        # Updates temperature per epoch.
        if update_temp:
            temperature = temperature * temp_decay
            model.update_temperature(temperature)
            print(temperature)

        total_loss = 0

        """
        for something:
          print(time-time.time())
          ....
          print ()


        """

        if dataloader:
            for x in data:
                start_train = time.time()
                x = x.to(device)
                if x.shape[0] < batch_size:
                    continue
                if CNN:
                    ch = int(x.shape[1] / (dim[0] * dim[1]))
                    x = x.view(x.shape[0], 1, 28, 28)
                    x = squeeze(x, ch, dim[0], dim[1])
                x = F.one_hot(x, num_classes=vocab_size).float()
                # print(x.shape)
                if CNN:
                    x = x.view((x.shape[0], -1, dim[0], dim[1], vocab_size))
                    # x = x.view((x.shape[0], x.shape[1], x.shape[2], -1))
                # print(x.shape)
                optimizer.zero_grad()

                zs = model.forward(x)
                # zs = F.one_hot(unsqueeze(torch.argmax(zs, dim=-1), 1, 28, 28), num_classes=vocab_size)

                if CNN:
                    zs = zs.view((zs.shape[0], -1, vocab_size))

                base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1).to(device)
                # print(zs.shape, base_log_probs_sm.shape)
                logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.

                loss = -torch.sum(logprob) / batch_size

                if start_bool:
                    avg_epoch_train_loss[e] = loss.item()

                    if test_per_epoch:
                        model.eval()
                        with torch.no_grad():
                            start_test = time.time()
                            if test_data.shape[0] > 15000:
                                loss_test = 0
                                for i in range(math.ceil(test_data.shape[0] / 15000)):
                                    print(i)
                                    if i == test_data.shape[0] // 15000:
                                        x_test = torch.from_numpy(test_data[15000 * i:]).to(device)
                                    else:
                                        x_test = torch.from_numpy(test_data[15000 * i:15000 * (i + 1)]).to(device)
                                        print(x_test.shape)
                                    if CNN:
                                        x_test = x_test.view(x_test.shape[0], 1, 28, 28)
                                        x_test = squeeze(x_test, ch, dim[0], dim[1])

                                    x_test = F.one_hot(x_test, num_classes=vocab_size).float()
                                    zs_test = model.forward(x_test)
                                    if CNN:
                                        zs_test = zs_test.view((zs_test.shape[0], -1, vocab_size))
                                    logprob_test = zs_test * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
                                    loss_test = loss_test - torch.sum(logprob_test)
                                loss_test = loss_test / test_data.shape[0]
                            else:
                                x_test = torch.from_numpy(test_data).to(device)
                                if CNN:
                                    x_test = x_test.view(x_test.shape[0], 1, 28, 28)
                                    x_test = squeeze(x_test, ch, dim[0], dim[1])
                                x_test = F.one_hot(x_test, num_classes=vocab_size).float()
                                zs_test = model.forward(x_test)
                                if CNN:
                                    zs_test = zs_test.view((zs_test.shape[0], -1, vocab_size))
                                logprob_test = zs_test * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
                                loss_test = -torch.sum(logprob_test) / zs_test.shape[0]
                            end_test = time.time()

                        epoch_test_loss[e] = loss_test.item()
                        epoch_test_time[e] = end_test - start_test

                        del loss_test
                        del zs_test
                        del logprob_test

                        print('Epoch train loss')
                        print(avg_epoch_train_loss[e])
                        print('Epoch test loss')
                        print(epoch_test_loss[e])
                        print(epoch_test_time[e])
                        start_bool = False
                model.train()
                loss.backward()
                optimizer.step()

                end_train = time.time()
                losses.append(loss.item())

                total_loss += loss.item()

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

                total_time = total_time + end_train - start_train

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
            end = time.time()

            losses.append(loss.item())

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
        scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        avg_epoch_train_loss[e + 1] = total_loss / batch_num
        epoch_train_time[e + 1] = total_time
        print('avg_epoch_loss')
        print(avg_epoch_train_loss[e + 1])
        print(epoch_train_time[e + 1])

        if test_per_epoch:
            model.eval()
            with torch.no_grad():
                start_test = time.time()
                if test_data.shape[0] > 15000:
                    loss_test = torch.tensor(0)
                    for i in range(math.ceil(test_data.shape[0] / 15000)):
                        if i == test_data.shape[0] // 15000:
                            x_test = torch.from_numpy(test_data[15000 * i:]).to(device)
                        else:
                            x_test = torch.from_numpy(test_data[15000 * i:15000 * (i + 1)]).to(device)
                        if CNN:
                            x_test = x_test.view(x_test.shape[0], 1, 28, 28)
                            x_test = squeeze(x_test, ch, dim[0], dim[1])

                        x_test = F.one_hot(x_test, num_classes=vocab_size).float()
                        zs_test = model.forward(x_test)
                        if CNN:
                            zs_test = zs_test.view((zs_test.shape[0], -1, vocab_size))
                        logprob_test = zs_test * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
                        loss_test = loss_test - torch.sum(logprob_test)
                    loss_test = loss_test / test_data.shape[0]
                else:
                    x_test = torch.from_numpy(test_data).to(device)
                    if CNN:
                        x_test = x_test.view(x_test.shape[0], 1, 28, 28)
                        x_test = squeeze(x_test, ch, dim[0], dim[1])
                    x_test = F.one_hot(x_test, num_classes=vocab_size).float()
                    zs_test = model.forward(x_test)
                    if CNN:
                        zs_test = zs_test.view((zs_test.shape[0], -1, vocab_size))
                    logprob_test = zs_test * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
                    loss_test = -torch.sum(logprob_test) / zs_test.shape[0]
                end_test = time.time()

                epoch_test_loss[e + 1] = loss_test.item()
                epoch_test_time[e + 1] = end_test - start_test
                print('Epoch test loss')
                print(epoch_test_loss[e + 1])
                print(epoch_test_time[e + 1])

            del loss_test
            del zs_test
            del logprob_test

    if save:
        losses = np.array(losses)
        np.save(save_path + '_losses.npy', losses)
        np.save(save_path + '_epoch_test_loss.npy', epoch_test_loss)
        np.save(save_path + '_epoch_train_time.npy', epoch_train_time)
        np.save(save_path + '_epoch_test_time.npy', epoch_test_time)
        np.save(save_path + '_avg_epoch_train_loss.npy', avg_epoch_train_loss)

    return min_loss, total_time, path_name, model, base_log_probs


def test_disc_flow(device, data, vocab_size, is_load_path=True, load_path=None, load_model=None,
                   load_base_log_probs=None, CNN=False, dim=None):
    batch_size = data.shape[0]
    print(batch_size)

    if is_load_path:
        open_path = torch.load(load_path)
        model.load_state_dict(open_path['model_state_dict'])
        base_log_probs = open_path['prior']
    else:
        model = load_model

    model.eval()

    # print(open_path['loss'])

    start = time.time()
    x = torch.from_numpy(data).to(device)
    if CNN:
        ch = int(x.shape[1] / (dim[0] * dim[1]))
        x = x.view(x.shape[0], 1, 28, 28)
        x = squeeze(x, ch, dim[0], dim[1])

    x = F.one_hot(x, num_classes=vocab_size).float()

    zs = model.forward(x)

    base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1).to(device)
    # print(zs.shape, base_log_probs_sm.shape)
    logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
    loss = -torch.sum(logprob) / batch_size
    end = time.time()
    final_time = end - start
    del x
    del model
    del base_log_probs
    return loss.item(), final_time