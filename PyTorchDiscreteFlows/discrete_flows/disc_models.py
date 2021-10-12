"""
author: trentbrick
Code taken from: https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py
Which is introduced and explained in the paper: https://arxiv.org/abs/1905.10347 
And modified for PyTorch. 
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PyTorchDiscreteFlows.discrete_flows.disc_utils import *


class Net(nn.Module):

    def __init__(self, ch_in, ch_out, lin_in, lin_out, vocab_size, alpha, beta, af='linear', scale=False, id=True):
        super(Net, self).__init__()
        self.af = af
        self.scale = scale
        self.id = id
        if self.af == 'linear':
            if self.scale == False:
                out = lin_out
            else:
                out = 2 * lin_out
            self.even = (lin_in == lin_out)
            self.lin1 = nn.Linear(lin_in, lin_in // alpha * beta)
            self.lin2 = nn.Linear(lin_in // alpha * beta, lin_in // (alpha ** 2) * beta)
            self.lin3 = nn.Linear(lin_in // (alpha ** 2) * beta, lin_in // (alpha ** 2) * beta)
            self.lin4 = nn.Linear(lin_in // (alpha ** 2) * beta, lin_in // alpha * beta)
            self.lin5 = nn.Linear(lin_in // alpha * beta, out)
            self.bn1 = nn.BatchNorm1d(alpha * lin_in, affine=True)
            self.bn2 = nn.BatchNorm1d(alpha * lin_in, affine=True)
            self.bn3 = nn.BatchNorm1d(alpha * lin_in, affine=True)
            self.bn4 = nn.BatchNorm1d(alpha * lin_in, affine=True)
            self.bn5 = nn.BatchNorm1d(out, affine=True)
            self.lin_skip = nn.Linear(lin_in, lin_out)
            self.bn_skip = nn.BatchNorm1d(lin_out, affine=True)
            if self.id:
                self.lin1.weight.data = torch.zeros_like(self.lin1.weight.data)
                self.lin2.weight.data = torch.zeros_like(self.lin2.weight.data)
                self.lin3.weight.data = torch.zeros_like(self.lin3.weight.data)
                self.lin4.weight.data = torch.zeros_like(self.lin4.weight.data)
                self.lin5.weight.data = torch.zeros_like(self.lin5.weight.data)
                self.lin1.weight.data[
                    ..., [i for i in range(self.lin1.weight.data.shape[1]) if i % vocab_size == 0]] = 0.05 * torch.ones(
                    (self.lin1.weight.data.shape[0], self.lin1.weight.data.shape[1] // vocab_size))
                self.lin2.weight.data[
                    ..., [i for i in range(self.lin2.weight.data.shape[1]) if i % vocab_size == 0]] = 0.05 * torch.ones(
                    (self.lin2.weight.data.shape[0], self.lin2.weight.data.shape[1] // vocab_size))
                self.lin3.weight.data[
                    ..., [i for i in range(self.lin3.weight.data.shape[1]) if i % vocab_size == 0]] = 0.05 * torch.ones(
                    (self.lin3.weight.data.shape[0], self.lin3.weight.data.shape[1] // vocab_size))
                self.lin4.weight.data[
                    ..., [i for i in range(self.lin4.weight.data.shape[1]) if i % vocab_size == 0]] = 0.05 * torch.ones(
                    (self.lin4.weight.data.shape[0], self.lin4.weight.data.shape[1] // vocab_size))
                self.lin5.weight.data[
                    ..., [i for i in range(self.lin5.weight.data.shape[1]) if i % vocab_size == 0]] = 0.05 * torch.ones(
                    (self.lin5.weight.data.shape[0], self.lin5.weight.data.shape[1] // vocab_size))
                self.lin_skip.weight.data = torch.zeros_like(self.lin_skip.weight.data)
                print(self.lin1.weight.data)
                # self.lin_skip.weight.data[..., [i for i in range(self.lin_skip.weight.data.shape[1]) if i%vocab_size==0]] = 0.1 * torch.ones((self.lin_skip.weight.data.shape[0], self.lin_skip.weight.data.shape[1]//vocab_size))
        else:
            self.conv1 = nn.Conv2d(ch_in, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv6 = nn.Conv2d(128, 64, 3, padding=1)
            self.conv7 = nn.Conv2d(64, 32, 3, padding=1)
            self.conv8 = nn.Conv2d(32, 16, 3, padding=1)
            self.conv9 = nn.Conv2d(16, ch_in * 2, 3, padding=1)
            # self.lin1 = nn.Linear(lin_in, lin_out)
            # self.lin2 = nn.Linear(lin_out, 2*lin_in)
            self.bn1 = nn.BatchNorm2d(16, affine=False)
            self.bn2 = nn.BatchNorm2d(32, affine=False)
            self.bn3 = nn.BatchNorm2d(64, affine=False)
            self.bn4 = nn.BatchNorm2d(128, affine=False)
            self.bn5 = nn.BatchNorm2d(128, affine=False)
            self.bn6 = nn.BatchNorm2d(64, affine=False)
            self.bn7 = nn.BatchNorm2d(32, affine=False)
            self.bn8 = nn.BatchNorm2d(16, affine=False)
            self.bn9 = nn.BatchNorm2d(ch_in * 2, affine=False)

    def forward(self, x, inv=False):
        x_ = x.clone()
        if self.af == 'linear':
            if id:
                x = F.relu(self.lin_skip(x))
                x_ = F.relu((self.lin2(F.relu((self.lin1(x_))))))
                x_ = F.relu((self.lin4(F.relu((self.lin3(x_))))))
                x_ = F.relu(self.lin5(x_))
            else:
                if self.even == False:
                    x = F.relu(self.bn_skip(self.lin_skip(x)))
                else:
                    x_ = F.relu(self.bn2(self.lin2(F.relu(self.bn1(self.lin1(x_))))))
                    x_ = F.relu(self.bn4(self.lin4(F.relu(self.bn3(self.lin3(x_))))))
                    x_ = F.relu(self.bn5(self.lin5(x_)))
            '''
            x_ = F.relu((self.lin2(F.relu((self.lin1(x_))))))
            x_ = F.relu((self.lin4(F.relu((self.lin3(x_))))))
            x_ = F.relu((self.lin5(x_)))
            '''
            if self.scale:
                x_l, x_r = torch.chunk(x_, 2, dim=1)
        else:
            B, C, H, W = x.shape
            if self.af == 'relu':
                x_ = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x_))))))
                x_ = F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x_))))))
                x_ = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x_))))))
                x_ = F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x_))))))
                x_ = F.relu(self.bn9(self.conv9(x_)))
            elif self.af == 'tanh':
                x_ = F.tanh(self.bn2(self.conv2(F.tanh(self.bn1(self.conv1(x_))))))
                x_ = F.tanh(self.bn4(self.conv4(F.tanh(self.bn3(self.conv3(x_))))))
                x_ = F.tanh(self.bn6(self.conv6(F.tanh(self.bn5(self.conv5(x_))))))
                x_ = F.tanh(self.bn8(self.conv8(F.tanh(self.bn7(self.conv7(x_))))))
                x_ = F.tanh(self.bn9(self.conv9(x_)))
            # x_ = x_.view(B, -1)
            # x_ = self.lin2(F.relu(self.lin1(x_)))
            x_l, x_r = torch.chunk(x_.view(B, 2 * C, H, W), 2, dim=1)
        if self.scale:
            # Adding skip net for both scale and shift
            x_l = x + x_l
            x_r = x + x_r
            x = torch.cat((x_l, x_r), dim=1)
        else:
            x = x_ + x
        return x


class DiscreteAutoFlowModel(nn.Module):
    # combines all of the discrete flow layers into a single model
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
         # from the data to the latent space. This is how the base code is implemented. 
        for flow in self.flows:
            z = flow.forward(z)
        return z

    def reverse(self, x):
        # from the latent space to the data
        for flow in self.flows[::-1]:
            x = flow.reverse(x)
        return x

class Reverse(nn.Module):
    """Swaps the forward and reverse transformations of a layer."""
    def __init__(self, reversible_layer, **kwargs):
        super(Reverse, self).__init__(**kwargs)
        if not hasattr(reversible_layer, 'reverse'):
            raise ValueError('Layer passed-in has not implemented "reverse" method: '
                        '{}'.format(reversible_layer))
        self.forward = reversible_layer.reverse
        self.reverse = reversible_layer.forward


class DiscreteAutoregressiveFlow(nn.Module):
    """A discrete reversible layer.
    The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
    The flow returns a Tensor of same shape and dtype. (To enable gradients, the
    input must have float dtype.)
    For the reverse pass, from data to latent the flow computes in serial:
    ```none
    outputs = []
    for t in range(length):
        new_inputs = [outputs, inputs[..., t, :]]
        net = layer(new_inputs)
        loc, scale = tf.split(net, 2, axis=-1)
        loc = tf.argmax(loc, axis=-1)
        scale = tf.argmax(scale, axis=-1)
        new_outputs = (((inputs - loc) * inverse(scale)) % vocab_size)[..., -1, :]
        outputs.append(new_outputs)
    ```
    For the forward pass from data to latent, the flow computes in parallel:
    ```none
    net = layer(inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    outputs = (loc + scale * inputs) % vocab_size
    ```
    The modular arithmetic happens in one-hot space.
    If `x` is a discrete random variable, the induced probability mass function on
    the outputs `y = flow(x)` is
    ```none
    p(y) = p(flow.reverse(y)).
    ```
    The location-only transform is always invertible ([integers modulo
    `vocab_size` form an additive group](
    https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
    is invertible if the scale and `vocab_size` are coprime (see
    [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
    """

    def __init__(self, layer, temperature, vocab_size):
        """Constructs flow.
        Args:
        layer: Two-headed masked network taking the inputs and returning a
            real-valued Tensor of shape `[..., length, 2*vocab_size]`.
            Alternatively, `layer` may return a Tensor of shape
            `[..., length, vocab_size]` to be used as the location transform; the
            scale transform will be hard-coded to 1.
        temperature: Positive value determining bias of gradient estimator.
        **kwargs: kwargs of parent class.
        """
        super().__init__()
        self.layer = layer
        self.temperature = temperature
        self.vocab_size = vocab_size

    def reverse(self, inputs, **kwargs):
        """Reverse pass for left-to-right autoregressive generation. Latent to data.
        Expects to recieve a onehot."""
        # inputs = torch.Tensor(inputs)
        length = inputs.shape[-2]
        if length is None:
            raise NotImplementedError(
                'length dimension must be known. Ensure input is a onehot with 3 dimensions (batch, length, onehot)')
        # Slowly go down the length of the sequence.
        # the batch is computed in parallel, dont get confused with it and the sequence components!
        # From initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
        # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
        outputs = self._initial_call(inputs[:, 0, :], length, **kwargs)
        # TODO: Use tf.while_loop. Unrolling is memory-expensive for big
        # models and not valid for variable lengths.
        for t in range(1, length):
            outputs = self._per_timestep_call(outputs,
                                              inputs[..., t, :],
                                              length,
                                              t,
                                              **kwargs)
        return outputs

    def _initial_call(self, new_inputs, length, **kwargs):
        """Returns Tensor of shape [..., 1, vocab_size].
        Args:
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output.
        length: Length of final desired sequence.
        **kwargs: Optional keyword arguments to layer.
        """
        inputs = new_inputs.unsqueeze(1)  # new_inputs[..., tf.newaxis, :] # batch x 1 x onehots
        # TODO: To handle variable lengths, extend MADE to subset its
        # input and output layer weights rather than pad inputs.
        padded_inputs = F.pad(
            inputs, (0, 0, 0, length - 1))

        """
        All this is doing is filling the input up to its length with 0s. 
        [[0, 0]] * 2 + [[0, 50 - 1], [0, 0]] -> [[0, 0], [0, 0], [0, 49], [0, 0]]
        what this means is, dont add any padding to the 0th dimension on the front or back. 
        same for the 2nd dimension (here we assume two tensors are for batches), for the length dimension, 
        add length -1 0s after. 

        """
        net = self.layer(padded_inputs, **kwargs)  # feeding this into the MADE network. store these as net.
        if net.shape[-1] == 2 * self.vocab_size:  # if the network outputted both a location and scale.
            loc, scale = torch.split(net, self.vocab_size,
                                     dim=-1)  # tf.split(net, 2, axis=-1) # split in two into these variables
            loc = loc[..., 0:1, :]  #
            loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = scale[..., 0:1, :]
            scale = one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = multiplicative_inverse(scale,
                                                   self.vocab_size)  # could be made more efficient by calculating the argmax once and passing it into both functions.
            shifted_inputs = one_hot_minus(inputs, loc)
            outputs = one_hot_multiply(shifted_inputs, inverse_scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            loc = loc[..., 0:1, :]
            loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            outputs = one_hot_minus(inputs, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        return outputs

    def _per_timestep_call(self,
                           current_outputs,
                           new_inputs,
                           length,
                           timestep,
                           **kwargs):
        """Returns Tensor of shape [..., timestep+1, vocab_size].
        Args:
        current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
            generated sequence Tensor.
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output given current_outputs.
        length: Length of final desired sequence.
        timestep: Current timestep.
        **kwargs: Optional keyword arguments to layer.
        """
        inputs = torch.cat([current_outputs,
                            new_inputs.unsqueeze(1)], dim=-2)
        # TODO: To handle variable lengths, extend MADE to subset its
        # input and output layer weights rather than pad inputs.

        padded_inputs = F.pad(
            inputs, (0, 0, 0, length - timestep - 1))  # only pad up to the current timestep

        net = self.layer(padded_inputs, **kwargs)
        if net.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(net, self.vocab_size, dim=-1)
            loc = loc[..., :(timestep + 1), :]
            loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = scale[..., :(timestep + 1), :]
            scale = one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = multiplicative_inverse(scale, self.vocab_size)
            shifted_inputs = one_hot_minus(inputs, loc)
            new_outputs = one_hot_multiply(shifted_inputs, inverse_scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            loc = loc[..., :(timestep + 1), :]
            loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            new_outputs = one_hot_minus(inputs, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        outputs = torch.cat([current_outputs, new_outputs[..., -1:, :]], dim=-2)
        return outputs

    def forward(self, inputs, **kwargs):
        """Forward pass returning the autoregressive transformation. Data to latent."""

        net = self.layer(inputs, **kwargs)
        if net.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(net, self.vocab_size, dim=-1)
            scale = one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            scaled_inputs = one_hot_multiply(inputs, scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            scaled_inputs = inputs
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        outputs = one_hot_add(scaled_inputs, loc)
        return outputs

    def log_det_jacobian(self, inputs):
        return torch.zeros((1)).type(inputs.dtype)


# Discrete Bipartite Flow
class DiscreteBipartiteFlow(nn.Module):
    """A discrete reversible layer.
    The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
    The flow returns a Tensor of same shape and dtype. (To enable gradients, the
    input must have float dtype.)
    For the forward pass, the flow computes:
    ```none
    net = layer(mask * inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    outputs = ((inputs - (1-mask) * loc) * (1-mask) * inverse(scale)) % vocab_size
    ```
    For the reverse pass, the flow computes:
    ```none
    net = layer(mask * inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    outputs = ((1-mask) * loc + (1-mask) * scale * inputs) % vocab_size
    ```
    The modular arithmetic happens in one-hot space.
    If `x` is a discrete random variable, the induced probability mass function on
    the outputs `y = flow(x)` is
    ```none
    p(y) = p(flow.reverse(y)).
    ```
    The location-only transform is always invertible ([integers modulo
    `vocab_size` form an additive group](
    https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
    is invertible if the scale and `vocab_size` are coprime (see
    [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
    """

    def __init__(self, layer, parity, temperature, vocab_size, dim, isimage=False, scale_opt=False):
        """Constructs flow.
        Args:
        layer: Two-headed masked network taking the inputs and returning a
            real-valued Tensor of shape `[..., length, 2*vocab_size]`.
            Alternatively, `layer` may return a Tensor of shape
            `[..., length, vocab_size]` to be used as the location transform; the
            scale transform will be hard-coded to 1.
        mask: binary Tensor of shape `[length]` forming the bipartite assignment.
        temperature: Positive value determining bias of gradient estimator.
        **kwargs: kwargs of parent class.
        """
        super().__init__()
        self.layer = layer
        self.parity = parity  # going to do a block split. #torch.tensor(mask).float()
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.dim = dim  # Used if a convolutional layer is need for images
        self.scale_opt = scale_opt
        self.isimage = isimage

    def update_temperature(self, temperature):
        self.temperature = temperature
        print('current temp' + str(self.temperature))

    def reverse(self, inputs, **kwargs):
        """reverse pass for bipartite data to latent."""
        # TODO: implement even odd shuffling.

        # Remove Embedded and flatten
        # Remove scale for now
        # Flatten -> Hidden -> Relu -> Output (Loc)
        z0, z1 = inputs[:, :inputs.shape[1] // 2].type(torch.float), inputs[:, inputs.shape[1] // 2:].type(
            torch.float)  # dim is proportionally divided in preporcessing
        # print(z0.shape)
        if self.parity:
            x0, x1 = z1, z0
        else:
            x0, x1 = z0, z1
        # print(z0.view(z0.shape[0], -1).shape)
        if self.isimage:
            layer_out = self.layer(x0.view(x0.shape[0], x0.shape[1], x0.shape[2], -1))
        else:
            layer_out = self.layer(x0.view(x0.shape[0], -1))
        # print(layer_out.shape)
        if self.scale_opt:
            loc, scale = torch.chunk(layer_out, 2, dim=1)
        else:
            loc = layer_out
        # print(loc.shape)
        # Reshape both loc and scale
        if self.isimage:
            loc = loc.view(loc.shape[0], loc.shape[1], loc.shape[2], int(loc.shape[3] / self.vocab_size), -1)
            if self.scale_opt:
                scale = scale.view(scale.shape[0], scale.shape[1], scale.shape[2],
                                   int(scale.shape[3] / self.vocab_size), -1)
        else:
            loc = loc.view(loc.shape[0], int(loc.shape[-1] / self.vocab_size), -1)
            if self.scale_opt:
                scale = scale.view(scale.shape[0], int(scale.shape[-1] / self.vocab_size), -1)
        # print(loc.shape)
        # print(scale.shape)

        loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        # print(x1.shape)
        if self.scale_opt:
            scale = one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = multiplicative_inverse(scale, self.vocab_size)
            shifted_inputs = one_hot_minus(x1, loc)
            x1 = one_hot_multiply(shifted_inputs, inverse_scale)
        else:
            x1 = one_hot_minus(x1, loc)

        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)

        return x

    def forward(self, inputs, **kwargs):
        # Remove Embedded and flatten
        # Remove scale for now
        # Flatten -> Hidden -> Relu -> Output (Loc)
        # outputting loc and scale
        # print(inputs.shape)
        x0, x1 = inputs[:, :inputs.shape[1] // 2].type(torch.float), inputs[:, inputs.shape[1] // 2:].type(
            torch.float)  # dim is proportionally divided in preporcessing
        # print(x0.shape)
        # print(x1.shape)
        if self.parity:
            z0, z1 = x1, x0
        else:
            z0, z1 = x0, x1
        # print(z0.view(z0.shape[0], -1).shape)
        # print(z1.shape)
        if self.isimage:
            layer_out = self.layer(z0.view(z0.shape[0], z0.shape[1], z0.shape[2], -1))
        else:
            layer_out = self.layer(z0.view(z0.shape[0], -1))
            # print(layer_out.shape)

        # print(torch.nn.functional.one_hot(layer_out.type(torch.long), num_classes=self.vocab_size))
        if self.scale_opt:
            loc, scale = torch.chunk(layer_out, 2, dim=1)
        else:
            loc = layer_out
            # print(loc)
        # print(loc.shape)
        # Reshape both loc and scale
        if self.isimage:
            loc = loc.view(loc.shape[0], loc.shape[1], loc.shape[2], int(loc.shape[3] / self.vocab_size), -1)
            if self.scale_opt:
                scale = scale.view(scale.shape[0], scale.shape[1], scale.shape[2],
                                   int(scale.shape[3] / self.vocab_size), -1)
        else:
            loc = loc.view(loc.shape[0], int(loc.shape[-1] / self.vocab_size), -1)
            if self.scale_opt:
                scale = scale.view(scale.shape[0], int(scale.shape[-1] / self.vocab_size), -1)

        # print(loc.shape)
        # print(scale.shape)

        if self.scale_opt:
            scale = one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            scaled_inputs = one_hot_multiply(z1, scale)
        else:
            scaled_inputs = z1
        # print(loc.shape)
        loc = one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        # print(loc.shape)
        # print(scaled_inputs.shape)
        # print(loc)
        # print(scaled_inputs.shape)
        # loc = torch.zeros((scaled_inputs.shape[0], scaled_inputs.shape[1], scaled_inputs.shape[2]))
        # loc[..., 0] = torch.ones((scaled_inputs.shape[0], scaled_inputs.shape[1]))
        # loc = loc.to('cuda')
        z1 = one_hot_add(scaled_inputs, loc)
        # print(z1.shape)
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        # print(z.shape)
        return z

    def log_det_jacobian(self, inputs):
        return torch.zeros((1)).type(inputs.dtype)