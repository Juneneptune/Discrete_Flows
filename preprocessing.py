import sklearn.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
from train_utils import *
import tensorflow_datasets as tfds


def binarize_digits():
    digit = datasets.load_digits(10, True)
    data = []
    for i in range(10):
        data.append(np.zeros((len(np.where(digit[1] == i)[0]), 64)))
    for i in range(10):
        data[i] = digit[0][np.where(digit[1] == i)[0]]

    # Binarize Data
    for i in range(10):
        data[i] = np.where(data[i] >= 10, 1, 0)
    return data


def all_binarize_digits():
    digit = datasets.load_digits(10, True)[0]
    data = np.where(digit >= 10, 1, 0)
    return data

def download_mnist():
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor()]))
    label = torch.zeros(60000)
    data = torch.zeros((60000, 28, 28))

    for i in range(60000):
        a, b = trainset[i]
        data[i] = a
        label[i] = b
        if (i % 1000 == 0):
            print(i)

    for i in range(10):
        print('step' + str(i))
        loc = torch.where(label == i)[0]
        store = torch.zeros((loc.size()[0], 28, 28))
        print(store.size())
        count = 0
        for j in loc:
            store[count] = data[j.numpy()]
            count += 1
        name = 'mnist' + str(i) + '.pt'
        torch.save(store, name)


def binarize_MNIST(digit):
  if digit == 'all':
    for i in range(10):
      if i == 0:
        data = torch.load('mnist' + str(i) + '.pt')
      else:
        temp = torch.load('mnist' + str(i) + '.pt')
        data = torch.vstack((data, temp))
  else:
    data = torch.load('mnist' + str(digit) + '.pt')
  data = np.where(data.view(data.shape[0], -1) >= 0.5, 1, 0)

  return data

class Digits(Dataset):

    def __init__(self, num):
        self.data = binarize_digits()[num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class K_fold_Digits(Dataset):

    def __init__(self, k, fold_num):
        data = all_binarize_digits()
        self.data, _ = kfold_splitter(data, data.shape[0], k, fold_num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MNIST(Dataset):
    def __init__(self, digit):
        self.digit = digit
        self.data = binarize_MNIST(digit)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def Sample_Model(model, base_log_probs, vocab_size, disc_layer_type, sample_row=2, sample_col=10, CNN=False, dim=(28, 14)):
  #Sample Pior
  prior = torch.distributions.OneHotCategorical(logits=base_log_probs)
  base = prior.sample([sample_row * sample_col])
  #Inverse model
  if CNN:
    base = base.view((base.shape[0], -1, dim[0], dim[1], vocab_size))
  data = model.reverse(base)
  print(data.shape)
  #Removes one hot
  sample = torch.argmax(data, dim=-1)
  print(sample.shape)

  sample = sample.cpu().detach().numpy()

  for i in range(sample_row):
    for j in range(sample_col):
      if j==0:
        im = sample[10*i+j].reshape(28,28)
      else:
        im = np.hstack((im, sample[10*i+j].reshape(28,28)))
    if i==0:
      full_im = im
    else:
      full_im = np.vstack((full_im, im))

  np.save('MNIST_' + str(disc_layer_type) + '_' + 'all' + '.npy', full_im)
  plt.imshow(full_im)
  plt.savefig('MNIST_' + 'all', dpi=1000)
  plt.gray()
  plt.show()


def preprocess_binary_mnist():
  train,test = tfds.load('binarized_mnist', split=['train', 'test'])
  new_train = tfds.as_numpy(train)
  new_test = tfds.as_numpy(test)
  flattened_images = []
  for i,ex in enumerate(new_train):
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    flattened_images.append(ex['image'].flatten())
  for i,ex in enumerate(new_test):
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    flattened_images.append(ex['image'].flatten())
  return np.array(flattened_images).astype(int)


def Mai_kfold_splitter(n, k, fold_num, train_proportion):
    if k != 1:  # for kfold = 1 just use the train_proportion to split
        fold_size = n // k
        test_inds = list(range(fold_num * fold_size, (fold_num + 1) * fold_size))
        set1 = set(test_inds)
        set2 = set(list(range(0, n)))
        train_inds = list(set2 - set1)
    else:
        train_inds = list(range(0, round(n * train_proportion)))
        test_inds = list(range(round(n * train_proportion), n))
    return train_inds, test_inds


def Mai_create_X_train_test(X, train_portion, kfolds, fold_num):
    '''
    If number of folds is 1, you need to specify the train_portion
    '''
    n, _ = X.shape

    train_ind_perm, test_ind_perm = Mai_kfold_splitter(n, kfolds, fold_num, train_portion)

    X_train = X[train_ind_perm, :]
    X_test = X[test_ind_perm, :]

    return X_train, X_test


class DATA(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def kfold_splitter(data,n,k,fold_num):
    fold_size = n//k
    test_data = data[fold_num*fold_size:(fold_num+1)*fold_size]
    train_data = np.vstack((data[0:fold_num*fold_size], data[(fold_num+1)*fold_size:n]))
    return train_data, test_data