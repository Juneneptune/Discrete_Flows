# Implementation_of_Discrete_Flows_Paper.ipynb

## Acknowledgements
Our implementation is based on this paper: https://arxiv.org/abs/1905.10347. 
This implementation is a slight modification of the code by Trenton Bricken taken from: https://github.com/TrentBrick/PyTorchDiscreteFlows/tree/master/discrete_flows (which was originally a modification from https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/discrete_flows.py and https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/utils.py)

## Modifications Done
https://github.com/TrentBrick/PyTorchDiscreteFlows/tree/master/discrete_flows code had runtime errors when applying the DiscreteBipartiteFlow class in https://github.com/TrentBrick/PyTorchDiscreteFlows/blob/master/discrete_flows/disc_models.py for dimensions larger than 2, and the code was not programmed to do odd dimensions. The embedding flow layer was removed (commonly used in NLP sequence data which is not the case for our experiments) and replaced with a single hidden layer (hidden layer with a ReLU activation function). The bipartite architecture was also adjusted to support odd dimensions for the DiscreteBipartiteFlow class.

## Key Notes
The code also had issues of using past versions of PyTorch functions (e.i. `torch.fft.fft()`) that were no longer compatible with the new version. Therefore, the version was downgraded to PyTorch 1.7.1. Moreover, the notebook had a `git` command line that directly downloaded the github repository `!git clone https://github.com/TrentBrick/PyTorchDiscreteFlows.git`.

## Running Experiment Code
To run the code, all codes in these sections must be run in these respective order (Fundamental Test section can be ignored):
- Setup
- Modified Discrete Bipartite Flow Model
  - Our modified DiscreteBipartiteFlow class.
- Synethetic Data / Functions
  - Own functions for generating synthetic data.
- Discrete Flow Functions
- Other Functions
- Synthetic Data Testing Bipartite
  - Experiments 1-5 for discrete bipartite flows.
- Synthetic Data Testing Autoregressive
  - Experiments 1-5 for discrete autoregressive flows.
- Mushroom Dataset Testing
  - Mushroom dataset experiments for both bipartite and autoregressive flows.
- Copula Data Testing
  - Experiments 6-9 for both bipartite and autoregressive flows.

## Output
The experment sections will output an average and standard deviation of the negative log-likelihood values and its computation time over a 5-fold cross validation.
