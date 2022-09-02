import torch
import pickle
import pandas as pd
import numpy as np

### IMPORTANT CONSTANTS ###
tarSeqLen = 23 # target DNA sequence's length
###########################

class GaussianNoise(torch.nn.Module):
  ''' Custom module to implement Gaussian random noise
  in nn.Sequential containers.

  Attributes:
    mean: Float indicating the Gaussian noise's mean.
    stdev: Float indicating the Gaussian noise's standard deviation.
  '''
  def __init__(self, mean, stdev):
    ''' Initializes GaussianNoise with mean and standard deviation.
    Input:
      mean: Float indicating the Gaussian noise's mean.
      stdev: Float indicating the Gaussian noise's standard deviation.
    Output:
      None
    '''
    super(GaussianNoise, self).__init__()
    self.mean = mean
    self.stdev = stdev

  def forward(self, ins):
    ''' Forward pass in layer. Only adds noise during training. '''
    if self.training:
      noise = ins.data.new(ins.size()).normal_(self.mean, self.stdev)
      return ins + noise
    return ins

class ConvolutionalNet(torch.nn.Module):
  ''' PyTorch convolutional neural network (CNN) regression model used for
  predicting CRISPR-Cas9 (off-)target activity values.

  Non-layer attributes:
    device: The compute device used for tensor computations (e.g., cpu, gpu).
    p: Dropout probability used in EncodeLayer1's nn.Dropout layer.
    mean: Mean used in EncodeLayer1's GaussianNoise layer.
    stdev: Standard deviation used in EncodeLayer1's GaussianNoise layer.
    batchnorm_momentum: Momentum of BatchNorm's running estimates.
    epiDim: Number of epigenetic features (always 22).

  Layer attributes:
    BatchNorm: 1D BatchNorm layer.
    EncodeLayer1: nn.Sequential block consisting of a Conv1d,
                  GaussianNoise, Dropout and then a LeakyReLU layer.
    EncodeLayer2: nn.Sequential block consisting of a Conv1d
                  and then a LeakyReLU.
    EncodeLayer3: nn.Sequential block consisting of a Conv1d, BatchNorm
                  and then a LeakyReLU.
    ConjoinedLayer1: nn.Sequential block consisting of a Conv1d, MaxPool1d
                  and then a ReLU.
    ConjoinedLayer2: nn.Sequential block consisting of a Conv1d, MaxPool1d
                  and then a ReLU.
    conjoinedLinear: nn.Linear layer.
  '''

  def __init__(self, epiDim, device=None,
               p = 0.0, mean = 0, stdev = 0.254, batchnorm_momentum = 0.1):
    ''' Initializes ConvolutionalNet with training parameters
    and neural net layers

    Input:
      epiDim: Number of epigenetic features (always 22).
      device: The compute device used for tensor computations
              (e.g., cpu, gpu).
      p: Dropout probability used in EncodeLayer1's nn.Dropout layer.
      mean: Mean used in EncodeLayer1's GaussianNoise layer.
      stdev: Standard deviation used in EncodeLayer1's GaussianNoise layer.
      batchnorm_momentum: Momentum of BatchNorm's running estimates.

    Output:
      None
    '''
    super(ConvolutionalNet, self).__init__()
    self.device = device
    self.p, self.mean, self.stdev = p, mean, stdev
    self.batchnorm_momentum = batchnorm_momentum
    self.epiDim = epiDim
    self.BatchNorm = torch.nn.BatchNorm1d(self.epiDim,
                                          momentum=self.batchnorm_momentum)
    self.BatchNorm.to(self.device)

    # Encode Layers
    # don't set batchnorm_momentum as this will probably mess up
    # the running stats
    self.EncodeLayer1 = torch.nn.Sequential(
      torch.nn.Conv1d(int(self.epiDim), 32, kernel_size=3, stride=2,
                      padding=0),
      GaussianNoise(self.mean, self.stdev), # set Gaussian mean and stdev
      torch.nn.Dropout(p=self.p), # set dropout p
      torch.nn.LeakyReLU(0.2)
      )
    self.EncodeLayer1.to(self.device)

    self.EncodeLayer2 = torch.nn.Sequential(
      torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0),
      torch.nn.LeakyReLU(0.2)
      )
    self.EncodeLayer2.to(self.device)


    self.EncodeLayer3 = torch.nn.Sequential(
      torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0),
      torch.nn.BatchNorm1d(128),
      torch.nn.LeakyReLU(0.2)
      )
    self.EncodeLayer3.to(self.device)

    # Conjoined Layers
    self.conjoinedLayer1 = torch.nn.Sequential(
      torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0),
      torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1), # preserve
                                                              # dimensions
      torch.nn.ReLU()
      )
    self.conjoinedLayer1.to(self.device)

    self.conjoinedLayer2 = torch.nn.Sequential(
      torch.nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
      torch.nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
      torch.nn.ReLU()
      )
    self.conjoinedLayer2.to(self.device)

    self.conjoinedLinear = torch.nn.Linear(512, 1)
    self.conjoinedLinear.to(self.device)

  def forward(self, x):
    ''' Performs a forward pass using the instantiated ConvolutionalNet
    model.

    Input:
      x: PyTorch tensor as input to the ConvolutionalNet model.
         The tensor contains the 22 epigenetic features.
         Dimension: (X.shape[0], epiDim, tarSeqLen), where
           - X.shape[0]         is the # of datapoints,
           - epiDim (== 22)     is the # of epigenetic features
           - tarSeqLen (== 23)  is the target sequence's length.

    Output:
      out: PyTorch tensor containing predicted CRISPR-Cas9 (off-)target
           cleavage activity values.
    '''
    x = x.view(x.size(0), self.epiDim, -1).to(self.device)
    x = self.BatchNorm(x)

    # forward pass thru encode layers
    x_enc = self.EncodeLayer1(x)
    x_enc = self.EncodeLayer2(x_enc)
    seq_encoding = self.EncodeLayer3(x_enc)

    # forward pass thru conjoined layers
    out = self.conjoinedLayer1(seq_encoding)
    out = self.conjoinedLayer2(out)

    # flatten last axis to 512 x 1
    out = out.reshape(out.size(0), -1)

    # forward pass thru linear layer
    out = self.conjoinedLinear(out)
    return out # regression

def vecToMatEncoding(X, numBpWise=0):
  ''' Formats the epigenetic feature-containing PyTorch tensor so that it
      is ready as input to the ConvolutionalNet model.

      To do this, we first transpose the first numBpWise*tarSeqLen columns
      in PyTorch Tensor X. These columns come in sets of tarSeqLen, where
      each set corresponds to a base pair-resolved computed nucleosome
      organization score. Then, for the remaining columns in X, we repeat
      each column tarSeqLen times. Finally, we concatenate the two tensors
      together to form the output. Here is a conceptual visualization of how
      this function transforms each column in X

      bp-resolved column features: a, b, ..., m (there are 13 of these)
      non-bp-resolved column features: n, o,..., v (there are 9 of these)
                | vecToMatEncoding
                |
                v
      X (size 308)     : [aa...abb...b...mm...mno...v]
      output (size 22 x 23): [[a...a], [b...b], ..., [m...m],
                              [n...n], [o...o], ..., [v...v],]

  Input:
    X: 2D PyTorch tensor. Each row represents a datapoint, and
       each column represents an epigenetic feature
       Dimension: (# of datapoints, 308), where
       308 = numBpWise * tarSeqLen
             + (# of non-bp-resolved (energy/experimental) epigenetic scores)
           = 13 * 23 + 9
    numBpWise: Number of base pair-resolved nucleosome organization-related
               scores/features (always 13).

  Output:
    PyTorch tensor ready for forward pass.
    Dimension: (X.shape[0], epiDim, tarSeqLen), where
      - X.shape[0]         is the # of datapoints,
      - epiDim (== 22)     is the # of epigenetic features
      - tarSeqLen (== 23)  is the target sequence's length.
  '''

  bpwise_features = []
  for i in range(numBpWise):
    # we format each base pair-wise feature to have
    # dimension (X.shape[0], 1, tarSeqLen)
    bpwise_features.append(X[:, i*tarSeqLen:(i+1)*tarSeqLen] \
                    .view(X.shape[0], tarSeqLen, 1) \
                    .transpose(1,2))

  # None indexing means adding an axis - could also use .unsqueeze() here
  epigenetics = X[:, numBpWise*tarSeqLen:, None].repeat(1, 1, tarSeqLen)

  # return epigenetics and bpwise_features
  return torch.cat(tuple(bpwise_features)+(epigenetics,), dim=1)
