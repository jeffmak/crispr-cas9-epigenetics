#@title LeNup (NN architecture)
import torch as th
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 5
pad_size = 2
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class GatedConv(nn.Module):
  ''' Gated convolution module consisting of two Conv2d + BatchNorm2d layers.

  Attributes:
    conv, batch, conv_ctrl, batch_ctrl
  '''
  def __init__(self, in_channels, out_channels, kernel_size):
    ''' Initializes the Conv2d and BatchNorm2d layers'''
    super(GatedConv, self).__init__()
    pad_size = ((kernel_size-1)//2,0)
    self.conv = nn.Conv2d(in_channels, out_channels,
                          (kernel_size, 1),
                          padding=pad_size)
    self.batch = nn.BatchNorm2d(out_channels)
    self.conv_ctrl = nn.Conv2d(in_channels, out_channels,
                               (kernel_size, 1),
                               padding=pad_size)
    self.batch_ctrl = nn.BatchNorm2d(out_channels)

  def forward(self, input):
    ''' Forward pass in block.'''
    return F.relu(self.batch(self.conv(input))) * \
           th.sigmoid(F.relu(self.batch_ctrl(self.conv_ctrl(input))))

class GatedConvolutionA(nn.Module):
  ''' Gated convolutional block A consisting of three GatedConv modules.

      Attributes:
        one: GatedConv with 1 x 1 kernel
        three: GatedConv with 3 x 3 kernel
        seven: GatedConv with 7 x 7 kernel
  '''
  def __init__(self, in_channels):
    ''' Initializes the three GatedConv modules. '''
    super(GatedConvolutionA, self).__init__()
    self.out_depth = 128
    self.one = GatedConv(in_channels, self.out_depth, 1)
    self.three = GatedConv(in_channels, self.out_depth, 3)
    self.seven = GatedConv(in_channels, self.out_depth, 7)

  def forward(self, input):
    ''' Forward pass in block.'''
    return th.cat((self.one(input),
                   self.three(input),
                   self.seven(input)),
                  dim=1) # concat channels

class GatedConvolutionBC(nn.Module):
  ''' Gated convolutional block B/C consisting of multiple GatedConv layers.

  Attributes:
    init_out_depth: The no. of output channels for layers col3_one and col4_out.
    out_depth: The no. of output channels for layers col1_one, col3_seven and col4_seven.

    col1_one, col2_avg_pool, col2_one_pooled, col3_one, col3_seven, col4_one, col4_seven
  '''

  def __init__(self, in_channels, type=None):
    super(GatedConvolutionBC, self).__init__()
    self.init_out_depth, self.out_depth = 64, 96
    self.col1_one = GatedConv(in_channels, self.out_depth, 1)
    assert type != None
    f_width = 7 if type == 'B' else (11 if type == 'C' else None)

    self.col2_avg_pool = nn.AvgPool2d((3,1), stride=(1,1), padding=(1,0))
    self.col2_one_pooled = GatedConv(in_channels, self.out_depth, 1)

    self.col3_one = GatedConv(in_channels, self.init_out_depth, 1)
    self.col3_seven = GatedConv(self.init_out_depth, self.out_depth, f_width)

    self.col4_one = GatedConv(in_channels, self.init_out_depth, 1)
    self.col4_seven = GatedConv(self.init_out_depth, self.out_depth, f_width)


  def forward(self, input):
    ''' Forward pass in block.'''
    return th.cat((self.col1_one(input),
                   self.col2_one_pooled(self.col2_avg_pool(input)),
                   self.col3_seven(self.col3_one(input)),
                   self.col4_seven(self.col4_one(input))),
                  dim=1)

class LeNup(nn.Module):
  ''' PyTorch convolutional neural network (CNN) model for LeNup.

  Attributes:
    gated_conv_a, max_pool_2, dropout
    gated_conv_b1, max_pool_4, dropout_2
    gated_conv_b2, gated_conv_c1, max_pool_3, dropout_3
    gated_conv_c2, dropout_4, avg_pool
    linear, batch, dropout_5
    linear_2

  '''
  def __init__(self):
    super(LeNup, self).__init__()

    self.gated_conv_a = GatedConvolutionA(4)
    self.max_pool_2 = nn.MaxPool2d((2,1))
    self.dropout = nn.Dropout(p=0.5)

    self.gated_conv_b1 = GatedConvolutionBC(384, type='B')
    self.max_pool_4 = nn.MaxPool2d((4,1))
    self.dropout_2 = nn.Dropout(p=0.3)

    self.gated_conv_b2 = GatedConvolutionBC(384, type='B')
    self.gated_conv_c1 = GatedConvolutionBC(384, type='C')
    self.max_pool_3 = nn.MaxPool2d((3,1))
    self.dropout_3 = nn.Dropout(p=0.3)

    self.gated_conv_c2 = GatedConvolutionBC(384, type='C')
    self.dropout_4 = nn.Dropout(p=0.3)
    self.avg_pool = nn.AvgPool2d((6,1))

    self.linear = nn.Linear(384, 384)
    self.batch = nn.BatchNorm1d(384)
    self.dropout_5 = nn.Dropout(p=0.3)

    self.linear_2 = nn.Linear(384, 2)

  def forward(self, inputs):
    ''' Forward pass in the CNN.'''
    output = self.gated_conv_a(inputs).to(device)
    output = self.max_pool_2(output)
    output = self.dropout(output)

    output = self.gated_conv_b1(output)
    output = self.max_pool_4(output)
    output = self.dropout_2(output)

    output = self.gated_conv_b2(output)
    output = self.gated_conv_c1(output)
    output = self.max_pool_3(output)
    output = self.dropout_3(output)

    output = self.gated_conv_c2(output)
    output = self.dropout_4(output)
    output = self.avg_pool(output)

    output = th.reshape(output, (output.shape[0],-1))
    output = self.linear(output)
    output = self.batch(output)
    output = F.relu(output)
    output = self.dropout_5(output)
    output = self.linear_2(output)
    return output
