#@title NeuralNetPredictor
import numpy as np
import torch as th
from lenup_nn import LeNup

class NeuralNetPredictor():
  ''' Wrapper Class for the LeNup model. '''

  @staticmethod
  def _one_hot_encode_col(in_str):
    ''' One-hot encodes an input DNA sequence.

    Input:
      in_str: input DNA sequence.

    Output:
        a NumPy array of size (len(in_str), 4).
    '''
    arr = np.array([{'A': 0, 'C': 1, 'G': 2, 'T':3}[base] for base in in_str])
    out = np.zeros((arr.size, 4), dtype=np.uint8)
    out[np.arange(arr.size), arr] = 1
    return np.array(out)

  @classmethod
  def _convert_batch(cls, net, input_seqs):
    ''' Converts off-target context DNA sequences into batches of one-hot encoded
    sequences.

    Input:
      cls: NeuralNetPredictor.
      net: the neural network model used for nucleosome occupancy prediction.
      input_seqs: list of 169bp off-target context DNA sequences.

    Output:
      NumPy array of size (min(len(input_seqs), batch_size), 23).
    '''
    num_out_bps = len(input_seqs[0]) - 147 + 1
    oh_seqs = [cls._one_hot_encode_col(seq) for seq in input_seqs]
    seq_147s = [np.stack([oh_seq[idx:idx+147,:] for idx in range(num_out_bps)]) \
                                                for oh_seq in oh_seqs]
    oh_seqs = np.stack(seq_147s, axis=0)


    inputs = oh_seqs.reshape(oh_seqs.shape[0] * oh_seqs.shape[1],
                             oh_seqs.shape[2],
                             oh_seqs.shape[3])

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    temp = th.Tensor(inputs).to(device).permute(0,2,1).unsqueeze(-1)
    out = th.sigmoid(net(temp)
                     .squeeze(-1))

    out = out[:, 1]
    out = out.reshape(oh_seqs.shape[0], oh_seqs.shape[1])
    return out

  @classmethod
  def batch_occupancy_scores(cls, input_seqs,
                             model=LeNup,
                             model_weights='model/lenup_h3q85c.th',
                             batch_size=128):
    """ Receives a list of strings representing and returns base pair-resolved
    nucleosome occupancy scores for each string.

    Input:
      cls: NeuralNetPredictor.
      input_seqs: list of 169bp off-target context DNA sequences.
      model: the neural network model class used for nucleosome occupancy
             prediction.
      model_weights: file location containing the PyTorch model weights.
      batch_size: batch size used during the model's forward pass.

    Output:
      NumPy array of size (min(len(input_seqs), batch_size), 23).
    """
    assert np.std([len(seq) for seq in input_seqs]) == 0

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    net = model().to(device)
    net.load_state_dict(th.load(model_weights, map_location=device))

    out = []
    with th.no_grad():
      net.eval()
      # split n into ceil(n/128) chunks to feed to neural net
      out = [cls._convert_batch(net,
                input_seqs[idx:idx+batch_size]).cpu() \
            for idx in range(0,len(input_seqs),batch_size)]
    out = th.cat(out, dim=0)
    return out
