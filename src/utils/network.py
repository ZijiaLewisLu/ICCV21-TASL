#!/usr/bin/python3

# import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from .grammar import SingleTranscriptGrammar
from .length_model import PoissonModel
from .utils import Recorder
import pickle
from collections import Counter

def to_np(tensor):
    return tensor.detach().cpu().numpy()


# buffer for old sequences (robustness enhancement: old frames are sampled from the buffer during training)
class Buffer(object):

    def __init__(self, buffer_size, n_classes):
        self.vfnames = []
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.gt_labels = []
        self.softlabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, vfname, features, transcript, framelabels, gt_labels, softlabels=None):
        assert features.shape[1] == len(framelabels)
        assert features.shape[1] == len(gt_labels)
        if len(self.features) < self.buffer_size:
            # sequence data 
            self.vfnames.append(vfname)
            self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            self.gt_labels.append(gt_labels)
            if softlabels is not None:
                self.softlabels.append(softlabels)
            # statistics for prior and mean lengths
            self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            self.vfnames[self.next_position] = vfname
            self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            self.gt_labels[self.next_position] = gt_labels
            if softlabels is not None:
                self.softlabels[self.next_position] = softlabels
            # statistics for prior and mean lengths
            self.instance_counts[self.next_position] = self.count(transcript) #np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = self.count(framelabels) #np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size
        # update frame selectors
        self.frame_selectors = []
        for seq_idx in range(len(self.features)):
            self.frame_selectors.extend([ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ])

    def count(self, labels):
        ct = Counter(labels)
        counts = np.zeros(self.n_classes, dtype=np.int)
        for i, k in ct.items():
            counts[i] = k
        return counts

    def random(self):
        i = np.random.choice(len(self.frame_selectors))
        return self.frame_selectors[i] # return sequence_idx and frame_idx within the sequence

    def n_frames(self):
        return len(self.frame_selectors)

    def load(self, buffer_save, dataset=None):

        if isinstance(buffer_save, str):
            with open(buffer_save, 'rb') as fp:
                buffer_save = pickle.load(fp)

        for v in buffer_save:
            transcript, framelabels, gt = buffer_save[v]
            self.vfnames.append(v)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            self.gt_labels.append(gt)
            if dataset is not None:
                self.features.append(dataset.features[v])

            # statistics for prior and mean lengths
            self.instance_counts.append( self.count(transcript) )
                    # np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( self.count(framelabels) )
                    # np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )

        if dataset is not None:
            # update frame selectors
            self.frame_selectors = []
            for seq_idx in range(len(self.features)):
                self.frame_selectors.extend([ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ])

    def save(self, buffer_save_file):
        buffer_save = {}
        for t in range(len(self.vfnames)):
            v = self.vfnames[t]
            transcript = self.transcript[t]
            label = self.framelabels[t]
            gt   = self.gt_labels[t]
            buffer_save[v] = [ transcript, label, gt ]
        with open(buffer_save_file, 'wb') as fp:
            pickle.dump(buffer_save, fp)

# wrapper class to provide torch tensors for the network
class DataWrapper(torch.utils.data.Dataset):

    # for each frame in the sequence, create a subsequence of length window_size
    def __init__(self, sequence, window_size = 21):
        self.features = []
        self.labels = []
        self.gt_labels = []
        # ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        # extract temporal window around each frame of the sequence
        for frame in range(sequence.shape[1]):
            left, right = max(0, frame - window_size // 2), min(sequence.shape[1], frame + 1 + window_size // 2)
            tmp = np.zeros((sequence.shape[0], window_size), dtype=np.float32 )
            tmp[:, window_size // 2 - (frame - left) : window_size // 2 + (right - frame)] = sequence[:, left : right]
            self.features.append(np.transpose(tmp))
            self.labels.append(-1) # dummy label, will be updated after Viterbi decoding
            self.gt_labels.append(-1)

    # add a sampled (windowed frame, label) pair to the data wrapper (include buffered data during training)
    # @sequence the sequence from which the frame is sampled
    # @label the Viterbi decoding label for the frame at frame_idx
    # @frame_idx the index of the frame to sample
    def add_buffered_frame(self, sequence, label, gt_label, frame_idx):
        left, right = max(0, frame_idx - self.window_size // 2), min(sequence.shape[1], frame_idx + 1 + self.window_size // 2)
        tmp = np.zeros((sequence.shape[0], self.window_size), dtype=np.float32 )
        tmp[:, self.window_size // 2 - (frame_idx - left) : self.window_size // 2 + (right - frame_idx)] = sequence[:, left : right]
        self.features.append(np.transpose(tmp))
        self.labels.append(label)
        self.gt_labels.append(gt_label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        assert idx < len(self)
        features = torch.from_numpy( self.features[idx] )
        if isinstance(self.labels[idx], np.ndarray):
            labels = torch.from_numpy(self.labels[idx])
        else:
            labels = torch.from_numpy( np.array([self.labels[idx]], dtype=np.int64) )
        gt_labels = torch.from_numpy( np.array([self.gt_labels[idx]], dtype=np.int64) )
        return features, labels, gt_labels

class Net(nn.Module):

    def __init__(self, input_dim, hidden_size, space_size, n_classes,
            pred_size=0, autoencoder_weight=0):
        super(Net, self).__init__()

        init = torch.nn.init.xavier_normal_

        self.n_classes = n_classes
        self.space_size = space_size
        self.pred_size = pred_size
        self.autoencoder_weight = autoencoder_weight

        self.gru = nn.GRU(input_dim, hidden_size, 1, 
                            bidirectional = False, batch_first = True)

        self.encode_matrix = nn.Parameter(torch.randn(n_classes, hidden_size, space_size))
        self.encode_bias   = nn.Parameter(torch.randn(1, n_classes, space_size))
        init(self.encode_matrix)
        init(self.encode_bias)

        if self.pred_size > 0:
            self.pred_matrix = nn.Parameter(torch.randn(n_classes, space_size, pred_size))
            self.pred_bias   = nn.Parameter(torch.randn(1, n_classes, pred_size))
            init(self.pred_matrix)
            init(self.pred_bias)

        self.decode_matrix = nn.Parameter(torch.randn(n_classes, space_size, hidden_size))
        self.decode_bias   = nn.Parameter(torch.randn(1, 1, n_classes, hidden_size))
        init(self.decode_matrix)
        init(self.decode_bias)

    def forward(self, x):
        # GRU
        output, dummy = self.gru(x)
        output = output[:, -1, :] # B, hidden_size
                
        self.feature = output

        # Encoder channel
        self.embs = torch.einsum("bh,ahs->bas", output, self.encode_matrix) # B, n_classes, space_size
        embs = self.embs = self.embs + self.encode_bias
        
        if self.pred_size > 0:
            self.pred = torch.einsum("bas,asp->bap", self.embs, self.pred_matrix)
            embs = self.pred = self.pred + self.pred_bias

        # Decoder channel
        self.recons = torch.einsum("bas,ash->bah", self.embs, self.decode_matrix) # B, n_classes, space_size
        self.recons = self.recons + self.decode_bias[0]

        score = (embs ** 2).sum(2)
        logprob = self.logit = nn.functional.log_softmax(score, dim=1) # tensor is of shape (batch_size, num_classes)

        feature = torch.unsqueeze(self.feature, 1) # B, 1, space_size
        self.recons_error = ((feature - self.recons)**2)
        self.recons_error = self.recons_error.sum(2) # B, n_classes
        self.recons_error = -nn.functional.log_softmax(-self.recons_error, dim=1)

        return output, logprob

    def forward_and_loss(self, inputs, targets):
        self.forward(inputs)

        self.loss = nn.functional.nll_loss(self.logit, targets)

        prob = torch.exp(self.logit)
        self.entropy = (- self.logit * prob).sum(1).mean()

        B = self.embs.size(0)
        self.recons_loss = self.recons_error[torch.arange(B), targets].mean()
        self.loss_all = self.loss + self.recons_loss * self.autoencoder_weight

class Forwarder(object):

    def __init__(self, net, n_classes):
        self.n_classes = n_classes
        self.net = net
        self.net.cuda()

    def _forward(self, data_wrapper, batch_size = 512):
        dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size = batch_size, shuffle = False)
        # forward all frames
        all_feature = []
        all_embs = []
        log_probs = []
        recons_error = []
        for data in dataloader:
            input, _, _ = data
            input = input.cuda()
            feauture, logit = self.net(input) # Batch_size, num_classes
            log_probs.append(logit)
            all_feature.append(feauture.detach().cpu().numpy())
            all_embs.append(self.net.embs.detach().cpu().numpy())
            recons_error.append(self.net.recons_error)

        all_feature = np.concatenate(all_feature, axis=0)
        all_embs    = np.concatenate(all_embs, axis=0)
        log_probs = torch.cat(log_probs, dim=0)
        recons_error = torch.cat(recons_error, dim=0) # num_frame x num_classes
        return all_feature, all_embs, log_probs, recons_error

    def forward(self, sequence, batch_size = 512, window_size = 21):
        data_wrapper = DataWrapper(sequence, window_size = window_size)
        return self._forward(data_wrapper, batch_size=batch_size)

    def load_model(self, model_file):
        # self.net.cpu()
        self.net.load_state_dict( torch.load(model_file) )
        # self.net.cuda()


