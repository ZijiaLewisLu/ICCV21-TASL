import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
from .grammar import SingleTranscriptGrammar
from .viterbi import Viterbi
from .network import Forwarder, Buffer, DataWrapper, Net

def to_np(tensor):
    return tensor.detach().cpu().numpy()

class Trainer(Forwarder):

    def __init__(self, decoder: Viterbi, net: Net, n_classes, buffer_size, 
                buffered_frame_ratio = 25, recorder=None):
        super(Trainer, self).__init__(net, n_classes)
        self.buffer = Buffer(buffer_size, n_classes)
        self.decoder = decoder
        self.buffered_frame_ratio = buffered_frame_ratio
        self.recorder = recorder

    def train(self, dataset,
            batch_size = 512, learning_rate = 0.1, window_size = 21, 
            edge_window = 10, edge_step = 5):
        
        vfname, sequence, transcript, gt_label = dataset.get()

        data_wrapper = DataWrapper(sequence, window_size=window_size)
        # forwarding and Viterbi decoding
        feature, embs, log_probs_origin, recons_error = self._forward(data_wrapper, batch_size=batch_size)
        log_probs = log_probs_origin.detach().cpu().numpy()

        # define transcript grammar and updated length model
        self.decoder.grammar = SingleTranscriptGrammar(transcript, self.n_classes)
        # decoding
        score, labels, segments = self.decoder.decode(log_probs)

        # record decode acc
        for (p, g) in zip(labels, gt_label):
            self.recorder.append("decode_acc", p==g)

        ### compute loss
        video_length = log_probs_origin.shape[0]
        optimizer = optim.SGD(self.net.parameters(), lr = learning_rate / batch_size)
        optimizer.zero_grad()
        penalty = -log_probs_origin

        loss1 = self.decoder.forward_score(penalty, segments, transcript, edge_window, edge_step)
        loss2 = self.decoder.incremental_forward_score(penalty, segments, transcript, edge_window, edge_step)
        loss = loss1 - loss2

        ### compute logsumexp gradient
        penalty_grad = torch.autograd.grad(loss, penalty)[0]

        penalty = penalty + self.net.autoencoder_weight * recons_error
        
        penalty.backward(penalty_grad)
        optimizer.step()

        # add sequence to buffer
        self.buffer.add_sequence(vfname, sequence, transcript, labels, gt_label)
        # update prior and mean lengths
        self.decoder.prior_model.update_prior(self.buffer)
        self.decoder.length_model.update_mean_lengths(self.buffer)
        
        ## save results
        logprob_loss = (-log_probs_origin*penalty_grad).sum().item() / video_length
        self.recorder.append("loss", logprob_loss)
        recons_loss = (recons_error*penalty_grad).sum().item() * self.net.autoencoder_weight / video_length
        self.recorder.append("recons_loss", recons_loss)

        return score, log_probs_origin, labels, feature

    def save_model(self, network_file, length_file, prior_file):
        torch.save(self.net.state_dict(), network_file)
        np.savetxt(length_file, self.decoder.length_model.mean_lengths)
        np.savetxt(prior_file, self.decoder.prior_model.prior)