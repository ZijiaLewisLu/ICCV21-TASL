from collections import Counter, defaultdict
import numpy as np


class Prior():
    def __init__(self, n_classes, prior=None):
        self.n_classes = n_classes
        if prior is not None:
            self.prior = prior
        else:
            self.prior = np.ones((self.n_classes), dtype=np.float32) / self.n_classes

    def score(self, labels, frame_sampling):
        l = labels[-2] 
        return -np.log(self.prior[l]) * np.array(frame_sampling, dtype=np.float32)

    def update_prior(self, buffer):
        # count labels
        self.prior = np.zeros((self.n_classes), dtype=np.float32)
        for label_count in buffer.label_counts:
            self.prior += label_count
        self.prior = self.prior / np.sum(self.prior)
        # backup to uniform probability for unseen classes
        n_unseen = sum(self.prior == 0)
        self.prior = self.prior * (1.0 - float(n_unseen) / self.n_classes)
        self.prior = np.array( [ self.prior[i] if self.prior[i] > 0 else 1.0 / self.n_classes for i in range(self.n_classes) ] )

    def log_prior(self):
        return np.log(self.prior)

