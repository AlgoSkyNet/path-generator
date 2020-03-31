import datetime
import numpy as np
import pandas_datareader as pdr
from esig import tosig
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

from utils.leadlag import leadlag
from cvae import CVAE

class Generator:
    def __init__(self, corpus, sig_order, **params):
        self.corpus = corpus
        self.order = sig_order

        # Model parameters
        n_latent = params.get("n_latent", 8)
        alpha = params.get("alpha", 0.003)

        self._build_dataset()
        self.generator = CVAE(n_latent=n_latent, alpha=alpha)


    def _logsig(self, path):
        return tosig.stream2logsig(path, self.order)

    def _build_dataset(self):
        self.logsigs = np.array([self._logsig(path) for path in tqdm(self.corpus, desc="Computing log-signatures")])

        self.scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
        self.logsigs_norm = self.scaler.fit_transform(self.logsigs)

    def train(self, n_epochs=10000):
        self.generator.train(self.logsigs_norm, data_cond=None, n_epochs=n_epochs)

    def generate(self, n_samples=None, normalised=False):
        generated = self.generator.generate(cond=None, n_samples=n_samples)

        if normalised:
            return generated

        if n_samples is None:
            return self.scaler.inverse_transform(generated.reshape(1, -1))[0]

        return self.scaler.inverse_transform(generated)
