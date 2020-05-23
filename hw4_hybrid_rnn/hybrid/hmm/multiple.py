from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from hybrid.hmm.single import GaussianHMM
import scipy
from glob import glob
import soundfile as sf
from os import path
import numpy as np
import pickle as pkl
import editdistance

np.random.seed(seed=273)

class FullGaussianHMM(object):
    """
    Full Gaussian Hidden Markov Model.
    """
    def __init__(self, Xtrain, single_digit_model_filename):
        """
        Initialize full Gaussian HMM (composition of smaller HMMs)
        ----
        Xtrain: training data
        single_digit_model_filename: filename of saved single digit model to be loaded
        """

        with open(single_digit_model_filename, "rb") as f:
            model = pkl.load(f)

        n_states = model[0].n_states
        n_dims = model[0].n_dims

        N = 10

        self.digit_states_per = n_states
        self.digit_states_total = self.digit_states_per * N

        self.begin_sil = self.digit_states_total
        self.end_sil = self.begin_sil + 1
        self.pause = self.end_sil + 1

        self.start_states = np.arange(0, self.digit_states_total, self.digit_states_per)
        self.stop_states = np.arange(self.digit_states_per - 1, self.digit_states_total, self.digit_states_per)

        self.total = self.pause + 1

        self.n_dims = n_dims

        self.mu = np.zeros((self.total, self.n_dims))
        self.sigma = np.zeros((self.total, self.n_dims))

        self.pi = np.zeros(self.total)
        self.pi[self.begin_sil] = 1.

        self.A = np.zeros((self.total, self.total))

        self.states2digit = dict()
        self.states2nondigit = dict()

        for digit in range(N):
            start = self.start_states[digit]
            stop = self.stop_states[digit]
            self.mu[start:stop + 1] = model[digit].mu
            self.sigma[start:stop + 1] = model[digit].sigma
            self.A[start:stop + 1, start:stop + 1] = model[digit].A
            self.A[stop, stop] = 0.99
            self.A[stop, self.pause] = 0.005
            self.A[stop, self.end_sil] = 0.005
            self.states2digit[tuple(range(start, stop + 1))] = str(digit)

        self.states2nondigit[(self.begin_sil,)] = "START"
        self.states2nondigit[(self.end_sil,)] = "END"
        self.states2nondigit[(self.pause,)] = "PAUSE"

        self.A[self.begin_sil, self.begin_sil] = 0.5
        self.A[self.begin_sil, self.start_states] = 0.05

        self.A[self.end_sil, self.end_sil] = 1.

        self.A[self.pause, self.pause] = 0.5
        self.A[self.pause, self.start_states] = 0.05

        X_concat = np.concatenate([x[int(len(x) / 2)].reshape(1, -1) for x in Xtrain])
        self.mu[self.pause] = X_concat.mean(axis=0)
        self.sigma[self.pause] = X_concat.var(axis=0)

        X_start_concat = np.concatenate([x[:10].reshape(10, -1) for x in Xtrain])
        self.mu[self.begin_sil] = X_start_concat.mean(axis=0)
        self.sigma[self.begin_sil] = X_start_concat.var(axis=0)

        X_end_concat = np.concatenate([x[-10:].reshape(10, -1) for x in Xtrain])
        self.mu[self.end_sil] = X_end_concat.mean(axis=0)
        self.sigma[self.end_sil] = X_end_concat.var(axis=0)

    def get_emissions(self, x):
        """
        log emissions for frames of x
        ----
        x: 2d-array of shape (Tx, 13) made up of MFCCs
        ----
        Returns log_emissions of shape (total number of states, Tx)
        """
        T, _ = x.shape
        log_B = np.full((self.total, T), -np.inf)
        for s in range(self.total):
            log_B[s] = multivariate_normal.logpdf(x, mean=self.mu[s], cov=np.diag(self.sigma[s]))
        log_B[:self.end_sil, T - 1] = -np.inf
        log_B[self.end_sil, T - 1] = 0.
        return log_B

    def forward(self, log_pi, log_A, log_B):
        """
        Forward algorithm.
        ----
        log_pi: starting log probabilities, shape (total number of states)
        log_A: transition log probabilities, shape (total number of states, total number of states)
        log_B: log emission probabilities of an example x
        ----
        Returns (log) alpha
        """
        _, T = log_B.shape
        log_alpha = np.zeros(log_B.shape)
        for t in range(T):
            if t == 0:
                log_alpha[:, t] = log_B[:, t] + log_pi
            else:
                log_alpha[:, t] = log_B[:, t] + logsumexp(log_A.T + log_alpha[:, t - 1], axis=1)
        return log_alpha

    def backward(self, log_A, log_B):
        """
        Backward algorithm.
        ----
        log_A: transition log probabilities, shape (total number of states, total number of states)
        log_B: log emission probabilities of an example x
        ----
        Returns (log) beta
        """
        _, T = log_B.shape
        log_beta = np.zeros(log_B.shape)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                log_beta[:, t] = 0.
            else:
                log_beta[:, t] = logsumexp(log_A + log_B[:, t + 1] + log_beta[:, t + 1], axis=1)
        return log_beta

    def viterbi(self, log_pi, log_A, log_B):
        """
        Viterbi algorithm.
        ----
        log_pi: starting log probabilities, shape (total number of states)
        log_A: transition log probabilities, shape (total number of states, total number of states)
        log_B: log emission probabilities of an example x
        ----
        Returns best state path, log_delta (full lattice)
        """
        _, T = log_B.shape
        log_delta = np.zeros(log_B.shape)
        for t in range(T):
            if t == 0:
                log_delta[:, t] = log_B[:, t] + log_pi
            else:
                log_delta[:, t] = log_B[:, t] + np.max(log_A.T + log_delta[:, t - 1], axis=1)

        q = np.zeros(T, dtype=np.int32)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                q[t] = np.argmax(log_delta[:, t])
                log_prob = log_delta[q[t], t]
            else:
                q[t] = np.argmax(log_delta[:, t] + log_A[:, q[t + 1]])

        return q, log_prob

    def score(self, x, log_emission=None):
        """
        Use forward-backward algorithm to
        compute log probability and posteriors.
        ------
        input:
        x :2d-array of shape (T, 13): MFCCs for a single example
        ------
        output:
        log_prob :scalar: log probability of observed sequence
        log_alpha :2d-array of shape (n_states, T): log prob of getting to state at time t from start
        log_beta :2d-array of shape (n_states, T): log prob of getting from state at time t to end
        gamma :2d-array of shape (n_states, T): state posterior probability
        xi :2d-array of shape (n_states, n_states): state transition probability matrix
        """
        T = len(x)

        log_pi = np.log(self.pi)
        log_A = np.log(self.A)
        if log_emission is not None:
            log_B = log_emission
        else:
            log_B = self.get_emissions(x)

        log_alpha = self.forward(log_pi, log_A, log_B)
        log_beta = self.backward(log_A, log_B)

        log_prob = logsumexp(log_alpha[:, T - 1])
        gamma = np.exp(log_alpha + log_beta - log_prob)

        xi = np.exp(log_alpha[None, :, :-1].T + log_A[None, :, :] + log_B[:, None, 1:].T + log_beta[:, None, 1:].T - log_prob)
        xi = xi.sum(axis=0)
        xi /= xi.sum(axis=1, keepdims=True).clip(1e-1)

        return log_prob, log_alpha, log_beta, gamma, xi

    def train(self, X, Y, log_emission=None):
        """
        Estimate model parameters.
        ------
        input:
        X: list of 2d-arrays of shape (Tx, 13): list of single digit MFCC features
        ------
        update model parameters (A, mu, sigma)
        """
        stats = {
            "gamma": np.zeros((self.total, 1)),
            "A": np.zeros((self.total, self.total)),
            "X": np.zeros((self.total, self.n_dims)),
            "X**2": np.zeros((self.total, self.n_dims))
        }
        
        for ii, (x, y) in enumerate(zip(X, Y)):

            if (ii + 1) % 50 == 0:
                print("{} examples...".format(ii + 1))

            y = np.array([0 if yy == 'o' else int(yy) for yy in y], dtype=np.int32)

            self.A[self.begin_sil, :self.digit_states_total] = 0.
            self.A[self.pause, :self.digit_states_total] = 0.

            self.A[self.begin_sil, self.start_states[y[0]]] = 1. - self.A[self.begin_sil, self.begin_sil]
            self.A[self.pause, self.start_states[y[1]]] = 1. - self.A[self.pause, self.pause]
            
            if log_emission is not None:
                log_prob, log_alpha, log_beta, gamma, xi = self.score(x, log_emission[ii].T)
            else:
                log_prob, log_alpha, log_beta, gamma, xi = self.score(x)

            stats["gamma"] += gamma.sum(axis=1, keepdims=True)
            stats["A"] += xi
            stats["X"] += gamma.dot(x)
            stats["X**2"] += gamma.dot(x**2)

        self.mu = stats["X"] / stats["gamma"]
        self.sigma = stats["X**2"] / stats["gamma"] - self.mu**2

        self.A = np.where(np.bitwise_or(self.A == 0.0, self.A == 1.0), self.A, stats["A"])  # update transition probabilities
        self.A /= self.A.sum(axis=1, keepdims=True)  # normalize transition probabilities
    
    def test(self, X, Y, log_emission=None):
        """
        Evaluate model on (X, Y) measured with word-error-rate (WER).
        ------
        input:
        X: list of 2d-arrays of shape (Tx, 13): list of single digit MFCC features
        Y: digit sequence
        ------
        Returns word-error-rate WER
        """

        self.A[self.begin_sil, self.start_states] = (1. - self.A[self.begin_sil, self.begin_sil]) / 10.
        self.A[self.pause, self.start_states] = 0.1
        
        N = 0
        wer = 0.
        for ii, (x, y) in enumerate(zip(X, Y)):
            log_pi = np.log(self.pi)
            log_A = np.log(self.A)
            if log_emission is not None:
                log_B = log_emission[ii].T
            else:
                log_B = self.get_emissions(x)

            q, log_prob = self.viterbi(log_pi, log_A, log_B)
            # num_states x num_frames
            cur, cur_seq, pred = None, [], []
            for qt in q:
                if cur == qt:
                    continue
                else:
                    cur_seq.append(qt)
                    cur = qt
                    if tuple(cur_seq) in self.states2digit:
                        pred.append(self.states2digit[tuple(cur_seq)])
                        cur_seq = []
                    elif tuple(cur_seq) in self.states2nondigit:
                        cur_seq = []

            N += len(y)
            wer += editdistance.eval(pred, y)

        return (wer / N)
