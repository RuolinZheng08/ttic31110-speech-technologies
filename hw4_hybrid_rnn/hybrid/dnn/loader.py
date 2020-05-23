"""
Prepares labeled data for a fame classifier starting from utterance level features.

Date: May, 2020
"""

import numpy as np
from operator import itemgetter


class DataLoader(object):
    """
    Prepares data for DNN training
    """
    def __init__(self, data_cfg):
        """
        Set up the loader
        ------
        input:
        data_cfg: dataset config from connfig.json
        """
        self.context_size = data_cfg["context_size"]
        if self.context_size%2 != 0:
            self.context_size += 1
        self.mfccs_dict = np.load(data_cfg["mfcc"], allow_pickle=True)
        self.state_seq_dict = np.load(data_cfg["state_seq"], allow_pickle=True)
        self.n_states = self.state_seq_dict['total_states']
        self.mfcc_dim = self.mfccs_dict['feature_dim']
        
    def flatten_mfcc(self, mfcc, num_datapoints):
        """
        Pads the mfcc matrix taking into account the context size and flattens it.
        ----
        mfcc: MFCC featue matrix corresponding to a single utterance
        num_datapoints: number of frames in the utterance
        ---
        Returns: padded and flattened mfcc features and indices to help with data chunking
        """
        extend_len = int(self.context_size/2)
        extend_array_start = np.tile(mfcc[0], (extend_len, 1))
        extend_array_end = np.tile(mfcc[-1], (extend_len, 1))
        extended_mfcc = np.vstack((extend_array_start, mfcc, extend_array_end))
        mfcc_flatten = np.ndarray.flatten(extended_mfcc)
        start_indices = [idx*self.mfcc_dim for idx in range(num_datapoints)]
        end_indices = [(self.context_size+1+idx)*self.mfcc_dim for idx in range(num_datapoints)]

        return mfcc_flatten, start_indices, end_indices

    def prepare_data(self, split):
        """
        Prepares data
        ----
        split: train/dev/test
        ---
        Returns:
            matrices for features, labels, one-hot labels
            a dictionary mapping utterance ids to frame ids
        """
        if split=="train":
            mfcc_list, state_seq_list = self.mfccs_dict["Xtrain"], self.state_seq_dict["Ytrain"]
        elif split=="dev":
            mfcc_list, state_seq_list = self.mfccs_dict["Xdev"], self.state_seq_dict["Ydev"]
        else:
            mfcc_list, state_seq_list = self.mfccs_dict["Xtest"], self.state_seq_dict["Ytest"]
        utt_num = 0
        feature_mat, label_mat, label_onehot_mat, utt_to_frames = [], [], [], {} 
        current_idx = 0
        for mfcc, state_seq in zip(mfcc_list, state_seq_list):
            utt_to_frames[utt_num] = []
            num_datapoints = len(state_seq)
            mfcc_flatten, start_indices, end_indices = self.flatten_mfcc(mfcc, num_datapoints)
            for i in range(num_datapoints):
                utt_to_frames[utt_num].append(current_idx)
                feature_mat.append([mfcc_flatten[start_indices[i]:end_indices[i]]])
                label_mat.append([state_seq[i]])
                label_onehot = np.zeros(self.n_states)
                label_onehot[state_seq[i]] = 1
                label_onehot_mat.append([label_onehot])
                current_idx += 1
            utt_num += 1

        return np.concatenate(feature_mat, axis=0), np.concatenate(label_mat, axis=0), np.concatenate(label_onehot_mat, axis=0), utt_to_frames

    def get_prior(self):
        """
        Computes the prior probability distribution for the state labels
        ------
        Returns prior probability array of length same as the total number of states
        """
        label_to_count = {}
        tot_labels = 0
        for label_seq in self.state_seq_dict["Ytrain"]:
            for label in label_seq:
                tot_labels += 1
                if label not in label_to_count:
                    label_to_count[label] = 0
                label_to_count[label] += 1
        labels = range(self.n_states)
        prior = np.array(itemgetter(*labels)(label_to_count))/tot_labels
        return prior

