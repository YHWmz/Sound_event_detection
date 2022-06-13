from typing import List, Union
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from h5py import File
from tqdm import tqdm
import torch.utils.data as tdata


def load_dict_from_csv(file, cols, sep="\t"):
    if isinstance(file, str):
        df = pd.read_csv(file, sep=sep)
    elif isinstance(file, pd.DataFrame):
        df = file
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


class InferenceDataset(tdata.Dataset):
    def __init__(self,
                 audio_file):
        super(InferenceDataset, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aids = list(self.aid_to_h5.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        return aid, feat


class TrainDataset(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx):
        super(TrainDataset, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]

        return aid, feat, target

import random
import copy

class TrainDataset_timeshift(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx):
        super(TrainDataset_timeshift, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.aids)*2

    def __getitem__(self, raw_index):
        index = raw_index // 2
        is_shift = raw_index % 2
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        if is_shift:
            shift_T = random.randint(20, 100)
            tmp_feat = copy.deepcopy(feat)
            if random.randint(0,1):
                feat[shift_T:, :] = tmp_feat[0:tmp_feat.shape[0] - shift_T, :]
                feat[0:shift_T, :] = tmp_feat[tmp_feat.shape[0] - shift_T:, :]
            else:
                feat[0:feat.shape[0] - shift_T, :] = tmp_feat[shift_T:, :]
                feat[feat.shape[0] - shift_T:, :] = tmp_feat[0:shift_T, :]

        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]

        return aid, feat, target

class TrainDataset_freshift(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx):
        super(TrainDataset_freshift, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.aids)*2

    def __getitem__(self, raw_index):
        index = raw_index // 2
        is_shift = raw_index % 2
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        if is_shift:
            shift_F = random.randint(0, 4)
            tmp_feat = copy.deepcopy(feat)
            if random.randint(0,1):
                feat[:, shift_F:] = tmp_feat[:, 0:tmp_feat.shape[1]-shift_F]
                feat[:, 0:shift_F] = tmp_feat[:, tmp_feat.shape[1]-shift_F:]
            else:
                feat[:, 0:feat.shape[1]-shift_F] = tmp_feat[:, shift_F:]
                feat[:, feat.shape[1]-shift_F:] = tmp_feat[:, 0:shift_F]

        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]

        return aid, feat, target


class TrainDataset_freaug(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx):
        super(TrainDataset_freaug, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx
        self.SpecAug = SpecAug()

    def __len__(self):
        return len(self.aids)*2

    def __getitem__(self, raw_index):
        index = raw_index // 2
        is_shift = raw_index % 2
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        if is_shift:
            shift_F = random.randint(0, 4)
            tmp_feat = copy.deepcopy(feat)
            if random.randint(0,1):
                feat[:, shift_F:] = tmp_feat[:, 0:tmp_feat.shape[1]-shift_F]
                feat[:, 0:shift_F] = tmp_feat[:, tmp_feat.shape[1]-shift_F:]
            else:
                feat[:, 0:feat.shape[1]-shift_F] = tmp_feat[:, shift_F:]
                feat[:, feat.shape[1]-shift_F:] = tmp_feat[:, 0:shift_F]

        a = np.random.rand(1)
        if a > 0.5:
            feat = self.SpecAug._ApplySpecAugment(feat)
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]

        return aid, feat, target


# import torchaudio.transforms as T
# import numpy as np
#
# class TrainDataset_timestrech(tdata.Dataset):
#     def __init__(self,
#                  audio_file,
#                  label_file,
#                  label_to_idx):
#         super(TrainDataset_timestrech, self).__init__()
#         self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
#         self.cache = {}
#         self.aid_to_label = load_dict_from_csv(label_file,
#             ("filename", "event_labels"))
#         self.aids = list(self.aid_to_label.keys())
#         first_aid = self.aids[0]
#         with File(self.aid_to_h5[first_aid], 'r') as store:
#             self.datadim = store[first_aid].shape[-1]
#         self.label_to_idx = label_to_idx
#         self.stretch = T.TimeStretch()
#
#     def __len__(self):
#         return len(self.aids)*2
#
#     def __getitem__(self, raw_index):
#         index = raw_index // 2
#         is_strctch = raw_index % 2
#         aid = self.aids[index]
#         h5_file = self.aid_to_h5[aid]
#         if h5_file not in self.cache:
#             self.cache[h5_file] = File(h5_file, 'r', libver='latest')
#         feat = self.cache[h5_file][aid][()]
#         feat = torch.as_tensor(feat).float()
#         if is_strctch:
#             T = feat.shape[0]
#             rate = np.random.random() * 0.3 + 0.7
#             tmp_feat[0] = self.stretch(feat.reshape[1, feat.shape[0], feat.shape[1]], rate)
#             print(tmp_feat.shape)
#             feat = feat*0
#             feat[0:tmp_feat.shape[0],:] = tmp_feat
#         label = self.aid_to_label[aid]
#         target = torch.zeros(len(self.label_to_idx))
#         for l in label.split(","):
#             target[self.label_to_idx[l]] = 1
#         # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]
#
#         return aid, feat, target

class TrainDataset_specaug(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx):
        super(TrainDataset_specaug, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx
        self.SpecAug = SpecAug()

    def __len__(self):
        return len(self.aids)*2

    def __getitem__(self, raw_index):
        index = raw_index//2
        is_SpecAug = raw_index%2
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float() #[501, 64]
        if is_SpecAug:
            feat = self.SpecAug._ApplySpecAugment(feat)
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]

        return aid, feat, target

class TrainDataset_mix(tdata.Dataset):
    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx):
        super(TrainDataset_mix, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]
        self.label_to_idx = label_to_idx
        self.Mixer = SpecAug()

    def __len__(self):
        return len(self.aids)*2

    def __getitem__(self, raw_index):
        index = raw_index//2
        is_Mix = raw_index%2
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float() #[501, 64]
        if feat.shape[0] < 501:
            tmp_feat = torch.zeros([501, feat.shape[1]])
            tmp_feat[0:feat.shape[0], :] = feat
            feat = tmp_feat
        if is_Mix:
            mix_idx = random.randint(0, self.__len__()//2-1)
            mix_aid = self.aids[mix_idx]
            mix_h5_file = self.aid_to_h5[mix_aid]
            if mix_h5_file not in self.cache:
                self.cache[mix_h5_file] = File(mix_h5_file, 'r', libver='latest')
            mix_feat = self.cache[mix_h5_file][mix_aid][()]
            mix_feat = torch.as_tensor(mix_feat).float()  # [501, 64]
            if mix_feat.shape[0] < 501:
                tmp_feat = torch.zeros([501, mix_feat.shape[1]])
                tmp_feat[0:mix_feat.shape[0], :] = mix_feat
                mix_feat = tmp_feat

            feat = feat*0.5 + mix_feat*0.5
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        if is_Mix:
            label = self.aid_to_label[mix_aid]
            for l in label.split(","):
                target[self.label_to_idx[l]] = 1
        # audio文件名，对应的feature，multi-hot的向量（但是没有时间轴） [501, 64] [10]
        return aid, feat, target


import random
import numpy as np

class SpecAug():
    def __init__(self, _nb_mel_bins=64):
        self._specaug = {"mF": 1,
                         "mT": 1,
                         "F_prop": 0.08,
                         "T_prop": 0.08}
        self._nb_mel_bins = _nb_mel_bins

    def _ApplySpecAugment(self, feat):
        '''
            feat.shape: [T, F]
        '''

        for i in range(self._specaug['mF']):
            f = random.randint(0, np.round(self._specaug['F_prop'] * feat.shape[-1]))
            if f != 0:
                f0 = random.randint(0, 64 - 1 - f)
                feat[:, f0: f0 + f] = 0
        for i in range(self._specaug['mT']):
            t = random.randint(0, np.round(self._specaug['T_prop'] * feat.shape[0]))
            if t != 0:
                t0 = random.randint(0, feat.shape[0] - 1 - t)
                feat[t0: t0 + t, :] = 0

        return feat

def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    if isinstance(tensorlist[0], np.ndarray):
        tensorlist = [torch.as_tensor(arr) for arr in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    length = [tensor.shape[0] for tensor in tensorlist]
    return padded_seq, length


def sequential_collate(return_length=True, length_idxs: List=[]):
    def wrapper(batches):
        seqs = []
        lens = []
        for idx, data_seq in enumerate(zip(*batches)):
            if isinstance(data_seq[0],
                          (torch.Tensor, np.ndarray)):  # is tensor, then pad
                data_seq, data_len = pad(data_seq)
                if idx in length_idxs:
                    lens.append(data_len)
            else:
                data_seq = np.array(data_seq)
            seqs.append(data_seq)
        if return_length:
            seqs.extend(lens)
        return seqs
    return wrapper
