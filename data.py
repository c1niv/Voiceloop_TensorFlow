# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm


class NpzFolder:
    NPZ_EXTENSION = 'npz'

    def __init__(self, root, single_spkr=False):
        self.root = root
        self.npzs = self.make_dataset(self.root)

        if len(self.npzs) == 0:
            raise(RuntimeError("Found 0 npz in subfolders of: " + root + "\n"
                               "Supported image extensions are: " +
                               self.NPZ_EXTENSION))

        if single_spkr:
            self.speakers = defaultdict(lambda: 0)
        else:
            self.speakers = []
            for fname in self.npzs:
                self.speakers += [os.path.basename(fname).split('_')[0]]
            self.speakers = list(set(self.speakers))
            self.speakers.sort()
            self.speakers = {v: i for i, v in enumerate(self.speakers)}

        code2phone = np.load(self.npzs[0])['code2phone']
        self.dict = {v: k for k, v in enumerate(code2phone)}

    def __getitem__(self, index):
        path = self.npzs[index]
        txt, feat, spkr = self.loader(path)

        return txt, feat, self.speakers[spkr]

    def __len__(self):
        return len(self.npzs)

    def make_dataset(self, dir):
        images = []

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.NPZ_EXTENSION in fname:
                    path = os.path.join(root, fname)
                    images.append(path)

        return images

    def loader(self, path):
        feat = np.load(path)
        txt = feat['phonemes'].astype('int64')
        audio = feat['audio_features']
        spkr = os.path.basename(path).split('_')[0]

        return txt, audio, self.speakers[spkr]

    def remove_too_long_seq(self, max_seq_len):
        # removes due to memory restrictions
        new_npzs = []
        npzs_len = len(self)
        print('\x1b[0;31;40m'+'removing sequences longer than ' + str(max_seq_len) +
              ' from database due to memory restrictions'+'\x1b[0m\n')
        for filepath in tqdm(self.npzs, total=npzs_len, unit="files",
                             desc='files parsed'):
            if len(np.load(filepath)['audio_features']) < max_seq_len:
                new_npzs.append(filepath)
        print('\x1b[0;31;40m' + 'database contains: ' + str(len(new_npzs)) + ' items' + '\x1b[0m')
        self.npzs = new_npzs


class TBPTTIter:
    """
    Iterator for truncated batch propagation through time(tbptt) training.
    Target sequence is segmented while input sequence remains the same.
    """
    def __init__(self, src, trgt, spkr, seq_len):
        self.seq_len = seq_len
        self.start = True

        self.speakers = spkr
        self.srcBatch = src[0]
        self.srcLenths = src[1]

        # split batch
        ind = list(range(self.seq_len, (-(-trgt[0].shape[0] // self.seq_len)) * self.seq_len, self.seq_len))
        self.tgtBatch = list(np.split(trgt[0], ind, axis=0))
        self.tgtBatch.reverse()
        self.len = len(self.tgtBatch)

        # split length list
        batch_seq_len = len(self.tgtBatch)
        self.tgtLengths = [self.split_length(l, batch_seq_len) for l in trgt[1]]
        self.tgtLengths = np.stack(self.tgtLengths)
        self.tgtLengths = list(np.split(self.tgtLengths, self.tgtLengths.shape[1], axis=1))
        self.tgtLengths = [x.squeeze() for x in self.tgtLengths]
        self.tgtLengths.reverse()

        assert len(self.tgtLengths) == len(self.tgtBatch)

    def split_length(self, seq_size, batch_seq_len):
        seq = [self.seq_len] * (seq_size // self.seq_len)
        if seq_size % self.seq_len != 0:
            seq += [seq_size % self.seq_len]
        seq += [0] * (batch_seq_len - len(seq))
        return np.array(seq)

    def __next__(self):
        if len(self.tgtBatch) == 0:
            raise StopIteration()

        if self.len > len(self.tgtBatch):
            self.start = False

        return (self.srcBatch, self.srcLenths), \
               (self.tgtBatch.pop(), self.tgtLengths.pop()), \
               self.speakers, self.start

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.len

