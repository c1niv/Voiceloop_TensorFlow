import tensorflow as tf
import numpy as np

# based on
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Dataset.py
def batchify(data):
    out, lengths = None, None

    # lengths = [x.size(0) for x in data]
    data = [np.array(x) for x in data]

    lengths = [x.shape[0] for x in data]
    max_length = max(lengths)

    if data[0].ndim == 1:
        out = np.zeros((len(data), max_length), dtype=data[0].dtype)
        for i in range(len(data)):
            data_length = data[i].shape[0]
            out[i][0:data_length] = data[i]
    else:
        feat_size = data[0].shape[1]
        out = np.zeros((len(data), max_length, feat_size), dtype=data[0].dtype)
        # out = data[0].new(len(data), max_length, feat_size).fill_(0)
        for i in range(len(data)):
            data_length = data[i].shape[0]
            out[i][0:data_length] = data[i]

    return out, lengths


def make_a_batch(npzFolder, indexes: list):
    full_txt  = []
    full_feat = []
    full_spkr = []
    for ind in indexes:
        txt, audio, spkr = npzFolder.loader(npzFolder.npzs[ind])
        full_txt.append(txt)
        full_feat.append(audio)
        full_spkr.append(spkr)
    srcBatch, srcLengths = batchify(full_txt)
    tgtBatch, tgtLengths = batchify(full_feat)
    # arrange dimension order properly for faster tensorflow run on gpu
    srcBatch = srcBatch.transpose(1, 0)  # TODO: check later if .ascontiguousarray should be added
    tgtBatch = tgtBatch.transpose(1, 0, 2)
    return (srcBatch, srcLengths), (tgtBatch, tgtLengths), np.array([full_spkr]).transpose(1, 0)

class Dataset_Iter:
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.batch_size = batch_size
        self.sampleorder = list(range(self.dataset_len))
        self.index = 0

    def sample(self, n=None):

        if n is None:
            n = self.batch_size
        n = max(n, 0)

        if self.index+n > self.dataset_len:
            n = self.dataset_len - self.index

        sampled = self.sampleorder[self.index:n+self.index]
        if len(sampled) == 0:
            raise StopIteration()

        self.index += n
        return sampled

    __next__ = sample

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def shuffle(self):
        self.index = 0
        np.random.shuffle(self.sampleorder)
