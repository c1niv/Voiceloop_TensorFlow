# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import phonemizer
import string
import pickle

import tensorflow as tf

from model import Loop
from data import NpzFolder
from utils import generate_merlin_wav, global_variable_list


parser = argparse.ArgumentParser(description='PyTorch Phonological Loop \
                                    Generation')
parser.add_argument('--npz', type=str, default='',
                    help='Dataset sample to generate.')
parser.add_argument('--text', default='',
                    type=str, help='Free text to generate.')
parser.add_argument('--spkr', default=0,
                    type=int, help='Speaker id.')
parser.add_argument('--checkpoint', default='checkpoints/vctk/lastmodel.pth',
                    type=str, help='Model used for generation.')
# init
args = parser.parse_args()

def text2phone(text, char2code):
    seperator = phonemizer.separator.Separator('', '', ' ')
    ph = phonemizer.phonemize(text, separator=seperator)
    ph = ph.split(' ')
    ph.remove('')

    result = [char2code[p.encode()] for p in ph]
    print(result)
    return np.array(result)


def trim_pred(out, attn):
    tq = abs(attn).sum(1)
    for stopi in range(1, tq.shape[0]):
        col_sum = abs(attn[:stopi, :]).sum(0)
        if tq[stopi] < 0.5 and col_sum[-1] > 4:
            break

    out = out[:stopi, :]
    attn = attn[:stopi, :]

    return out, attn


def npy_loader_phonemes(path):
    feat = np.load(path)

    txt = feat['phonemes'].astype('int64')

    audio = feat['audio_features']

    return txt, audio


def main():
    train_args = pickle.load(open('args.pckl', 'rb'))
    train_args.batch_size = 1
    train_args.noise = 0

    train_dataset = NpzFolder(train_args.data + '/numpy_features')
    char2code = train_dataset.dict
    spkr2code = train_dataset.speakers
    norm_path = train_args.data + '/norm_info/norm.dat'

    if args.spkr not in range(len(spkr2code)):
        print('ERROR: Unknown speaker id: %d.' % args.spkr)
        return
    train_args.improve_model = True
    train_args.act_fcn = 'relu'
    train_args.fix_model = True
    model = Loop(train_args)
    input0 = tf.placeholder(tf.int64, [None, None])
    speaker = tf.placeholder(tf.int32, [None, 1])  # speaker identity
    target0 = tf.placeholder(tf.float32, [None, None, 63])
    start = tf.placeholder(tf.bool, shape=(), name='start_new_batch')
    train_flag = tf.placeholder(tf.bool, shape=(), name='train_flag')  # in generator case = False
    output, attns = model.forward(input0, speaker, target0, start, train_flag)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # overwrite initial values with trained values
    saver = tf.train.Saver(global_variable_list)

    txt, feat, spkr, output_fname = None, None, None, None
    if args.npz is not '':
        txt, feat = npy_loader_phonemes(args.npz)

        txt = np.expand_dims(txt, 1)
        feat = np.expand_dims(feat, 1)
        spkr = np.array([[args.spkr]])
        fname = os.path.basename(args.npz)[:-4]
        output_fname = fname + '.gen_' + str(args.spkr)

    elif args.text is not '':
        txt = text2phone(args.text, char2code)
        feat = np.random.randn(txt.shape[0]*20, 63)*1e-38
        spkr = args.spkr

        txt = np.expand_dims(txt,1)
        feat = np.expand_dims(feat,1)
        spkr = np.array([[spkr]])

        # slugify input string to file name
        fname = args.text.replace(' ', '_')
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        fname = ''.join(c for c in fname if c in valid_chars)
        output_fname = fname + '.gen_' + str(args.spkr)

    else:
        print('ERROR: Must supply npz file path or text as source.')
        return

    # Restore variables from disk.    #model.load_state_dict(weights)###
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, args.checkpoint)
        print("Model restored.")
        out, attn = sess.run([output, attns],
                             feed_dict={input0: txt, speaker: spkr,
                                        target0: feat,
                                        start: True, train_flag: False})
    print("Tensorflow finished")
    # python post processing
    out, attn = trim_pred(out, attn)
    # save out data for post processing in external tools
    output_dir = os.path.join(os.path.dirname(args.checkpoint), 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # run external tools

    files_to_write = []

    files_to_write.append([out,
                           output_dir,
                           output_fname,
                           norm_path])
    if args.npz is not '':
        files_to_write.append([feat[:, 0, :],
                               output_dir,
                               os.path.basename(args.npz)[:-4] + '.orig',
                               norm_path])

    pickle.dump(files_to_write, open("tmp_out.pckl", "wb"), protocol=2)

    # generate_merlin_wav(out,
    #                     output_dir,
    #                     output_fname,
    #                     norm_path)
    # # if npz was supplied, generate as well the original "bad" TTS
    # if args.npz is not '':
    #     output_orig_fname = os.path.basename(args.npz)[:-4] + '.orig'
    #     generate_merlin_wav(feat[:, 0, :],
    #                         output_dir,
    #                         output_orig_fname,
    #                         norm_path)


if __name__ == '__main__':
    main()
