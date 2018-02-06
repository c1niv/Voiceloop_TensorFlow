import argparse
import os
import tensorflow as tf

from tqdm import tqdm

from data import NpzFolder, TBPTTIter
from batch_creation import Dataset_Iter, make_a_batch
from model import Loop, MaskedMSE
from utils import gradient_check_and_clip, ceil_on_division, global_variable_list

import pickle

######################################################
#                  ARGPARSE - START
######################################################
parser = argparse.ArgumentParser(description='Tensorflow Loop')
# Env options:
parser.add_argument('--epochs', type=int, default=92, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outpath', type=str, default='models', metavar='E',
                    help='Experiment output path')
parser.add_argument('--expName', type=str, default='vctk', metavar='E',
                    help='Experiment name')
parser.add_argument('--data', default='data/vctk',
                    metavar='D', type=str, help='Data path')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
# parser.add_argument('--gpu', default=0,
#                     metavar='G', type=int, help='GPU device ID')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize train and validation loss.')
# Data options
parser.add_argument('--seq-len', type=int, default=200,
                    help='Sequence length for tbptt')
parser.add_argument('--max-seq-len', type=int, default=1000,
                    help='Max sequence length for tbptt')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--clip-grad', type=float, default=0.5,
                    help='maximum norm of gradient clipping')
parser.add_argument('--ignore-grad', type=float, default=10000.0,
                    help='ignore grad before clipping')
# Model options
parser.add_argument('--vocabulary-size', type=int, default=44,
                    help='Vocabulary size')
parser.add_argument('--output-size', type=int, default=63,
                    help='Size of decoder output vector')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='Hidden layer size')
parser.add_argument('--K', type=int, default=10,
                    help='No. of attention guassians')
parser.add_argument('--noise', type=int, default=4,
                    help='Noise level to use')
parser.add_argument('--attention-alignment', type=float, default=0.05,
                    help='# of features per letter/phoneme')
parser.add_argument('--nspk', type=int, default=22,
                    help='Number of speakers')
parser.add_argument('--mem-size', type=int, default=20,
                    help='Memory number of segments')
# Model improvements - tests
parser.add_argument('--act-fcn', type=str, default='relu',
                    help='The activation function')
parser.add_argument('--fix-model', type=bool, default=False,
                    help='The original model had 2 fully connected layers without activation function in between them')
parser.add_argument('--improve-model', type=bool, default=False,
                    help='Added densenet like connection')
# parser.add_argument('--change-attention', type=bool, default=False,
#                     help='change the attention model') needs more tweaking, changed other options

# init
args = parser.parse_args()
expName = args.expName
args.expName = os.path.join(args.outpath, 'checkpoints', args.expName)
os.makedirs(args.expName, exist_ok=True)

pickle.dump(args, open('args.pckl', 'wb'))


######################################################
#                  ARGPARSE - END
######################################################

def main():
    # load datasets
    train_dataset_path = os.path.join(args.data, 'numpy_features')
    train = NpzFolder(train_dataset_path)
    train.remove_too_long_seq(args.max_seq_len)
    train_loader = Dataset_Iter(train, batch_size=args.batch_size)
    train_loader.shuffle()

    valid_dataset_path = os.path.join(args.data, 'numpy_features_valid')
    valid = NpzFolder(valid_dataset_path)
    valid_loader = Dataset_Iter(valid, batch_size=args.batch_size)
    valid_loader.shuffle()

    # train_loader = Dataset_Iter(valid, batch_size=args.batch_size)

    # initiate tensorflow model
    input0 = tf.placeholder(tf.int64, [None, None])
    input1 = tf.placeholder(tf.float32, [None])  # contains length of sentence
    speaker = tf.placeholder(tf.int32, [None, 1])  # speaker identity
    target0 = tf.placeholder(tf.float32, [None, None, 63])
    target1 = tf.placeholder(tf.float32, [None])  # apparently speaker identity
    # idente  = tf.placeholder(tf.float32, [None,256])
    # s_t = tf.placeholder(tf.float32, [64,319,20])
    # mu_t = tf.placeholder(tf.float32, [64,10])
    # context  = tf.placeholder(tf.float32, [64,64,256])
    start = tf.placeholder(tf.bool, shape=(), name='start_new_batch')
    train_flag = tf.placeholder(tf.bool, shape=(), name='train_flag')
    # out_seq = tf.placeholder(tf.float32, [None, None, 63])
    # attns_seq = tf.placeholder(tf.float32, [None, None, 63])

    model = Loop(args)

    # Define loss and optimizer
    output, attns = model.forward(input0, speaker, target0, start, train_flag)
    loss_op = MaskedMSE(output, target0, target1)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    train_op, clip_flag = gradient_check_and_clip(loss_op, optimizer, args.clip_grad, args.ignore_grad)
    merged = tf.summary.merge_all()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(global_variable_list)
    load_model = not args.checkpoint == ''
    save_model = True
    best_eval = float('inf')
    sess_idx = 0
    train_losses = []
    valid_losses = []
    with tf.Session() as sess:
        # Run the initializer

        train_writer = tf.summary.FileWriter("%s/%s/train" % (args.outpath, expName), sess.graph)
        valid_writer = tf.summary.FileWriter("%s/%s/valid" % (args.outpath, expName), sess.graph)

        # Restore variables from disk.
        sess.run(init)
        if load_model:
            saver.restore(sess, args.checkpoint)
            print("Model restored from file: %s" % args.checkpoint)

        for epoch in range(args.epochs):
            train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch,
                              total=ceil_on_division(len(train_loader), args.batch_size))
            # Train data
            for batch_ind in train_enum:
                batch_loss_list = []
                (srcBatch, srcLengths), (tgtBatch, tgtLengths), full_spkr = \
                    make_a_batch(train_loader.dataset, batch_ind)
                batch_iter = TBPTTIter((srcBatch, srcLengths), (tgtBatch, tgtLengths), full_spkr, args.seq_len)
                for (srcBatch, srcLenths), (tgtBatch, tgtLengths), spkr, start2 in batch_iter:
                    loss, _, clip_flag1, summary = sess.run([loss_op, train_op, clip_flag, merged],
                                                            feed_dict={input0: srcBatch, speaker: spkr,
                                                                       target0: tgtBatch, target1: tgtLengths,
                                                                       start: start2, train_flag: True})
                    train_writer.add_summary(summary, sess_idx)
                    sess_idx += 1
                    if not clip_flag1:
                        batch_loss_list.append(loss)
                    else:
                        print('-') # if too many - appear, there are exploding gradients
                train_losses.append(batch_loss_list)
                if len(batch_loss_list) != 0:
                    batch_loss = sum(batch_loss_list)/len(batch_loss_list)
                    batch_loss_list.append(batch_loss)
                else:
                    batch_loss = -1.
                train_enum.set_description('Train (loss %.2f) epoch %d' %
                                           (batch_loss, epoch))
                train_enum.update(srcBatch.shape[0])

            # Validate data
            valid_enum = tqdm(valid_loader, desc='Validating epoch %d' % epoch,
                              total=ceil_on_division(len(valid_loader), args.batch_size))
            batch_loss_list = []
            for batch_ind in valid_enum:
                (srcBatch, srcLengths), (tgtBatch, tgtLengths), full_spkr = \
                    make_a_batch(valid_loader.dataset, batch_ind)

                loss, summary = sess.run([loss_op, merged],
                                         feed_dict={input0: srcBatch, speaker: full_spkr,
                                                    target0: tgtBatch, target1: tgtLengths,
                                                    start: True, train_flag: False})
                batch_loss_list.append(loss)
                train_enum.set_description('Train (loss %.2f) epoch %d' %
                                           (loss, epoch))
                valid_writer.add_summary(summary, sess_idx)
                sess_idx += 1
                valid_enum.set_description('Validating (loss %.2f) epoch %d' %
                                           (loss, epoch))
            if len(batch_loss_list) != 0:
                valid_losses.append(batch_loss_list)
                valid_loss = sum(batch_loss_list)/len(batch_loss_list)
            else:
                valid_loss = 99999.
            if valid_loss < best_eval and save_model:
                best_eval = valid_loss
                save_path = saver.save(sess, "%s/bestmodel.ckpt" % args.expName)
                print("NEW BEST MODEL!, model saved in file: %s" % save_path)
            print('Final validation loss for epoch %d is: %.2f' % (epoch, valid_loss))
            train_loader.shuffle()
            valid_loader.shuffle()

        if save_model:
            save_path = saver.save(sess, "%s/model.ckpt" % args.expName)
            print("Model saved in file: %s" % save_path)

        train_writer.close()
        valid_writer.close()


if __name__ == '__main__':
    main()
