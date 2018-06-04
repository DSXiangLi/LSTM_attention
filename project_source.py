# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 08:15:46 2016

@author: lingxilove
"""

import gzip
import numpy
import os
import pickle
import random
import stat
import subprocess
import urllib

import theano
import theano.tensor as T



def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def shuffle(lol, seed):
    """
    Suffle inplace each list in the same order

    :type lol: list
    :param lol: list of list as input

    :type seed: int
    :param sedd: seed the shuffling

    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def contextwin(l, win):
    """
    Return a list of list of indexes corresponding to context windows
    surrounding each word in the sentence

    :type win: int
    :param win: the size of the window given a list of indexes composing a sentence

    :type l: list or numpy.array
    :param l: array containing the word indexes

    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + + win // 2 * [-1]

    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

    
def conlleval(p, g, w, filename, script_path):
    """
    Evaluate the accuracy using conlleval.pl

    :type p: list
    :param p: predictions

    :type g: list
    :param g: groundtruth

    :type w: list
    :param w: corresponding words

    :type filename: string
    :param filename: name of the file where the predictions are written. It
    will be the input of conlleval.pl script for computing the performance in
    terms of precision recall and f1 score.

    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    """
    Run conlleval.pl perl script to obtain precision/recall and F1 score.

    :type filename: string
    :param filename: path to the file

    """
    _conlleval = os.path.join('./', 'conlleval.pl')
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        print('Downloading conlleval.pl from %s' % url)
        urllib.urlretrieve(url, _conlleval)
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()).encode('utf-8'))
    stdout = stdout.decode('utf-8')
    out = None

    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    # To help debug
    if out is None:
        print(stdout.split('\n'))
    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(foldnum=3):
    '''
    Load the ATIS dataset

    :type foldnum: int
    :param foldnum: fold number of the ATIS dataset, ranging from 0 to 4.

    '''

    filename = 'atis.fold'+str(foldnum)+'.pkl.gz'
    atis_url = 'http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        try:
            new_path = os.path.join(
                os.path.split(__file__)[0],
                ".",
                "data",
                dataset
            )
        except:
            new_path = str(os.getcwd()) + \
                "\\data\\" +\
                dataset

        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (atis_url + dataset)
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    filename = check_dataset(filename)
    f = gzip.open(filename, 'rb')
    try:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set, dicts = pickle.load(f)
        
    return train_set, valid_set, test_set, dicts
