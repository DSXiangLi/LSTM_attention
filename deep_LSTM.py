# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import copy
import numpy
import os
import random
import timeit

import theano
from theano import tensor as T


from project_source import check_dir,shuffle,contextwin,conlleval,load_data

def adagrad(params, gparams, learning_rate = 0.1, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(numpy.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        acc_new = acc + g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g / T.sqrt(acc_new + epsilon)))
    return updates

# deep 2-layer lstm class
class LSTM_2(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh1, nh2, nc, ne, de, cs, normal=True, window_size=3,longdependence=None ):
        
        """Initialize the parameters for the RNNSLU
        :param nh: dimension of the hidden layer - # hidden neurons
        :param nc: number of classes -  output class 
        :param ne: number of word embeddings in the vocabulary - # vocabulary
        :param de: dimension of the word embeddings - vector size 
        :param cs: word window context size - con etxt window
        :param normal: normalize word embeddings after each update or no
        """
        # parameters of the model
        self.window_size =window_size
        ### emb the embedding matrix for all the vocabulary
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        ### weight for input 
        self.wxi1 = theano.shared(name='wxi1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh1))
                                .astype(theano.config.floatX))
        self.wxi2 = theano.shared(name='wxi2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh2))
                                .astype(theano.config.floatX))
        self.wxf1 = theano.shared(name='wxf1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh1))
                                .astype(theano.config.floatX))
        self.wxf2 = theano.shared(name='wxf2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh2))
                                .astype(theano.config.floatX))
        self.wxc1 = theano.shared(name='wxc1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh1))
                                .astype(theano.config.floatX))
        self.wxc2 = theano.shared(name='wxc2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh2))
                                .astype(theano.config.floatX))           
        self.wxo1 = theano.shared(name='wxo1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh1))
                                .astype(theano.config.floatX))
        self.wxo2 = theano.shared(name='wxo2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh2))
                                .astype(theano.config.floatX))      
                                
        ### weight for t-1 hidden layer 
        self.whi1 = theano.shared(name='whi1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh1))
                                .astype(theano.config.floatX))
        self.whi2 = theano.shared(name='whi2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh2, nh2))
                                .astype(theano.config.floatX))
        self.whf1 = theano.shared(name='whf1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh1))
                                .astype(theano.config.floatX))
        self.whf2 = theano.shared(name='whf2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh2, nh2))
                                .astype(theano.config.floatX))        
        self.whc1 = theano.shared(name='whc1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh1))
                                .astype(theano.config.floatX))
        self.whc2 = theano.shared(name='whc2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh2, nh2))
                                .astype(theano.config.floatX))
        self.who1 = theano.shared(name='who1',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh1, nh1))
                                .astype(theano.config.floatX)) 
        self.who2 = theano.shared(name='who2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh2, nh2))
                                .astype(theano.config.floatX))  
                                
        ### weight for memeory cell [diagonal matrix]
        self.wci1 = theano.shared(name='wci1',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh1))
                                .astype(theano.config.floatX))
        self.wci2 = theano.shared(name='wci2',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh2))
                                .astype(theano.config.floatX))
        self.wcf1 = theano.shared(name='wcf1',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh1))
                                .astype(theano.config.floatX))
        self.wcf2 = theano.shared(name='wcf2',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh2))
                                .astype(theano.config.floatX))
        self.wco1 = theano.shared(name='wco1',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh1))
                                .astype(theano.config.floatX))
        self.wco2 = theano.shared(name='wco2',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh2))
                                .astype(theano.config.floatX))
   
        ### weight for the output layer 
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh2, nc))
                               .astype(theano.config.floatX))
        
        ### bias
        self.bi1 = theano.shared(name='bi1',
                                value=numpy.zeros(nh1,
                                dtype=theano.config.floatX))
        self.bi2 = theano.shared(name='bi2',
                                value=numpy.zeros(nh2,
                                dtype=theano.config.floatX))
        self.bf1 = theano.shared(name='bf1',
                                value=numpy.zeros(nh1,
                                dtype=theano.config.floatX))
        self.bf2 = theano.shared(name='bf2',
                                value=numpy.zeros(nh2,
                                dtype=theano.config.floatX))
        self.bc1 = theano.shared(name='bc1',
                                value=numpy.zeros(nh1,
                                dtype=theano.config.floatX))
        self.bc2 = theano.shared(name='bc2',
                                value=numpy.zeros(nh2,
                                dtype=theano.config.floatX))
        self.bo1 = theano.shared(name='bo1',
                                value=numpy.zeros(nh1,
                                dtype=theano.config.floatX))
        self.bo2 = theano.shared(name='bo2',
                                value=numpy.zeros(nh2,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        
        ### Initialization for recurrence
        self.h01 = theano.shared(name='h01',
                                value=numpy.zeros(nh1,
                                dtype=theano.config.floatX))
        self.c01 = theano.shared(name='c01',
                        value=numpy.zeros(nh1,
                        dtype=theano.config.floatX))
        self.h02 = theano.shared(name='h02',
                                value=numpy.zeros(nh2,
                                dtype=theano.config.floatX))
        self.c02 = theano.shared(name='c02',
                        value=numpy.zeros(nh2,
                        dtype=theano.config.floatX))
        
        # bundle
        self.params = [self.wxi1,self.wxf1,self.wxc1,self.wxo1 ,\
                       self.wxi2,self.wxf2,self.wxc2,self.wxo2 ,\
                       self.whi1,self.whf1,self.whc1,self.who1 ,\
                       self.whi2,self.whf2,self.whc2,self.who2 ,\
                       self.wco1, self.wco2,\
                       self.bi1,self.bf1,self.bc1,self.bo1 ,\
                       self.bi2,self.bf2,self.bc2,self.bo2 ,\
                       self.w,self.b, self.h01,self.c01,\
                       self.h02,self.c02]
                       
        # word embeding: use vector of [de] to represent each wrod [trained parameter]
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels [ column ]
     
        def recurrence(x_t, h_tm1, h_tm2, c_tm1, c_tm2):
            i_t1 = T.nnet.sigmoid( T.dot(x_t, self.wxi1) + T.dot(h_tm1, self.whi1) + T.dot(c_tm1, self.wci1) + self.bi1 )
            f_t1 = T.nnet.sigmoid( T.dot(x_t, self.wxf1) + T.dot(h_tm1, self.whf1) + T.dot(c_tm1, self.wcf1) + self.bf1 )
            
            c_t1 = T.tanh(T.dot(x_t, self.wxc1) + T.dot(h_tm1, self.whc1) + self.bc1)
            c_t1 = f_t1 * c_tm1+ i_t1 * c_t1
            
            o_t1 = T.nnet.sigmoid( T.dot(x_t, self.wxo1) + T.dot(h_tm1, self.who1) + T.dot(c_t1, self.wco1) + self.bo1 )
            
            h_t1 = o_t1 * T.tanh(c_t1) 


            i_t2 = T.nnet.sigmoid( T.dot(h_t1, self.wxi2) + T.dot(h_tm2, self.whi2) + T.dot(c_tm2, self.wci2) + self.bi2 )
            f_t2 = T.nnet.sigmoid( T.dot(h_t1, self.wxf2) + T.dot(h_tm2, self.whf2) + T.dot(c_tm2, self.wcf2) + self.bf2 )
            
            c_t2 = T.tanh(T.dot(h_t1, self.wxc2) + T.dot(h_tm2, self.whc2) + self.bc2)
            c_t2 = f_t2 * c_tm2+ i_t2 * c_t2
            
            o_t2 = T.nnet.sigmoid( T.dot(h_t1, self.wxo2) + T.dot(h_tm2, self.who2) + T.dot(c_t2, self.wco2) + self.bo2 )
            
            h_t2 = o_t2 * T.tanh(c_t2)



            s_t = T.nnet.softmax(T.dot(h_t2, self.w) + self.b)





            return [h_t1, h_t2, c_t1, c_t2, s_t]

        #x.shape[0] number of words: # samples
        [h_t1, h_t2, c_t1, c_t2, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h01, self.h02, self.c01, self.c02, None],     
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        
        ## used for SGD
        sentence_gradients = T.grad(sentence_nll, self.params)
        #sentence_updates = OrderedDict((p, p - lr*g)
        #                               for p, g in
        #                               zip(self.params, sentence_gradients))
        sentence_updates = OrderedDict(adagrad(params=self.params, gparams=sentence_gradients, learning_rate = lr, epsilon = 1e-6))

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):
         ## add contect window and change into col[time-sequence]
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y
        
        self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()
            
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))

def test_rnnslu(**kwargs):
    """
    Wrapper function for training and testing RNNSLU

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """
    # process input arguments
    param = {
        'fold': 3,
        'lr': 0.1,
        'verbose': True,
        'decay': False,
        'win': 3,
        'nhidden1': 200,
        'nhidden2': 300,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 60,
        'savemodel': False,
        'normal': False,
        'folder':'../result',
        'longdependence':None
    }
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # create result folder if not exists
    check_dir(param['folder'])

    # load the dataset
    print('... loading the dataset')
    train_set, valid_set, test_set, dic = load_data(param['fold'])

    # create mapping from index to label, and index to word
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items()) # change label2index - index2label
    idx2word = dict((k, v) for v, k in dic['words2idx'].items()) # change words2index - index2words

    # unpack dataset
    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    vocsize = len(dic['words2idx']) # # of words
    nclasses = len(dic['labels2idx']) # # of classes 
    nsentences = len(train_lex) #  # training sample [a batch is all the words in a sentence]

    ## get the label for (input,output) for test and valid set 
    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    print('... building the model')
    lstm = LSTM_2(
        nh1=param['nhidden1'],
        nh2=param['nhidden2'],
        nc=nclasses,
        ne=vocsize,
        de=param['emb_dimension'],
        cs=param['win'],
        normal=param['normal'],
        longdependence = param['longdependence']
        )

    ## build the model for mini-batch
    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    
    for epoch in range(param['nepochs']):

        param['ce'] = epoch
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            lstm.train(x, y, param['win'], param['clr'])
            print('[learning] epoch %i >> %2.2f%%' % (
                epoch, (i + 1) * 100. / nsentences), end=' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), end='')

        # evaluation // back into the real world : idx -> words
        predictions_test = [map(lambda x: idx2label[x],
                            lstm.classify(numpy.asarray(
                            contextwin(x, param['win'])).astype('int32')))
                            for x in test_lex]
        predictions_valid = [map(lambda x: idx2label[x],
                             lstm.classify(numpy.asarray(
                             contextwin(x, param['win'])).astype('int32')))
                             for x in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             param['folder'] + '/current.test.txt',
                             param['folder'])
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              param['folder'] + '/current.valid.txt',
                              param['folder'])

        if res_valid['f1'] > best_f1:

            if param['savemodel']:
                lstm.save(param['folder'])

            #best_lstm = copy.deepcopy(lstm)
            best_f1 = res_valid['f1']

            if param['verbose']:
                print('NEW BEST: epoch', epoch,
                      'valid F1', res_valid['f1'],
                      'best test F1', res_test['f1'])

            param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            param['be'] = epoch
            os.rename(param['folder'] + '/current.test.txt',
                      param['folder'] + '/best.test.txt')
            os.rename(param['folder'] + '/current.valid.txt',
                      param['folder'] + '/best.valid.txt')
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            #lstm = best_lstm

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['vf1'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])
    
    return lstm


if __name__ == '__main__':
    test_rnnslu()
