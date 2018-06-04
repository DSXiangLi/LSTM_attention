# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
#import copy
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
    
class LSTM(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, ne, de, cs, normal=True, longdependence=False,optimization=None):
        
        """Initialize the parameters for the RNNSLU
        :param nh: dimension of the hidden layer - # hidden neurons
        :param nc: number of classes -  output class 
        :param ne: number of word embeddings in the vocabulary - # vocabulary
        :param de: dimension of the word embeddings - vector size 
        :param cs: word window context size - con etxt window
        :param normal: normalize word embeddings after each update or no
        """
        ### emb the embedding matrix for all the vocabulary
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        ### weight for input 
        self.wxi = theano.shared(name='wxi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wxf = theano.shared(name='wxf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wxc = theano.shared(name='wxc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))           
        self.wxo = theano.shared(name='wxo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))    
                                
        ### weight for t-1 hidden layer 
        self.whi = theano.shared(name='whi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.whf = theano.shared(name='whf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))       
        self.whc = theano.shared(name='whc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.who = theano.shared(name='who',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))   
                                
        ### weight for memeory cell [diagonal matrix]
        ## Initialization requirement : initialize as the diagonal matrix. becuase this the proportion of
        ## the memory from history/current state for each neuron. 
        self.wci = theano.shared(name='wci',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh))
                                .astype(theano.config.floatX))
        self.wcf = theano.shared(name='wcf',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh))
                                .astype(theano.config.floatX))
        self.wco = theano.shared(name='wco',
                                value= 0.2 * numpy.diag(numpy.random.uniform(0.0, 1.0,
                                nh))
                                .astype(theano.config.floatX))
   
        ### weight for the output layer 
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        
        ### bias
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bc = theano.shared(name='bc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        
        ### Initialization for recurrence
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                        value=numpy.zeros(nh,
                        dtype=theano.config.floatX))
        
        # bundle
        self.params = [self.wxi,self.wxf,self.wxc,self.wxo ,\
                       self.whi,self.whf,self.whc,self.who ,\
                       self.wco,\
                       self.bi,self.bf,self.bc,self.bo ,\
                       self.w,self.b, self.h0,self.c0]
                       
        # word embeding: use vector of [de] to represent each wrod [trained parameter]
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs)) ## flatten the matrix in to (n_step,dimension * context widnow)
        y_sentence = T.ivector('y_sentence')  # labels [ column ]
     
        ################################## Different recurrent Method #############################
        ########################### AR + Bi-direction LSTM + Attention + None #####################
        if longdependence=='AR':           
            ## goal: give longer dependence on the otuput sequence, by directly using the previous hidden layer 
            ## value on the output layer, in addition to the hidden layer. [Ar[1]-lstm + lag_Moving[3]]
            self.war0 = theano.shared(name='w0',
                       value=numpy.diag(numpy.ones(nc))
                       .astype(theano.config.floatX))
            self.war1 = theano.shared(name='w1',
                                   value=numpy.zeros((nc, nc))
                                   .astype(theano.config.floatX))       
            self.war2 = theano.shared(name='w2',
                                   value=numpy.zeros((nc, nc))
                                   .astype(theano.config.floatX))
            
            self.params = [self.wxi,self.wxf,self.wxc,self.wxo ,\
                       self.whi,self.whf,self.whc,self.who ,\
                       self.wco,\
                       self.bi,self.bf,self.bc,self.bo ,\
                       self.w,self.b, self.h0,self.c0,\
                       self.war0,self.war1,self.war2]     
              
            def recurrence(x_t,h_tm1,h_tm2,c_tm1):
                i_t = T.nnet.sigmoid( T.dot(x_t, self.wxi) + T.dot(h_tm1, self.whi) + T.dot(c_tm1, self.wci) + self.bi )
                f_t = T.nnet.sigmoid( T.dot(x_t, self.wxf) + T.dot(h_tm1, self.whf) + T.dot(c_tm1, self.wcf) + self.bf )
                
                c_t = T.tanh(T.dot(x_t, self.wxc) + T.dot(h_tm1, self.whc) + self.bc)
                c_t = f_t * c_tm1+ i_t * c_t
                
                o_t = T.nnet.sigmoid( T.dot(x_t, self.wxo) + T.dot(h_tm1 , self.who) + T.dot(c_t, self.wco) + self.bo )
                
                h_t = o_t * T.tanh(c_t)  
                ## change dimension from nh to nc
                p_t_0 = T.dot(h_t, self.w)
                p_t_1 = T.dot(h_tm1, self.w) 
                p_t_2 = T.dot(h_tm2, self.w) 

                ## compute output label dependency from history output
                q_t_0 = T.dot(p_t_0,self.war0) 
                q_t_1 = T.dot(p_t_1,self.war1) 
                q_t_2 = T.dot(p_t_2,self.war2) 

                ## incorporate moving average
                s_t = self.b + q_t_0 + q_t_1 +q_t_2
                return [h_t,h_tm1,c_t,s_t]
            
            [h,_,c,s], _ = theano.scan(fn=recurrence,
                                    sequences=x,
                                    outputs_info=[self.h0,self.h0,self.c0, None],     
                                    n_steps=x.shape[0])    
            s = T.nnet.softmax(s) 
            p_y_given_x_sentence = s
            
        else:
            def recurrence(x_t, h_tm1,c_tm1):
                i_t = T.nnet.sigmoid( T.dot(x_t, self.wxi) + T.dot(h_tm1, self.whi) + T.dot(c_tm1, self.wci) + self.bi )
                f_t = T.nnet.sigmoid( T.dot(x_t, self.wxf) + T.dot(h_tm1, self.whf) + T.dot(c_tm1, self.wcf) + self.bf )
                
                c_t = T.tanh(T.dot(x_t, self.wxc) + T.dot(h_tm1, self.whc) + self.bc)
                c_t = f_t * c_tm1+ i_t * c_t
                
                o_t = T.nnet.sigmoid( T.dot(x_t, self.wxo) + T.dot(h_tm1, self.who) + T.dot(c_t, self.wco) + self.bo )
                
                h_t = o_t * T.tanh(c_t)   
                s_t = T.dot(h_t, self.w) + self.b
                return [h_t, c_t ,s_t]

            #shape h[x.shape[0],nh],s[x.shape[0],1,nc]
            [h,c,s], _ = theano.scan(fn=recurrence,
                                    sequences=[x],
                                    outputs_info=[self.h0,self.c0, None],     
                                    n_steps=x.shape[0])

            s = T.nnet.softmax(s)
            p_y_given_x_sentence = s

        ## get the highest probability
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        
        ## used for SGD
        sentence_gradients = T.grad(sentence_nll, self.params)
        if optimization ==None:
            sentence_updates = OrderedDict((p, p - lr*g)
                                           for p, g in
                                           zip(self.params, sentence_gradients))
        elif optimization =='Adagrad':
            sentence_updates = OrderedDict(adagrad(params=self.params, gparams=sentence_gradients, learning_rate = lr, epsilon = 1e-6))
        else:
            pass
        
        self.normalize = theano.function(inputs=[],
                                 updates={self.emb:
                                          self.emb /
                                          T.sqrt((self.emb**2)
                                          .sum(axis=1))
                        
                                          .dimshuffle(0, 'x')})
        self.normal = normal

        ## add momentum and ada
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        

    def train(self, x, y, window_size, learning_rate):
         ## add contect window and change into col[time-sequence]
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y
        
        self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()
            


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
        'nhidden': 300,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 60,
        'normal': False,
        'folder':'../result',
        'longdependence':None,
        'optimization':'Adagrad'
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

    train_lex = train_lex + test_lex
    train_y = train_y + test_y
    train_ne = train_ne + test_ne

    vocsize = len(dic['words2idx']) # # of words
    nclasses = len(dic['labels2idx']) # # of classes 
    nsentences = len(train_lex) #  # training sample [a batch is all the words in a sentence]

    ## get the label for (input,output) for test and valid set 
    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    

    print('... building the model')
    lstm = LSTM(
        nh=param['nhidden'],
        nc=nclasses,
        ne=vocsize,
        de=param['emb_dimension'],
        cs=param['win'],
        normal=param['normal'],
        longdependence = param['longdependence'],
        optimization = param['optimization']
        )

    ## build the model for mini-batch
    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    
    for epoch in range(param['nepochs']):

        param['ce'] = epoch
        tic = timeit.default_timer()
        print('epoch %i out of %i' %(epoch,param['nepochs']) )
        
        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            input_length = len(x)
            lstm.train(x, y, param['win'], param['clr'])
            print('[learning] epoch %i >> %2.2f%%' % (
                epoch, (i + 1) * 100. / nsentences), end=' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), end='')

        # evaluation // back into the real world : idx -> words
        predictions_valid = [map(lambda x: idx2label[x],
                             lstm.classify(numpy.asarray(
                             contextwin(x, param['win'])).astype('int32')))
                             for x in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              param['folder'] + '/current.valid.txt',
                              param['folder'])

        if res_valid['f1'] > best_f1:

            best_f1 = res_valid['f1']

            if param['verbose']:
                print('NEW BEST: epoch', epoch,
                      'best test F1', res_valid['f1'])

            param['tf1'] = res_valid['f1']
            param['tp'] = res_valid['p']
            param['tr'] = res_valid['r']
            param['be'] = epoch
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5

        if param['clr'] < 1e-5:
            break
                

    print('BEST RESULT: epoch', param['be'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])
    
    return lstm
