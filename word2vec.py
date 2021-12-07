#!/usr/bin/env python

import pdb
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = np.exp(x)/(1 + np.exp(x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 

    #N, d =  outsideVectors.shape
    s = len(outsideWordIdx)
    uTv = np.matmul(outsideVectors,centerWordVec) #outsideVectors (N x d), centerWordVec (d x 1) (should be 1 x d in theory?) --> uTv (N x 1)
    y_hat = softmax(uTv) # --> y_hat (N x 1) one score for each outsideVector
    loss = -np.log(y_hat[outsideWordIdx]) # --> loss (N x 1)
    d_value = np.tile(y_hat, (s, 1)) # --> s x N
    np.add.at(d_value, (range(s), outsideWordIdx), -1) # --> d_value s x N, -1 added to the outsideWordIdx in each row to match the 1-d outsideWordIdx case
    gradCenterVec = np.matmul(outsideVectors.T, d_value.T) # outsideVectors.T (d x N), d_value.T (N x s) --> gradCenterVec (d x s)
    gradOutsideVecs = np.swapaxes(np.matmul(d_value.T[:,:,None], centerWordVec.T[None, None, :]), 1, 2) # d_value.T (N x s), centerWordVec.T (1 x d), swap s and d 
                                                                                                        # --> gradOutsideVecs want (N x d x s)
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """
    
    s = len(outsideWordIdx)
    negSampleWordIndices = np.empty((K,s), dtype= int)  # negSampleWordIndices should be (K, s)
    for owv in range(s):
        for k in range(K):
            newidx = dataset.sampleTokenIdx()
            while newidx == outsideWordIdx[owv]:
                newidx = dataset.sampleTokenIdx()
            negSampleWordIndices[k][owv] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVe
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.

    N, d =  outsideVectors.shape
    s = len(outsideWordIdx)
    ind_thru_N = range(N)
    K = negSampleWordIndices.shape[0]

    gradOV_multval =  -np.ones((N, s)); 
    gradOV_multval[outsideWordIdx, range(s)] = 1; # --> gradOV_multval with 1 in the outsideWordIndices of each row and -1 else (N, s)
    uTv = np.matmul(outsideVectors,centerWordVec) #outsideVectors (N x d), centerWordVec (d x 1) --> uTv (N x 1)
    uTv_weighted = np.multiply(gradOV_multval.T, uTv).T # gradOV_multval (N x s), uTv (N x 1) --> broadcast weighted (N x s) scores
    s_uTv = sigmoid(uTv_weighted) # --> s_uTv (N x s) weighted sigmoids

    loss =  -np.log(s_uTv[outsideWordIdx, range(s)]) - np.sum(np.log(s_uTv[negSampleWordIndices, range(s)]), 0) 
    # s_uTv (N x s) indexed in N, s_uTv (indexed along K x s matrix), summed  --> loss (s x 1)

    negSampleWordIndices_flattened = np.matrix.flatten(negSampleWordIndices) # (K*s x 1)
    A = np.transpose(np.reshape(outsideVectors[negSampleWordIndices_flattened], [K,s,d]), (2, 0, 1)) #this is (d, K, s)
    tiled_s = np.tile(range(s), K)
    B = np.reshape( (1-s_uTv[negSampleWordIndices_flattened, tiled_s]), (K,s) ) # want B to be (K, s) (which it is)
    gradCenterVec =  -np.multiply(outsideVectors[outsideWordIdx].T, (1-s_uTv[outsideWordIdx, range(s)])) + np.sum( np.multiply(A, B), 1)
    # centerWordVec (d x 1), 1-s_uTv[outsideWordIdx, range(s) (s x 1) --> gradOutsideVecs (N, d, s) edited in dimension d x s
    # gradOV_multval[outsideWordIdx, range(s)] (s x 1), s_uTv[outsideWordIdx, range(s)] (s x 1), centerWordVec (d x 1) --> editing gradOutsideVecs want (d x 1) in each dim 2  
 
    gradOutsideVecs = np.zeros((N, s, d)) # --> gradOutsideVecs (N, s, d)
    gradOutsideVecs[outsideWordIdx, range(s), :] = -np.outer((1-s_uTv[outsideWordIdx, range(s)]), centerWordVec)
    # outsideVectors[indexing] (s x d) and (K x d x s) resp, s_uTv[indexing](K x s to s) and (K x s), summed over K resp --> gradCenterVec (d x s)

    C = np.outer(centerWordVec, B).T # centerWordVec (d x 1), B (K x s) --> C (K*s x d) 
    np.add.at(gradOutsideVecs, (negSampleWordIndices_flattened, tiled_s), C) #take the vector at k(i,j) (ith neg sample, goes between (here) 0 and 4 for 10 numbers
                                                                                #jth outerWordIdx, goes between (here) 0 and 4 for 10 numbers
    gradOutsideVecs = np.transpose(gradOutsideVecs, (0, 2, 1))
    
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)

    centerWordIdx = word2Ind[currentCenterWord] # --> centerWordIdx (1 x 1)
    centerWordVec = centerWordVectors[centerWordIdx] # --> centerWordVec (d x 1) 
    outsideWordIndices = [word2Ind[i] for i in outsideWords] # outsideWords (< 2w strings) --> outideWordIndices (s x 1)
    many_loss, many_gradCenter, many_gradOutside = \
            word2vecLossAndGradient(centerWordVec, outsideWordIndices, outsideVectors, dataset) # centerWordVec (d x 1), outsideWordIndices (s x 1), outsideVectors (N x d)
                                                                                                # --> many_loss (s x 1), many_gradCenter (d x s), many_gradOutside (N x d x s)
    loss += sum(many_loss)
    gradCenterVecs[centerWordIdx] += np.sum(many_gradCenter, 1)
    gradOutsideVectors += np.sum(many_gradOutside, 2)

    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()

