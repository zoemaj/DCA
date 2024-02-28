import numpy as np
import numba
from numba import jit
import os

def weights(input_file, threshold, output_file) :
    """
    load the MSA from the input file, compute the weight for each sequence and write them in the output file
    :param input_file: name of the input file containing the MSA preprocessed using "execute_preprocessing.py"
    :type input_file: string
    :param output_file: name of the output file
    :type output_file: string
    :return: nothing
    :rtype: None
    """
    #load the MSA in the input file
    MSA = np.genfromtxt(input_file, delimiter=',')
    #check if the folder exists
    if not os.path.exists(os.path.dirname(output_file)):
        #create it
        os.makedirs(os.path.dirname(output_file))
    #write the weight in the output file
    np.savetxt(output_file, get_weights(MSA,float(threshold)))

@jit(nopython=True, parallel=True) #parallelise using numba
def get_weights(MSA,threshold) :
    """
    compute the weight of each sequence of the MSA given in paramter

    :param MSA: preprocessed MSA
    :type MSA: numpy array
    :return: weights of each sequence
    :rtype: numpy array
    """
    weights = np.zeros(len(MSA))
    for i, seq in enumerate(MSA) :
        if i%50 == 0 : print(i)
        weights[i] = weight(seq, MSA,threshold)
    return weights

@jit(nopython=True) #parallelise using numba
def weight(seq, MSA, threshold) :
    """
    compute the weight of the sequence seq given the MSA and return the weight

    :param seq: sequence of the MSA for which we want to compute the weight
    :type seq: numpy array
    :param MSA: MSA to which the sequence belong
    :type seq: numpy array
    :return: weight of the sequence
    :rtype: numpy array
    """
    #we compute the weight with a threshold of 80% similarity
    #modif add 27-02-24. BEFORE only this: return 1/(np.sum(np.sum(seq == MSA, axis=1)/len(seq) > threshold))
    #check that the sum is not 0
    N= np.sum(np.sum(seq == MSA, axis=1)/len(seq) > threshold)
    if N == 0 :
        return 1.0
    else :
        return 1.0/N


