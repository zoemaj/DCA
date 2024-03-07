import numpy as np
import numba
from numba import jit
import os

def weights(input_file, threshold, output_file) :
    '''
    This function write the weights of the different sequences in the MSA 
    input:
        input_file  ->  name of the input file containing the MSA preprocessed
                        string
        threshold   ->  threshold of similarity to consider two sequences as similar (percentage of identical amino acids in the same position in the two sequences)
                        float
        output_file ->  name of the output file
                        string
    '''
    #load the MSA in the input file
    MSA = np.genfromtxt(input_file, delimiter=',')
    #check if the folder exists and create it if not
    if not os.path.exists(os.path.dirname(output_file)):
        #create it
        os.makedirs(os.path.dirname(output_file))
    #check if the file already exists
    else:
        try:
            with open(output_file, "r") as file: #"r" is for read
                #ask if we want to overwrite it
                overwrite=input("Do you want to overwrite it? (yes/no)")
                while overwrite!="yes" and overwrite!="no":
                    input("Please enter yes or no")
                if overwrite=="no":
                    return
        except:
            pass
    #write the weight in the output file
    np.savetxt(output_file, get_weights(MSA,float(threshold)))

@jit(nopython=True, parallel=True) #parallelise using numba
def get_weights(MSA,threshold) :
    """
    This function compute the weight of each sequence given the threshold of similarity.
    input:  
        MSA         ->  table of sequences
                        csv file, shape (N,L) with N the number of homologous sequences and L the length of a sequence
        threshold   ->  percentage of similarity between two sequences considered as identical
                        two sequences are the same if (number of identical a.a at the good position)/L > threshold 
                        float
    """
    weights = np.zeros(len(MSA))
    for i, seq in enumerate(MSA) :
        if i%50 == 0 : print(i)
        weights[i] = weight(seq, MSA,threshold)
    return weights

@jit(nopython=True) #parallelise using numba
def weight(seq, MSA, threshold) :
    """
    This function will, according to the threshold, computes the weight of a sequence with the others in the MSA
    input:
        seq         ->  the seq from which we want to extract the weight
                        sequence of int numbers corresponding to the different a.a
        MSA         ->  table of sequences
                        csv file, shape (N,L) with N the number of homologous sequences and L the length of a sequence
        threshold   ->  percentage of similarity between two sequences considered as identical
                        two sequences are the same if (number of identical a.a at the good position)/L > threshold 
                        float
    output:
        weight      -> 1/(number of similar sequences according to the threshold)
                        float 
    """
    N= np.sum(np.sum(seq == MSA, axis=1)/len(seq) > threshold)
    if N == 0 :
        return 1.0
    else :
        return 1.0/N


