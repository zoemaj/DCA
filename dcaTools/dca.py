import numpy as np
import scipy.spatial.distance
import pandas as pd
import Bio.SeqIO
import matplotlib.pyplot as plt
import sequenceHandler as sh
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from math import ceil

''' 
This file contains the function plotTopContacts that plots the top N ranked DCA predictions overlaip on the structual contact map
originally written by the Duccio Malinverni https://doi.org/10.1007/978-1-4939-9608-7_16.
several modifications have been made to the original code.
'''

def extract_index_in_square(degree,N,k,l):
    """Extract the indices of the square of degree degree around the index (k,l)
    input:
            - degree    ->      degree of the square
                                int
            - N         ->      size of the square
                                int
            - k         ->      index k
                                int
            - l         ->      index l
                                int
    output:
            - indices_possibles     ->      list of the indices of the square of degree degree around the index (k,l)
                                            list of tuples
    """
    #for first squares we take the elemet of dcaErrors of index (k+1,l) or (k,l+1) or (k+1,l+1) or (k-1,l) or (k,l-1) or (k-1,l-1)
    min_k=max(0,k-degree)
    min_l=max(0,l-degree)
    max_k=min(N-1,k+degree)
    max_l=min(N-1,l+degree)
    indices_possibles=[]
    if min_k==(k-degree):
        indices_possibles+=[(min_k,i) for i in range(min_l,max_l+1)]
    if min_l==(l-degree):
        indices_possibles+=[(i,min_l) for i in range(min_k,max_k+1)]

    if max_k==(k+degree):
        indices_possibles+=[(max_k,i) for i in range(min_l,max_l+1)]
    if max_l==(l+degree):
        indices_possibles+=[(i,max_l) for i in range(min_k,max_k+1)]
    #remove the repetition of the index (k,l)
    indices_possibles=list(set(indices_possibles))
    return indices_possibles


###########################################################################################################################################
#********************************************** MAIN FUNCTION *****************************************************************************
########################################################################################################################################### 

def extractTopContacts(dcaFile,numTopContacts,diagIgnore=4,errors=None,penalizingErrors=False,Nsquare=4,sigma=1,WithoutGauss=False,coordinatesSquares=None,CutError_L=None,length_prot1=0):
    """Extract the top N ranked DCA contacts.
    input:
            - dcaFile           ->      path where to find the file containing the DCA predictions
                                        string
            - numTopContacts    ->      number of top DCA predictions to plot
                                        int

    default input:
            - diagIgnore        ->      how much of the diagonal we will not consider
                                        int, default=4  
            - errors            ->      list of errors maps
                                        list of numpy arrays, default=None
            - penalizingErrors  ->      if True, we penalize the errors
                                        bool, default=False
            - Nsquare           ->      size of the square around the center of the gaussian
                                        int, default=4  
            - sigma             ->      sigma of the gaussian
                                        float, default=1                
            - WithoutGauss      ->      if True, we do not use the gaussian
                                        bool, default=False                 
            - coordinatesSquares ->     list of coordinates of the squares
                                        list of lists, default=None             
            - CutError_L        ->      cut the errors in the diagonal +- CutError_L
                                        int, default=None               
            - length_prot1      ->      length of the first protein
                                        int, default=0          
    output:
            - dcaContacts       ->      list of the top N DCA contacts
                                        numpy array, dim=(N,2)
            - sortedScores      ->      list of the scores of the top N DCA contacts
                                        numpy array, dim=(N,)
            - errors_maps       ->      list of the errors maps
                                        list of numpy arrays
            - Ntop              ->      number of top DCA contacts
                                        int             
    """
    ##############################################################
    ##################### initialisation #########################
    ##############################################################
    dca_original=scipy.spatial.distance.squareform(np.loadtxt(dcaFile))
    N=dca_original.shape[0] #get the size of the dca
    if diagIgnore>0:
        print(f"Ignoring the first {diagIgnore} diagonal bands")
        for i in range(1,diagIgnore+1): #we put to 0 the elements of the diagonal +- diagIgnore
            matrix_diag_inf=np.diag(np.ones(N-i),k=i)
            matrix_diag_sup=np.diag(np.ones(N-i),k=-i)
            matrix_diag_inf=1-matrix_diag_inf
            matrix_diag_sup=1-matrix_diag_sup
            dca_original=dca_original*matrix_diag_inf #multiply each element of dca by the matrix_diag_inf
            dca_original=dca_original*matrix_diag_sup #multiply each element of dca by the matrix_diag_sup
    
    dca_tot=np.zeros((N,N)) #initialize the dca_tot
    errors_maps=[] #initialize the errors_maps
    print("Welcome in dca extraction :) ")
    print("Number of top contacts to extract:",numTopContacts)
    print("Size of the DCA:",N)
    ##############################################################
    ##############################################################



    ################################################################################################
    ####################################### errors extraction ######################################
    ################################################################################################
    if errors!=None:
        for id,error in enumerate(errors):
            #create a gaussian with N_neighbors dots around such that:
            #-> [i,j] is the center of the gaussian
            #-> the others dots [k,l] are the neighbors of the center with k,l in [i-N_neighbors/2,i+N_neighbors/2]x[j-N_neighbors/2,j+N_neighbors/2]
            if error.all() !=None:
                ########### compute the errors ########################
                if WithoutGauss:
                    print("Without Gaussian....")
                    dcaErrors=error
                else:
                    print(f"Gaussian errors with sigma={sigma} and Nsquare={Nsquare}...")
                    if not penalizingErrors:
                        print("But we are not penalizing the errors... If you want to penalize the errors, please set penalizingErrors=True")
                    alpha_i=[]
                    dcaErrors=np.zeros((N,N))
                    alpha_i=list([np.exp(-i**2/(2*(sigma**2))) for i in range(Nsquare)]) #compute the alpha_i coefficients
                    for i,alpha in enumerate(alpha_i): #compute the maxerr and minerr that we authorise      
                        print(f"Treatment of the square {i+1} with alpha={alpha}...")
                        if length_prot1==0:       
                            for k in tqdm(range(0,N,1)): #compute the dcaErrors with gaussian around the center
                                for l in range(0,N,1):
                                    indices_possibles=extract_index_in_square(i,N,k,l)
                                    dcaErrors[k,l]+=alpha*sum([error[a,b] for a,b in indices_possibles])    
                        else:
                            for k in tqdm(range(0,length_prot1)):
                                for l in range(length_prot1,N):
                                    indices_possibles=extract_index_in_square(i,N,k,l)
                                    dcaErrors[k,l]+=alpha*sum([error[a,b] for a,b in indices_possibles])
                            for k in tqdm(range(length_prot1,N)):
                                for l in range(0,length_prot1):
                                    indices_possibles=extract_index_in_square(i,N,k,l)
                                    dcaErrors[k,l]+=alpha*sum([error[a,b] for a,b in indices_possibles])
                ####################################################

                ########### initialisation ########################
                dca_new=np.zeros((N,N)) #initialize the new dca with very small values
                if coordinatesSquares!=[] and coordinatesSquares!=None:
                    coordinatesSquare=coordinatesSquares[id]
                    if coordinatesSquare==None:
                            continue
                    [x_s,y_s,w_s,h_s,_]=coordinatesSquare
                    #we will see the map beginning by 1 but the vectors and matrix beginn by 0
                    x_init=x_s-1
                    y_init=y_s-1
                    x_end=x_s-1+w_s
                    y_end=y_s-1+h_s 
                    error_map=np.zeros((N,N))
                    error_map[x_init:x_end+1,y_init:y_end+1]=dcaErrors[x_init:x_end+1,y_init:y_end+1]
                elif length_prot1!=0:
                    x_init=length_prot1
                    y_init=0
                    x_end=N-1
                    y_end=length_prot1-1
                    error_map=dcaErrors
                else:
                    x_init=0
                    y_init=0
                    x_end=N-1
                    y_end=N-1
                    error_map=dcaErrors
                ####################################################
                if penalizingErrors:
                    if not WithoutGauss: #we remove the border
                        
                        x_init=x_init+ceil(Nsquare/2)
                        y_init=y_init+ceil(Nsquare/2)
                        x_end=x_end-ceil(Nsquare/2)
                        y_end=y_end-ceil(Nsquare/2)
                    #attribute the errors to the dca_new in the good region
                    dca_new[x_init:x_end+1,y_init:y_end+1]=dca_original[x_init:x_end+1,y_init:y_end+1] - error_map[x_init:x_end+1,y_init:y_end+1] 
                   
                    #don't penalize the errors in the diagonal +- CutError_L (and that are inside of the square/border)
                    for i in range(N):
                        for j in range(N): 
                            if abs(i-j)<int(CutError_L) and (i>=x_init and i<=x_end and j>=y_init and j<=y_end):
                                dca_new[i,j]=dca_original[i,j]
                                error_map[i,j]=0
                else: #we do not penalize the errors
                    for i in range(N):
                        for j in range(N): #remove the elements in the diagonal +- CutError_L or outside of the square/border
                            if (i>=x_init and i<=x_end and j>=y_init and j<=y_end):
                                dca_new[i,j]=dca_original[i,j]
                errors_maps.append(error_map) #add the error map to the list of errors_maps
                dca_new-=np.diag(np.diag(dca_new)) #put the diagonal to 0
                for i in range(N):
                    for j in range(N):
                        if dca_tot[i,j]==0 and dca_new[i,j]!=0: #if we have not added the element in dca_tot and it is not 0 in our dca
                            dca_tot[i,j]=dca_new[i,j]                      
    else: #no errors
        dca_tot=dca_original
        if length_prot1!=0:
            dca_tot[:length_prot1,:length_prot1]=0
            dca_tot[length_prot1:,length_prot1:]=0  
            dca_tot[:length_prot1,length_prot1:]=0 #since for length_prot1!=0 we will only plot one part (and not also the symmetrical) we put to 0 the elements that are not in the square
    
    ################################################################################################
    ######################### TREATMENT OF THE SCORES ##############################################
    ################################################################################################
    if coordinatesSquares!=[] and coordinatesSquares!=None:
        dcaContacts=[]
        sortedScores=np.zeros(numTopContacts)
        for coordinatesSquare in coordinatesSquares:
            if coordinatesSquare==None:
                continue
            [x_s,y_s,w_s,h_s,N_s]=coordinatesSquare
            #we don't have necessarily a symmetrical square
            if x_s>y_s:
                down_triangular = np.tril(dca_tot,-1) # Keep only the lower triangular part
                matrix_triangular=down_triangular[x_s-1:x_s-1+w_s+1,y_s-1:y_s-1+h_s+1].flatten()
            else:
                upper_triangular = np.triu(dca_tot,1)
                matrix_triangular=upper_triangular[x_s-1:x_s-1+w_s+1,y_s-1:y_s-1+h_s+1].flatten()
            
            sortedScoresSquare=-np.sort(matrix_triangular) #sort them with higher number first
            sortedScoresSquare=sortedScoresSquare[sortedScoresSquare!=0]
        
            dca_tot_square=np.copy(dca_tot)

            #remove the elements that are not in the square
            if x_s>1:
                dca_tot_square[:x_s-1,:]=0
            if x_s-1+w_s < dca_tot.shape[0]:
                dca_tot_square[x_s-1+w_s+1:,:]=0
            if y_s>1:
                dca_tot_square[:,:y_s-1]=0
            if y_s-1+h_s < dca_tot.shape[0]:  
                dca_tot_square[:,y_s-1+h_s+1:]=0

            dcaContactsSquare=np.argwhere((dca_tot_square!=0) & (dca_tot_square>=-sortedScoresSquare[N_s])) 
            contactRanksSquare=np.argsort(-dca_tot_square[dcaContactsSquare[:,0],dcaContactsSquare[:,1]]) # Sort the contacts by score
            dcaContactsSquare = dcaContactsSquare[contactRanksSquare,:] # Sort the contacts by score
            
            
            #count how many times we have two times the same pair but with [i,j] and [j,i]
            N_top=N_s
            print("N_top:",N_top)
            for i in range(len(dcaContactsSquare)):
                if i>N_top:
                    #print("i:",i)
                    break
                x,y=dcaContactsSquare[i]
                dca_list_pairs=list(dcaContactsSquare[i+1:N_top+1])
                for id, el in enumerate(dca_list_pairs):
                    if el[0]==y and el[1]==x:
                        N_top+=1
                        print("new N_top:",N_top)
            dcaContactsSquare=dcaContactsSquare[:N_top,:] #adapt the number of contacts to the number of contacts that we want to plot
            print("dcaContactsSquare:",dcaContactsSquare)
            for pair in dcaContactsSquare:
                i=pair[0]
                j=pair[1]
                found=False
                if dcaContacts==[]:
                    dcaContacts.append([i,j])
                    continue
                for pair2 in dcaContacts:
                    i2=pair2[0]
                    j2=pair2[1]
                    if i==i2 and j==j2:
                        found=True
                        break
                if not found:
                    dcaContacts.append([i,j])

        #transform the list into a numpy array with dim (Ntop,2)
        Ntop=len(dcaContacts)
        dcaContacts=np.asarray(dcaContacts)
    else:
        upper_triangular = np.tril(dca_tot,1)
        
        
        sortedScores=-np.sort(upper_triangular.flatten())

        sortedScores=sortedScores[sortedScores!=0]
        
        dcaContacts=np.argwhere((dca_tot!=0) & (dca_tot>=-sortedScores[numTopContacts])) #take = if there are several times the same value sortedScores[numTopContacts]
        contactRanks=np.argsort(-dca_tot[dcaContacts[:,0],dcaContacts[:,1]])
        dcaContacts=dcaContacts[contactRanks,:]
        if length_prot1==0:
            Ntop=numTopContacts*2
        else:
            Ntop=numTopContacts
        dcaContacts=dcaContacts[:Ntop,:]

    max_to_print=min(10,len(dcaContacts))
    print(f"The top {max_to_print} contacts are:",dcaContacts[:max_to_print])
        
    
    dcaContacts=dcaContacts.astype(int)
    
    return dcaContacts,sortedScores,errors_maps,Ntop


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

    