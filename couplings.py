import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import torch
import os

#############################################################################
######### THIS IS FROM AUDE MAYER 2022, NOT USED IN THIS VERSION ############
#############################################################################
def prediction(input, model, model_type) :
    if model_type == "non-linear" : return model.non_linear(input)
    elif model_type == "mix" : return model.linear(input) + model.non_linear(input)
    else : print("error with model type")  
#############################################################################
#############################################################################  

def extract_couplings(model, model_type, original_shape) :
    """
    This function extracts the couplings coefficients from the model and returns them
    inputs:     model               ->         pytorch model computed with model.py
                model_type          ->         type of the model (linear, non-linear or mix) !!!! in this version, only linear !!!!!
                original_shape      ->         original shape of the MSA (L,K)
    outputs:    couplings           ->         couplings coefficients, shape (L*K,L*K)
    """
    (L,K) = original_shape
    #for a linear model, the couplings are directly the weights learned
    if model_type == "linear" :
        return np.array(model.masked_linear.linear.weight.detach().cpu())
        #note zozo: It returns a new tensor without requires_grad = True. The gradient with respect to this tensor will no longer be computed.
    #############################################################################
    ######### THIS IS FROM AUDE MAYER 2022, NOT USED IN THIS VERSION ############
    #############################################################################
    elif model_type == "non-linear" or model_type == "mix":
        model.eval()
        with torch.no_grad() :
            #bias
            zeros = torch.zeros((L*K,L*K))
            pred_zero = prediction(zeros, model, model_type)
            pred_zero = pred_zero.detach()
            #prediction of delta_{i'=i} = bias + couplings[i]
            input_i = torch.eye(L*K)
            pred_i = prediction(input_i, model, model_type)
            pred_i = pred_i.detach()            
            return np.array(pred_i - pred_zero)
    #############################################################################
    #############################################################################
    else : print("error with model type")


def ising_gauge(couplings, model, model_type, original_shape, data_per_col) :
    """
    apply Ising gauge on the coupling coefficients

    if model_type is "linear", only the second order Ising gauge is applies
    if model_type is "non-linear" or "mix", both second and third order Ising gauge are applied

    :param couplings: couplings coefficients on which ising gauge is needed
    :type couplings: numpy array
    :param model: pytorch model from which the couplings have been extracted
    :type model: nn.Module
    :param original_shape: original shape of the MSA (L,K)
    :type original_shape: tuple of int
    :return: coupling coefficients after Ising gauge
    :rtype: numpy array
    """
    (L,K) = original_shape
    #############################################################################
    ######### THIS IS FROM AUDE MAYER 2022, NOT USED IN THIS VERSION ############
    #############################################################################
    if model_type == "non-linear" or model_type == "mix" :
        model.eval()
        with torch.no_grad() :
            #bias
            zeros = torch.zeros((L*K,L*K))
            pred_zero = prediction(zeros, model, model_type)
            pred_zero = pred_zero.detach()
            #extract third order interaction coefficients from the model and use them to apply third order ising gauge
            for l in range(L) :
                print("gauge process on triplets : ", l, "/", L)
                nb_rows = (L-1) * K
                input_j = torch.cat((torch.eye(nb_rows)[:,:l*K], torch.zeros(nb_rows, K), torch.eye(nb_rows)[:,l*K:]), dim=1)
                pred_j = prediction(input_j, model, model_type)
                pred_j = pred_j.detach()
                for k in range(K) :
                    input_i_j = torch.clone(input_j)
                    input_i_j[:,l*K+k] = 1
                    pred_i_j = prediction(input_i_j, model, model_type)
                    pred_i_j = pred_i_j.detach()
                    input_i = torch.zeros(nb_rows, L*K)
                    input_i[:,l*K+k] = 1
                    pred_i = prediction(input_i, model, model_type)
                    pred_i = pred_i.detach()
                    #third order interaction coefficients between residue k and all other residues for all categories
                    triplets_l = np.array(pred_i_j - pred_i - pred_j + pred_zero[:len(pred_i_j)])
                    #third order ising gauge
                    couplings[l*K+k,:] += 2 * (np.sum(triplets_l,axis=0) / K)
    #############################################################################
    #############################################################################
                    
    #second order Ising gauge
    if model_type == "linear":
        new_couplings = np.copy(couplings)
        # Initialize arrays outside the loops
        sum_row = np.zeros((L * K, L * K))
        sum_col = np.zeros((L * K, L * K))
        sum_rowcol = np.zeros((L * K, L * K))
        print("--------- sum over b of C_ia,jb -----------")
        for col_i in range(0, L * K):
            mask_col = (data_per_col[col_i % K, col_i // K] != 1)
            sum_row[:, col_i] = np.where(mask_col, np.sum(couplings[:, col_i:col_i + K], axis=1), 0)
        print("--------- sum over a of C_ia,jb -----------")
        for row_i in range(0, L * K):
            mask_row = (data_per_col[row_i % K, row_i // K] != 1)
            sum_col[row_i, :] = np.where(mask_row, np.sum(couplings[row_i:row_i + K, :], axis=0), 0)
        print("--------- sum over a,b of C_ia,jb -----------")
        for row_i in range(0, L * K, K):
            for col_i in range(0, L * K):
                mask_row = (data_per_col[row_i % K, row_i // K] != 1)
                mask_col = (data_per_col[col_i % K, col_i // K] != 1)
                sum_rowcol[row_i:row_i + K, col_i] = np.sum(sum_col[row_i:row_i + K, col_i], axis=0)
                sum_rowcol[row_i:row_i + K, col_i] = np.where(mask_row & mask_col, sum_rowcol[row_i:row_i + K, col_i], 0)
        print("--------- computing the new_couplings -----------")
        for row_i in range(L * K):
            for col_i in range(L * K):
                K_beta = np.sum(data_per_col[:, col_i // K] == 0)
                K_alpha = np.sum(data_per_col[:, row_i // K] == 0)
                new_couplings[row_i, col_i] = (
                    new_couplings[row_i, col_i] - sum_row[row_i, col_i] / K_alpha
                    - sum_col[row_i, col_i] / K_beta
                    + sum_rowcol[row_i, col_i] / (K_alpha * K_beta)
                )
        return new_couplings

@jit(nopython=True, parallel=True) #parallelise using numba
def average_product_correction(f) :
    """
    apply the average product correction on f, a numpy array containing the couplings,
    and return the corrected f
    :param f: array on which we want to apply the average product correction
    :type f: numpy array
    :return: f after average product correction
    :rtype: numpy array
    """
    shape = f.shape
    f_i_s = np.sum(f, 1)/(shape[1]-1)
    f_j_s = np.sum(f, 0)/(shape[0]-1)
    f_ = np.sum(f)/(shape[0]*(shape[1]-1))
    for i in range(shape[0]) :
        for j in range(shape[1]) :
            if j!= i : f[i,j] = f[i,j]-f_i_s[i]*f_j_s[j]/f_    
    return f

def couplings(model_name, L, K, number_model=1, type_average='average_couplings', output_name='/', figure=False, data_per_col='/', model_type="linear") :
    
    #now we have one model or a list of models with <model_name>_<number> such that <number> is 1,2,...number_model
    #we need to extract the model(s) and apply the couplings function on it/each of them
    #then we need to average the couplings
    ''' 
    This function extracts the couplings coefficients from the model(s) and applies the Ising gauge on it/them
    inputs:     model_name          ->         name of the model or the <model_name> for each model <model_name>_<number>
                model_type          ->         type of the model (linear, non-linear or mix) !!!! in this version, only linear !!!!!
                L                   ->         length of the sequences
                K                   ->         number of categories max 21 or 29 with taxonomies
                data_per_col        ->         name of the file containing the data per column.
                                                It is an array of shape (K,L) where a 1 means that this a.a is not present in the column of the MSA (N,L)
                number_model        ->         number of models to average 
                type_average        ->         type of average to apply on the couplings
                output_name         ->         name of the output file
    '''

    ###################################################################################
    ################# EXTRACTION OF THE MODEL(S) ######################################
    ############ & data_per_col (the same for every model(s)) #########################
    ###################################################################################
    models=[]
    number_model=int(number_model)
    if number_model>1:
        for m in range(number_model):
            print("model_name:", model_name+'_' + str(m))
            models.append(torch.load(model_name+'_' + str(m)))
    else:
        models.append(torch.load(model_name))
    L = int(L)
    K = int(K)
    #load the data_per_col .txt file and convert it into a numpy array
    if data_per_col=="/":
        #take the path of the model_name and add the name 'data_per_col.txt' to it
        data_per_col = os.path.dirname(model_name)
        data_per_col = os.path.join(data_per_col, 'data_per_col.txt')
        print("take the default data_per_col file: ", data_per_col)
    data_per_col=np.loadtxt(data_per_col)
    ###################################################################################
    ###################################################################################
    if output_name=="/":
        #take the path of the model_name and add the name 'couplings' to it
        output_name = os.path.dirname(model_name)
        output_name = os.path.join(output_name, 'couplings')
    ###################################################################################
    ################# WEIGHT EXTRACTION AND ISING GAUGE FOR THE MODEL(S) ##############
    ################### (depend if we consider the taxonomy or not)####################
    ###################################################################################
    print("------------ weight extraction and Ising gauge for the model(s) ------------")

    #createthe name
    name_couplings_before = os.path.dirname(output_name) # Path for the couplings directory without the last 
    #add the name 'couplings_before_ising' to the couplings_path
    name_couplings_before = os.path.join(name_couplings_before, 'couplings_before_ising/')
    #check if it exist
    if os.path.isfile(os.path.join(name_couplings_before, "couplings_before_ising.txt")):
        print("couplings_before_ising.txt already exists, we don't compute it again")
        average_couplings=np.loadtxt(os.path.join(name_couplings_before, "couplings_before_ising.txt"))
        #print("shape of couplings before ising:", average_couplings.shape) #oke
        if K>21:
            L=L-1 #lose a dimension with class
            data_per_col=data_per_col[:,:-1] #remove the last column corresponding to the class type
        print('L,K:(',L,',',K,')')
    else:
        #average_couplings=np.zeros((L*K,L*K))
        if K>21:
            print("K=",K," we have the class type for each sequence")
            average_couplings=np.zeros(((L-1)*K,(L-1)*K)) #with class
        else: #K=21
            print("K=",K," we don't have the class type for each sequence")
            average_couplings=np.zeros((L*K,L*K))
        ALL_couplings=[]
        for model in models:
            couplings = extract_couplings(model, model_type, (L,K)) #we got an array of dim (L*K,L*K)
            if K>21:
                #with class we need to remove the last blocs of size K
                print("treatment of couplings: remove the last column corresponding to the class type")
                couplings=couplings[:-K,:-K]
            ALL_couplings.append(couplings)
            average_couplings += couplings
        if K>21:
            L=L-1 #lose a dimension with class
            data_per_col=data_per_col[:,:-1] #remove the last column corresponding to the class type
        print('L,K:(',L,',',K,')')
        average_couplings=average_couplings/number_model
        #
        # Create the directory and its parent directories if they don't exist
        os.makedirs(name_couplings_before, exist_ok=True)
        np.savetxt(os.path.join(name_couplings_before, "couplings_before_ising.txt"), average_couplings)
        print("couplings before ising gauge saved in the file: ", os.path.join(name_couplings_before, "couplings_before_ising.txt"))
    

    ###################################################################################
    ###################################################################################

    ###################################################################################
    ################# PLOT OF THE COUPLINGS BEFORE ISING ##############################
    ###################################################################################
    if figure==True:
        print("------------ plot before the ising gauge ------------")
        plt.figure()
        plt.plot(np.triu(average_couplings).flatten(), np.tril(average_couplings).T.flatten(), '.')
        plt.plot(np.linspace(-2, 3), np.linspace(-2, 3), '--')
        plt.xlabel("$C_{\lambda \kappa lk}$", fontsize=18)
        plt.ylabel("$C_{lk \lambda \kappa}$", fontsize=18)
        plt.grid()
        #save it
        plt.savefig(os.path.join(name_couplings_before, "couplings_before_ising.png"))
    else:
        print("No plot before the ising gauge, because figure=False")
    
    ##################################################################################
    ##################################################################################
    
    ##################################################################################
    ################# COMPUTATION OF THE ISING GAUGE #################################
    ##################################################################################
    '''
    There are 3 cases:
    (1) number_model=1: 
        we consider only one model so len(models)=1 and the model is the index 0.
    (2) number_model>1 and type_average="average_couplings"
        we apply ising gauge on each model and then average the couplings
    (3) number_model>1 and type_average="average_couplings_frob"
        we apply ising gauge on each model, and apply the average product correction on each couplings before averaging them

    So for (2), (3) we need to apply ising on each couplings
    AND for (1) and (2) we have an average_couplings (the average_couplings = the couplings for the only model in case (1))

    '''

    print("------------ computation of ising gauge ------------")
    if number_model>1:
        step=1
        if type_average=="average_couplings_frob" or type_average=="average_couplings":
            # Path for the couplings directory without the last part of the output_name (to stock in the folder)
            couplings_path = os.path.dirname(output_name)
            #add the name 'couplings_models' to the couplings_path
            couplings_path = os.path.join(couplings_path, 'couplings_with_ising_models/')
            # Create the directory and its parent directories if they don't exist
            os.makedirs(couplings_path, exist_ok=True)
            print("on every model")
            average_couplings=np.zeros((L*K,L*K))
            ALL_couplings_ising=[]
            for couplings,model in zip(ALL_couplings,models):
                print("gauge process on model : ", step, "/", number_model)
                #look if the file with path couplings_path and name couplings_ising_step.txt exists
                #if it exists, load it and don't do ising_gauge again
                #if it doesn't exist, do ising_gauge and save it
                if os.path.isfile(os.path.join(couplings_path, "couplings_ising_" + str(step-1) + ".txt")):
                    print("couplings_ising_" + str(step-1) + ".txt already exists, we don't compute it again")
                    couplings = np.loadtxt(os.path.join(couplings_path, "couplings_ising_" + str(step-1) + ".txt"))
                else:
                    couplings = ising_gauge(couplings, model, model_type, (L,K), data_per_col)
                    np.savetxt(os.path.join(couplings_path, "couplings_ising_" + str(step-1) + ".txt"), couplings)
                    print("couplings after ising gauge saved in the file: ", os.path.join(couplings_path, "couplings_ising_" + str(step-1) + ".txt"))
                ALL_couplings_ising.append(couplings)
                average_couplings += couplings
                print("couplings after ising gauge shape:", average_couplings.shape)
                step+=1
            average_couplings=average_couplings/number_model
        else:
            print("error with type_average")
    else: # we do ising on the average of the models (number_model=1)
        print("only on the couplings from the average model (because selected to do only on 1model)")
        couplings_path = os.path.dirname(output_name)
        #add the name 'couplings_after_ising' to the couplings_path
        couplings_path = os.path.join(couplings_path, 'couplings_after_ising/')
        #check if it exist
        if os.path.isfile(os.path.join(couplings_path, "couplings_after_ising.txt")):
            print("couplings_after_ising.txt already exists, we don't compute it again")
            average_couplings=np.loadtxt(os.path.join(couplings_path, "couplings_after_ising.txt"))
        else:
            average_couplings_old=np.copy(average_couplings)
            #print("shape data_per_col:", data_per_col.shape)
            average_couplings=ising_gauge(average_couplings_old,models[0],model_type,(L,K), data_per_col)
            # Create the directory and its parent directories if they don't exist
            os.makedirs(couplings_path, exist_ok=True)
            np.savetxt(os.path.join(couplings_path, "couplings_after_ising.txt"), average_couplings)
            print("couplings after ising gauge saved in the file: ", os.path.join(couplings_path, "couplings_after_ising.txt"))
    
            

    ##################################################################################
    ##################################################################################
    
    ###################################################################################
    ################# PLOT OF THE COUPLINGS AFTER ISING ###############################
    ###################################################################################
    if figure==True:
        print("------------ plot after the ising gauge ------------")
        #plot symmetry of coupling coeff after Ising gauge
        plt.figure()
        plt.plot(np.triu(average_couplings).flatten(), np.tril(average_couplings).T.flatten(), '.')
        plt.plot(np.linspace(-2, 3), np.linspace(-2, 3), '--')
        plt.xlabel("$C_{\lambda \kappa lk}$", fontsize=18)
        plt.ylabel("$C_{lk \lambda \kappa}$", fontsize=18)
        plt.grid()
        #save it
        plt.savefig(os.path.join(couplings_path, "couplings_after_ising.png"))
    else:
        print("No plot after the ising gauge, because figure=False")
    ##################################################################################
    ##################################################################################
    

    ##################################################################################
    ################# AVERAGE PRODUCT CORRECTION #####################################
    ##################################################################################
    '''
    There are 2 cases:
    (1) number_model=1:, number_model>1 and type_average="average_couplings"
        we apply the average product correction on the average_couplings
    (2) number_model>1 and type_average="average_couplings_frob":
        we apply the average product correction on each couplings before averaging them
    '''
    print("------------ average product correction ------------")
    output_directory = os.path.dirname(output_name) # Path for the couplings directory without the last part of the output_name (to stock in the folder)
    # saved the last part as the name of the file
    output_name = os.path.basename(output_name)
    if number_model==1:
        #add the name 'average-models' to the output_directory path
        output_directory = os.path.join(output_directory, 'average-models/')
    else:
        if type_average=="average_couplings_frob":
            output_directory = os.path.join(output_directory, 'average-models-and-frob/')
        elif type_average=="average_couplings":
            output_directory = os.path.join(output_directory, 'average-couplings/')
    #check if it exist
    if os.path.isfile(os.path.join(output_directory, output_name)):
        print("The final couplings file already exists, we don't compute it again")
    else:
        n=0
        print("---------------- Frobenius ----------------")
        if number_model>1 and type_average=="average_couplings_frob":
            for couplings in ALL_couplings_ising:
                couplings+=0.5*(couplings + couplings.T)
            #reshape couplings in a L x L array where each element contains the K x K categorical couplings to apply frobenius norm on each element
                matrix = []
                for i in range(L) :
                    rows = []
                    for j in range(L) :
                        rows.append(couplings[i*K:(i+1)*K, j*K:(j+1)*K])
                    matrix.append(rows)
                couplings = np.array(matrix)
                #frobenius norm
                couplings = np.linalg.norm(couplings, 'fro', (2, 3))
                #average product correction
                couplings = average_product_correction(couplings)
                #reshape in form (0,1) (0,2) ... (1,2) (1,3) ...
                couplings = np.triu(couplings)
                tmp = []
                for i in range(L) :
                    for j in range(i+1, L) :
                        tmp.append(couplings[i,j])
                couplings = np.array(tmp) 
                if n==0:
                    average_couplings=np.copy(couplings)
                    n=1
                average_couplings += couplings
            average_couplings = average_couplings/number_model
            print("couplings after frob shape:", average_couplings.shape)
        else: #average couplings or for number_model=1
            average_couplings+=0.5*(average_couplings + average_couplings.T)
            #reshape couplings in a L x L array where each element contains the K x K categorical couplings to apply frobenius norm on each element
            matrix = []
            for i in range(L) :
                rows = []
                for j in range(L) :
                    rows.append(average_couplings[i*K:(i+1)*K, j*K:(j+1)*K])
                matrix.append(rows)
            average_couplings = np.array(matrix)
            average_couplings = np.linalg.norm(average_couplings, 'fro', (2, 3))
            average_couplings = average_product_correction(average_couplings)
            average_couplings = np.triu(average_couplings)
            tmp = []
            for i in range(L) : 
                for j in range(i+1, L) :
                    tmp.append(average_couplings[i,j]) #(0,1), (0,2),...(1,2),(1,3),....(L-1,L) -> L*(L-1)/2 elements in total
            average_couplings = np.array(tmp)
        ##################################################################################
        ##################################################################################
        
        #################################################################################
        ################# SAVING THE COUPLINGS ##########################################
        #################################################################################
        
        os.makedirs(output_directory, exist_ok=True) # Create the directory and its parent directories if they don't exist
        #print the path of: os.path.join(output_directory, output_name)
        
        np.savetxt(os.path.join(output_directory, output_name), average_couplings)
        print("The final couplings file is saved in the file: ", os.path.join(output_directory, output_name))
        print("---------------- END ----------------")
        #################################################################################
        #################################################################################