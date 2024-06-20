import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import gc

#####################################################################################################
##################################### functions to compute the errors ###############################
#####################################################################################################

def ErrorZai(amino,true_amino,bias=False):
    '''
    This function compute the error of the prediction of the amino acid a 
    input:
            amino: the prediction of the amino acid a, shape (K)
            true_amino: the true amino acid a, shape (K)
    output:
            error_ai: the error of the prediction of the amino acid a, shape (1,K)
                    delta Z_a0, delta Z_a1, ..., delta Z_aK
    '''
   
    K=true_amino.shape[0]
    sum_exp_amino=np.sum(np.exp(amino))#apply np.exp on each element of amino and sum them
    assert np.allclose(sum_exp_amino,np.zeros_like(sum_exp_amino))==False
    delta_s_ai=np.zeros_like(true_amino)
    for i in range(0,K):
        if amino[i]!=0:
            delta_s_ai[i]=abs(true_amino[i]-amino[i])/abs(amino[i])
        else:
            delta_s_ai[i]=0
    delta_z_ai=np.reshape(delta_s_ai,(K,1))
        
    return delta_z_ai

def ErrorWij_before_ising(model,L,K,device,length_prot1):
    ''' 
    Compute the error of Wa_i,b_j such that delta(W_a_i,b_j)=delta(Z_a_i(input with a_b_j=1 and the others 0))
    '''
    #diagonal matrix of shape (L*K,L*K)
    input=torch.eye(L*K)
    input=torch.reshape(input,(L*K,L*K))
    input=input.to(device)
    output_ai=torch.zeros((L*K,L*K))
    N=L*K
    #compute the error for each a_i
    print("Error Wai,bj before ising...")
    error_Wai_bj=ErrorBias(model,L,K,device)/L #divice each element by L
    print("shape error_Wai_bj:",error_Wai_bj.shape)
    print("length_prot1:",length_prot1)
    print("Extract the Zai,bj...")
    output_ai=model.masked_linear.linear(input) 
    output_ai = torch.reshape(output_ai, (N,L, K))#reshape 
    SoftMax=torch.nn.Softmax(dim=2) #apply softmax on the last dimension
    output_ai = SoftMax(output_ai) #apply softmax
    output_ai = torch.reshape(output_ai, (N, L*K)) 
    output_ai=output_ai.detach().numpy()
    print("done")
    for seq in tqdm(range(0,N)):
        for amino_id in range(0,L*K,K):
            amino=output_ai[seq,amino_id:amino_id+K]
            true_label=input[seq,amino_id:amino_id+K]
            error_Wai_bj[amino_id:amino_id+K,seq]+=ErrorZai(amino,true_label)[:,0]
    del output_ai
    torch.cuda.empty_cache()
    assert np.allclose(error_Wai_bj,np.zeros_like(error_Wai_bj))==False
    return error_Wai_bj


def ErrorBias(model,L,K,device):
    #fow now consider the bias as the difference with the average
    input=torch.zeros(L*K)
    #put on the device
    input=input.to(device)
    model_on_device=model.to(device)
    output=model_on_device.masked_linear.linear(input)
    output = torch.reshape(output, (1,L, K))#reshape 
    SoftMax=torch.nn.Softmax(dim=2)
    output = SoftMax(output) #apply softmax
    output = torch.reshape(output, (1, L*K)) 
    output=output.detach().numpy()
    input=input.detach().numpy()
    input=input.reshape(1,L*K)
    error_bias=np.zeros((L*K,1))
    for a in tqdm(range(0,L*K,K)):
        amino=output[0,a:a+K]
        true_label=input[0,a:a+K]
        delta_bias=ErrorZai(amino,true_label,bias=True) #delta Z_a0, delta Z_a1, ..., delta Z_aK
        error_bias[a:a+K]=delta_bias
    del output
    torch.cuda.empty_cache()
    error_bias_final=np.repeat(error_bias,L*K,axis=1)

    return error_bias_final


def ErrorWij_after_ising(error_Wij_before_ising,data_per_col,L,K,length_prot1):
    print("Error Wai,bj after ising...")

    ErrorWij_after_ising=np.zeros((L*K,L*K))

    sum_over_i=np.zeros(L*K) 
    sum_over_j=np.zeros(L*K) #sum_over_j[0] correspond to the amino 0 of value 0, sum_over_j[K-1] correspond to the amino 0 of value K-1, sum_over_j[K] correspond to the amino 1 of value 0, ...
    for amino_id in range(0,L*K,K):
        for value_amino in range(0,L*K):
            sumi=np.sum(error_Wij_before_ising[amino_id:amino_id+K,value_amino],axis=0)
            sum_over_i[value_amino]=sumi
            sumj=np.sum(error_Wij_before_ising[value_amino,amino_id:amino_id+K],axis=0)
            sum_over_j[value_amino]=sumj


    if length_prot1==0:
        for amino_i in tqdm(range(0,L*K)):
            for amino_j in range(0,L*K):
                if data_per_col[amino_i%K,amino_i//K]==0 and data_per_col[amino_j%K,amino_j//K]==0: #both are true so the amino is possible!
                    amino_i_id=amino_i//K
                    amino_j_id=amino_j//K
                    K_a=np.sum(data_per_col[:,amino_i_id]!=1)
                    K_a=int(K_a)
                    K_b=np.sum(data_per_col[:,amino_j_id]!=1)
                    K_b=int(K_b)
                    sum_over_j_but_not_the_j_fixed=sum_over_j[amino_i]-error_Wij_before_ising[amino_i,amino_j]
                    sum_over_i_but_not_the_i_fixed=sum_over_i[amino_j]-error_Wij_before_ising[amino_i,amino_j]
                    sum_over_i_and_j_but_not_the_j_fixed=np.sum(sum_over_j[amino_i_id:(amino_i_id+K)]-error_Wij_before_ising[amino_i_id:(amino_i_id+K),amino_j])
                    sum_over_i_and_j_but_not_the_i_and_j_fixed=sum_over_i_and_j_but_not_the_j_fixed-(sum_over_j[amino_i]-error_Wij_before_ising[amino_i,amino_j])
                    ErrorWij_after_ising[amino_i,amino_j]=sum_over_i_and_j_but_not_the_i_and_j_fixed*abs(1.0/(K_a*K_b))
                    ErrorWij_after_ising[amino_i,amino_j]+=sum_over_j_but_not_the_j_fixed*abs(1.0/(K_a*K_b)-1.0/K_b)
                    ErrorWij_after_ising[amino_i,amino_j]+=sum_over_i_but_not_the_i_fixed*abs(1.0/(K_a*K_b)-1.0/K_a)
                    ErrorWij_after_ising[amino_i,amino_j]+=error_Wij_before_ising[amino_i,amino_j]*abs(1.0/(K_a*K_b)-1.0/K_b-1.0/K_a+1.0)

    else: #only attribute it for (amino_i,amino_j) in [0:length_prot1,length_prot1:] and [length_prot1:,0:length_prot1]
        for amino_i in tqdm(range(0,length_prot1*K)):
            for amino_j in range(length_prot1*K,L*K):
                mask_amino_i=torch.tensor(data_per_col[amino_i%K,amino_i//K]==0)
                mask_amino_j=torch.tensor(data_per_col[amino_j%K,amino_j//K]==0)
                if mask_amino_i and mask_amino_j:
                    amino_i_id=amino_i//K
                    amino_j_id=amino_j//K
                    K_a=np.sum(data_per_col[:,amino_i_id]!=1)
                    K_a=int(K_a)
                    K_b=np.sum(data_per_col[:,amino_j_id]!=1)
                    K_b=int(K_b)

                    sum_over_j_but_not_the_j_fixed=sum_over_j[amino_i]-error_Wij_before_ising[amino_i,amino_j]
                    sum_over_i_but_not_the_i_fixed=sum_over_i[amino_j]-error_Wij_before_ising[amino_i,amino_j]
                    sum_over_i_and_j_but_not_the_j_fixed=np.sum(sum_over_j[amino_i_id:amino_i_id+K]-error_Wij_before_ising[amino_i:amino_i+K,amino_j])
                    sum_over_i_and_j_but_not_the_i_and_j_fixed=sum_over_i_and_j_but_not_the_j_fixed-(sum_over_j[amino_i]-error_Wij_before_ising[amino_i,amino_j])
                    ErrorWij_after_ising[amino_i,amino_j]=sum_over_i_and_j_but_not_the_i_and_j_fixed*abs(1.0/(K_a*K_b))
                    ErrorWij_after_ising[amino_i,amino_j]+=sum_over_j_but_not_the_j_fixed*abs(1.0/(K_a*K_b)-1.0/K_b)
                    ErrorWij_after_ising[amino_i,amino_j]+=sum_over_i_but_not_the_i_fixed*abs(1.0/(K_a*K_b)-1.0/K_a)
                    ErrorWij_after_ising[amino_i,amino_j]+=error_Wij_before_ising[amino_i,amino_j]*abs(1.0/(K_a*K_b)-1.0/K_b-1.0/K_a+1.0)

                    #do also the same for the element (amino_j,amino_i)
                    sum_over_j_but_not_the_j_fixed=sum_over_j[amino_j]-error_Wij_before_ising[amino_i,amino_j]
                    sum_over_i_but_not_the_i_fixed=sum_over_i[amino_i]-error_Wij_before_ising[amino_i,amino_j]
                    sum_over_i_and_j_but_not_the_i_and_j_fixed=np.sum(sum_over_j[amino_j_id:amino_j_id+K])-error_Wij_before_ising[amino_i,amino_j]
                    ErrorWij_after_ising[amino_j,amino_i]=sum_over_i_and_j_but_not_the_i_and_j_fixed*abs(1.0/(K_a*K_b))
                    ErrorWij_after_ising[amino_j,amino_i]+=sum_over_j_but_not_the_j_fixed*abs(1.0/(K_a*K_b)-1.0/K_a)
                    ErrorWij_after_ising[amino_j,amino_i]+=sum_over_i_but_not_the_i_fixed*abs(1.0/(K_a*K_b)-1.0/K_b)
                    ErrorWij_after_ising[amino_j,amino_i]+=error_Wij_before_ising[amino_i,amino_j]*abs(1.0/(K_a*K_b)-1.0/K_b-1.0/K_a+1.0)


            
    #verify that the matrix is not null
    assert np.allclose(ErrorWij_after_ising,np.zeros_like(ErrorWij_after_ising))==False
    return ErrorWij_after_ising

def ErrorWij_frobenius(error_Wij_after_ising,couplings_after_ising,couplings_after_frobenius,L,K,length_prot1):
    print("Error Wai,bj frobenius...")
    couplings_after_ising=0.5*(couplings_after_ising+couplings_after_ising.T) #force it to be symmetric
    error_Wij_after_ising=0.5*(error_Wij_after_ising+error_Wij_after_ising.T) #force it to be symmetric
    assert np.allclose(error_Wij_after_ising,np.zeros_like(error_Wij_after_ising))==False
    #verify if the matrix is symmetric
    assert np.allclose(couplings_after_ising,couplings_after_ising.T) #np.allclose is used to compare two arrays
    assert np.allclose(error_Wij_after_ising,error_Wij_after_ising.T) #np.allclose is used to compare two arrays
    #assert-> if the condition is false, it will raise an error
    error_Wab=np.zeros((L,L))
    if length_prot1==0:
        for amino_a in tqdm(range(0,L*K)):
            a=amino_a//K
            for amino_b in range(0,L*K):
                b=amino_b//K
                if couplings_after_ising[amino_a,amino_b]!=0:
                    error=abs(couplings_after_ising[amino_a,amino_b])*error_Wij_after_ising[amino_a,amino_b]
                    error_Wab[a,b]+=error
        assert np.allclose(error_Wab,error_Wab.T) #after the first forloop ->oke!
        for a in range(0,L):
            for b in range(0,L):
                if error_Wab[a,b]!=0 and couplings_after_frobenius[a,b]!=0:
                    error_Wab[a,b]=error_Wab[a,b]/couplings_after_frobenius[a,b]
    else:
        #only attribute it for (amino_i,amino_j) in [0:length_prot1,length_prot1:] and [length_prot1:,0:length_prot1]
        for amino_i in tqdm(range(0,length_prot1*K)):
            for amino_j in range(length_prot1*K,L*K):
                a=amino_i//K
                b=amino_j//K
                if couplings_after_ising[amino_i,amino_j]!=0:
                    error=abs(couplings_after_ising[amino_i,amino_j])*error_Wij_after_ising[amino_i,amino_j]
                    error_Wab[a,b]+=error
                    error_Wab[b,a]+=error

        for a in range(0,length_prot1):
            for b in range(length_prot1,L):
                if error_Wab[a,b]!=0 and couplings_after_frobenius[a,b]!=0:
                    error_Wab[a,b]=error_Wab[a,b]/couplings_after_frobenius[a,b]
                    error_Wab[b,a]=error_Wab[a,b]

    
    assert np.allclose(error_Wab,error_Wab.T) #after the second forloop
    assert np.allclose(error_Wab,np.zeros_like(error_Wab))==False
    return error_Wab

def ErrorWij_average_product(errorWab,couplings_after_average_product,L,length_prot1):
    assert np.allclose(errorWab,errorWab.T) #oke
    print("Error Wai,bj average product...")
    sum_fixed_alpha_over_beta=np.zeros(L)
    sum_fixed_beta_over_alpha=np.zeros(L)
    for amino_id in range(L):
        sum_fixed_alpha_over_beta[amino_id]=np.sum(couplings_after_average_product[amino_id,:])
        sum_fixed_beta_over_alpha[amino_id]=np.sum(couplings_after_average_product[:,amino_id])
        #should find the same if symmetric
        assert np.allclose(sum_fixed_alpha_over_beta[amino_id],sum_fixed_beta_over_alpha[amino_id]) #oke
    sum_over_alpha_over_beta=np.sum(sum_fixed_alpha_over_beta) #constant
    errorWab_new=np.zeros((L,L))
    if length_prot1==0:
        for alpha in tqdm(range(L)):
            for beta in range(alpha,L,1):
                errorWab_new[alpha,beta]=abs(1-2*sum_fixed_alpha_over_beta[alpha]*(sum_over_alpha_over_beta-sum_fixed_alpha_over_beta[alpha]))*errorWab[alpha,beta]
                errorWab_new[alpha,beta]+=(2*sum_fixed_alpha_over_beta[alpha]*(sum_over_alpha_over_beta-2*sum_fixed_alpha_over_beta[alpha]))*errorWab[alpha,beta]
                errorWab_new[alpha,beta]+=(sum_fixed_alpha_over_beta[alpha]**2/(sum_over_alpha_over_beta**2))*errorWab[alpha,beta]
                errorWab_new[beta,alpha]=errorWab_new[alpha,beta]
    else:
        for alpha in tqdm(range(0,length_prot1)):
            for beta in range(length_prot1,L):
                errorWab_new[alpha,beta]=abs(1-2*sum_fixed_alpha_over_beta[alpha]*(sum_over_alpha_over_beta-sum_fixed_alpha_over_beta[alpha]))*errorWab[alpha,beta]
                errorWab_new[alpha,beta]+=(2*sum_fixed_alpha_over_beta[alpha]*(sum_over_alpha_over_beta-2*sum_fixed_alpha_over_beta[alpha]))*errorWab[alpha,beta]
                errorWab_new[alpha,beta]+=(sum_fixed_alpha_over_beta[alpha]**2/(sum_over_alpha_over_beta**2))*errorWab[alpha,beta]
                errorWab_new[beta,alpha]=errorWab_new[alpha,beta]
    assert np.allclose(errorWab_new,errorWab_new.T)
    assert np.allclose(errorWab_new,np.zeros_like(errorWab_new))==False
    return errorWab_new

#####################################################################################################
#####################################################################################################
#####################################################################################################


    
    


#####################################################################################################
####################### functions to extract the couplings ##########################################
#####################################################################################################
def prediction(input, model, model_type) :
    #for a linear model, the couplings are directly the weights learned
    if model_type == "linear" :
        return np.array(model.masked_linear.linear.weight.detach().cpu())
    if model_type == "non-linear" : return model.non_linear(input)
    else : print("error with model type")   

def extract_couplings_W3(model, model_type, original_shape, indexes_batch,length_prot1) :
    (L,K) = original_shape
    #----------------------(1)--------------------------
    #extract one prediction for only zeros to have L*K output Z_ia=W1_ia
    input=torch.zeros(L*K)
    pred_0_1seq=prediction(torch.zeros(L*K), model, model_type)
    pred_0_1seq=pred_0_1seq.detach() #dimension (1,L*K)

    #---------------------(2)---------------------------
    #take two amino acids j*K+b and k*K+y that are one and the others zero 
    input[list(indexes_batch)]=1.
    #extract one prediction that consit of L*K output Z_ia_kb_ky= W1_ia + W2_ia_jb + W2_ia_ky + W3_ia_kb_ky 
    output_seq=prediction(input, model, model_type)
    output_seq=output_seq.detach() #dimension (1,L*K)

    #---------------------(3)---------------------------
    #we want to find W3_ia_kb_ky = Z_ia_kb_ky - W1_ia - W2_ia_jb - W2_ia_ky
    #we need to predict W2_ia_jb and W2_ia_ky
    input_j=torch.zeros(L*K)
    #alpha for j is the first elements of indexe_batch
    input_j[indexes_batch[0]]=1
    output_j=prediction(input_j, model, model_type)
    output_j=output_j.detach() #DIMENSION: (1,L*K)
    W_ia_jbeta=output_j-pred_0_1seq

    #we need to predict W_ia_kgamma
    input_k=torch.zeros(L*K)
    input_k[indexes_batch[1]]=1
    output_k=prediction(input_k, model, model_type)
    output_k=output_k.detach() #DIMENSION: (1,L*K)
    W_ia_kgamma=output_k-pred_0_1seq
    
    W_ialpha_jbeta_kgamma=output_seq-W_ia_jbeta-W_ia_kgamma-pred_0_1seq #DIMENSION: (1,L*K) 

    return W_ialpha_jbeta_kgamma

def extract_couplings(model, model_type, original_shape,data_per_col,path,device,length_prot1) :
   
    (L,K) = original_shape
    if model_type=="non-linear":
        print("Extract couplings for non-linear model...")
        print("Path to stock the W2:",path)
        path=os.path.join(path,"W2")
        if not os.path.exists(path):
            os.makedirs(path)
        #check if W2_init does not exist
        if not os.path.exists(os.path.join(path,"W2_init.npy")):
            print("Extract W2_init...")
            #(1)
            #L*K prediction of W1, shape (L*K,L*K)
            #each predictions will give the L*K output Zi,alpha=W0,i,alpha for every possible i and alpha
            zeros = torch.zeros((L*K,L*K)) 
            pred_zero = prediction(zeros, model, model_type)
            pred_zero = pred_zero.detach()
            #(2)
            #L*K prediction of W1 by taken only one amino acid at a time, shape (L*K,L*K)
            #the predictions will give Zi,alpha=W1,i,alpha+W1,i,alpha,j,beta for every possible i and alpha and j and beta
            #By taken a0,0, a0,1, a0,2, ..., aL,K -> The first prediction will give all Zi,alpha=W1,i,alpha+W2,i,alpha,0,0 for every possible i and alpha
            #-> The second prediction will give all Zi,alpha=W1,i,alpha+W2,i,alpha,0,1 for every possible i and alpha and so one
            only_one_amino=torch.eye(L*K) 
            #extract the prediction of W_ij such that i is the first axis and j the second axis
            pred_Wij = prediction(only_one_amino, model, model_type)
            pred_Wij = pred_Wij.detach()
            W2=pred_Wij-pred_zero
            #export W2 before the correction of the non-linear part. W2 is already defined above
            np.save(os.path.join(path,"W2_init"),W2.detach().cpu().numpy())
            del W2
            del pred_zero
            del pred_Wij
            torch.cuda.empty_cache()
            gc.collect()
        

        batch_size=20 #only look W_ialpha_jbeta_kgamma for j=[0,1,2,batch_size-1] and k=[0,1,2,batch_size-1], and then for j=[batch_size,5,6,2batch_size-1] and k=[batch_size,5,6,2batch_size-1] and so on
        print("Batch_size:",batch_size)
        #create a directory to save the W2 after each batch with same path as the model
        
        
        
        print("Extract the different combinaison to have only two ones per sequence...")
        indexes=[]
        #W_3_2d=torch.zeros(L*K,L*K)
        for idx in range(0,L*K,K):
            x=list(range(idx,idx+K))
            y=list(range(idx+K,L*K))
            indexes.extend( list(np.array(np.meshgrid(x,y)).T.reshape(-1,2)) ) #pairs of columns
        #example: L=6, K=2: indexes=[[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]]
        #for index in indexes[step:step+batch_size]:
        print("Extraction of W3 and W2 with batch (to avoid memory error)...")
        print("This opperation can take a long time, please do something else by waiting...")
        for id in range(0,L*K,batch_size*K):
            #check if we already have the W2
            if os.path.exists(os.path.join(path,"W2_0-"+str(id+batch_size*K-1)+".npy")):
                #W2=np.load(os.path.join(path,"W2_0-"+str(id+batch_size*K-1)+".npy"))
                continue
            else:   
                #extract the W2 before this one:
                if id-batch_size*K>=0:
                    W2=np.load(os.path.join(path,"W2_0-"+str(id-1)+".npy"))
                    #convert W2 to a tensor
                    W2=torch.tensor(W2,device=device)
                else:
                    #load W2_init
                    W2=np.load(os.path.join(path,"W2_init.npy"))
                    #convert W2 to a tensor
                    W2=torch.tensor(W2,device=device)
                #if it is the last term we can have a dimension smaller than batch_size*K
                if id+batch_size*K>L*K:
                    batch_size=L*K-id
                #print the percentage of progression percentage of id/(batch_size*K)*100
                Progression=id/(L*K)*100
                if Progression%5==0:
                    print("Progression:",Progression,"%")
                #need to keep only the indexes_batch in the specific batch
                #we will only keep the indexes of the pairs (j,k) such that j>id and k<id+batch_size*K (we look only the half of the matrix since symmetric)
                indexes_batch=np.array(indexes)[np.where(np.array(indexes)[:,1]<id+batch_size*K)[0]]
                indexes_batch=np.array(indexes_batch)[np.where(np.array(indexes_batch)[:,0]>=id)[0]]
                if Progression%5==0:
                    print("Extract W3...")
                W_3_batch=torch.zeros(L*K,batch_size*K,batch_size*K) #Creation of W3_ialpha_jbeta_kgamma
                for indexe_batch in indexes_batch:
                    #need to translate by indexes_batch[0,0] to have the position in the batch
                    indexe0=indexe_batch[0]-indexes_batch[0,0]
                    indexe1=indexe_batch[1]-indexes_batch[0,0]
                    #extraction of the couplings, we extract the L*K terms W_ialpha_jbeta_kgamma for every i, alpha, and jbeta given by indexe_batch[0], and kgamma given by indexe_batch[1] 
                    W_ialpha_jbeta_kgamma=extract_couplings_W3(model, model_type, (L,K), indexe_batch,length_prot1) #DIMENSION: (1,L*K) 
                    W_3_batch[:,indexe0,indexe1]=W_ialpha_jbeta_kgamma #DIMENSION: (L*K,batch_size*K,batch_size*K)
                    #now we need to complete the other half of the matrix
                    W_3_batch[:,indexe1,indexe0]=W_ialpha_jbeta_kgamma #DIMENSION: (L*K,batch_size*K,batch_size*K)
                    #and the terms not present correspond when j=k which is not possible
                
                if Progression%5==0:
                    print("Ising gauge on W3...")  
                    print("shape W_3_batch before ising :",W_3_batch.shape)  
                #now we need to apply ising gauge on W_3_batch
                W_3_batch=ising_gauge_W3(W_3_batch, (L,K),indexes_batch,batch_size,data_per_col,device,Progression) # shape of W_3_batch: (L*K,batch_size*K,batch_size*K)
                if Progression%5==0:
                    print("Extract W2...")

                #for indexe_batch in indexes_batch:
                sum_a=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
                sum_b=torch.zeros((L*K,batch_size*K,batch_size*K),device=device) #couplings is 3 dim now
                sum_y=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
                for a in range(0, L * K):  
                    i=a//K #exemple: i=0, K=2, k=0 | i=1, K=2, k=0 | i=2, K=2, k=1 | i=3, K=2, k=1
                    k=a%K
                    mask=torch.tensor(data_per_col[k,i]!=1,device=device)
                    sum_a[a,:,:]=torch.where(mask,torch.sum(W_3_batch[i:i+K,:,:],axis=0),0) #compute the sum over the a
                for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                    #b represent the true position of the amino acid in the sequence LK
                    j=b//K
                    k=b%K
                    mask=torch.tensor(data_per_col[k,j]!=1,device=device)
                    #in our batch the index 0 start by indexes_batch[0,0]
                    b_batch=b-indexes_batch[0,0]
                    j_batch=b_batch//K
                    sum_b[:,b_batch,:]=torch.where(mask,torch.sum(W_3_batch[:,j_batch:j_batch+K,:],axis=1),0)
                    #it is identical for the other axis (don't need to do again a forloop)
                    sum_b[:,:,b_batch]=torch.where(mask,torch.sum(W_3_batch[:,:,j_batch:j_batch+K],axis=2),0)
                
                for a in range(L*K):
                    K_a=np.sum(data_per_col[:,a//K]==0)
                    for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                        K_b=np.sum(data_per_col[:,b//K]==0)
                        b_batch=b-indexes_batch[0,0]
                        for y in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                            K_y=np.sum(data_per_col[:,y//K]==0)
                            y_batch=y-indexes_batch[0,0]
                            W2[a,b]=W2[a,b]+sum_a[a,b_batch,y_batch]/K_a+sum_b[a,b_batch,y_batch]/K_b+sum_y[a,b_batch,y_batch]/K_y
                W2=W2.detach().cpu().numpy()
                np.save(os.path.join(path,"W2_0-"+str(id+batch_size*K-1)),W2.detach().cpu().numpy())
                del W2
                del W_3_batch
                del sum_a
                del sum_b
                del sum_y
                torch.cuda.empty_cache()
            
        #extract W2
        W2=np.load(os.path.join(path,"W2_0-"+str(L*K-1)+".npy"))
        
        
        
    if model_type=="linear":
        print("Extract couplings for linear model...")
        W2 = prediction(torch.zeros(L*K), model, model_type)
        if length_prot1!=0:
            assert np.allclose(W2[0:length_prot1*K,0:length_prot1*K],np.zeros_like(W2[0:length_prot1*K,0:length_prot1*K]))
            assert np.allclose(W2[length_prot1*K:,length_prot1*K:],np.zeros_like(W2[length_prot1*K:,length_prot1*K:]))
    return W2
#####################################################################################################
#####################################################################################################

#####################################################################################################
####################### functions to apply the ising gauge ##########################################
#####################################################################################################

######################################### 3D #########################################################
def ising_gauge_W3(couplings, original_shape,indexes_batch,batch_size,data_per_col,device,Progression):
    (L,K) = original_shape
    #------------------ (1) remove the single sums -----------------
    # we need to remove the sum over a of C_ia,jb,ky
    # we need to remove the sum over b of C_ia,jb,ky
    # we need to remove the sum over y of C_ia,jb,ky
    sum_a=torch.zeros((L*K,batch_size*K,batch_size*K),device=device) #couplings is 3 dim now
    sum_b=torch.zeros((L*K,batch_size*K,batch_size*K),device=device) #couplings is 3 dim now
    sum_y=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
    new_couplings_3d=couplings.clone().to(device)
    if Progression%5==0:
        print("new_couplings_3d shape:",new_couplings_3d.shape)
    
        #indexes_batch is for example i=4,5,6 
        print("------------------------ sum over a of C_ia,jb,ky -----------------------------")
    for a in range(0, L * K):  
        i=a//K #exemple: i=0, K=2, k=0 | i=1, K=2, k=0 | i=2, K=2, k=1 | i=3, K=2, k=1
        k=a%K
        mask=torch.tensor(data_per_col[k,i]!=1,device=device)
        sum_a[a,:,:]=torch.where(mask,torch.sum(new_couplings_3d[i:i+K,:,:],axis=0),0)
    if Progression%5==0:
        print("--------- sum over b of C_ia,jb,ky, and sum over c of C_ia,jb,ky --------------")
    for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
        #b represent the true position of the amino acid in the sequence LK
        j=b//K
        k=b%K
        mask=torch.tensor(data_per_col[k,j]!=1,device=device)
        #in our batch the index 0 start by indexes_batch[0,0]
        b_batch=b-indexes_batch[0,0]
        j_batch=b_batch//K
        sum_b[:,b_batch,:]=torch.where(mask,torch.sum(new_couplings_3d[:,j_batch:j_batch+K,:],axis=1),0)
        #it is identical for the other axis (don't need to do again a forloop)
        sum_b[:,:,b_batch]=torch.where(mask,torch.sum(new_couplings_3d[:,:,j_batch:j_batch+K],axis=2),0)


    #------------------ (2) remove the double sums -----------------
    sum_ab=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
    sum_ay=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
    sum_by=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
    if Progression%5==0:
        print("-----------------------  sum over a,b of C_ia,jb,ky  --------------------------")
        print("-----------------------  sum over a,y of C_ia,jb,ky  --------------------------")
    for a in range(0,L*K):
        for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
            #b represent the true position of the amino acid in the sequence LK
            k_a=a%K
            k_b=b%K
            i=a//K
            j=b//K
            mask_a=torch.tensor(data_per_col[k_a,i]!=1,device=device)
            mask_b=torch.tensor(data_per_col[k_b,j]!=1,device=device)
            #in our batch the index 0 start by indexes_batch[0,0]
            b_batch=b-indexes_batch[0,0]
            sum_ab[a,b_batch,:]=torch.where(mask_a&mask_b,torch.sum(sum_b[i:i+K,b_batch,:],axis=0),0)
            sum_ay[a:,b_batch]=torch.where(mask_a&mask_b,torch.sum(sum_y[i:i+K,:,b_batch],axis=0),0)

    if Progression%5==0:
        print("----------------------- sum over b,y of C_ia,jb,ky --------------------------")
    for y in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
        for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
            #y and b represent the true positions of the amino acids in the sequence LK
            k=y//K
            j=b//K
            k_y=y%K
            k_b=b%K
            mask_y=torch.tensor(data_per_col[k_y,k]!=1,device=device)
            mask_b=torch.tensor(data_per_col[k_b,j]!=1,device=device)
            #in our batch the index 0 start by indexes_batch[0,0]
            y_batch=y-indexes_batch[0,0]
            b_batch=b-indexes_batch[0,0]
            j_batch=b_batch//K
            
            sum_by[:,b_batch,y_batch]=torch.where(mask_b&mask_y,torch.sum(sum_y[:,j_batch:j_batch+K,y_batch],axis=1),0)

    sum_aby=torch.zeros((L*K,batch_size*K,batch_size*K),device=device)
    if Progression%5==0:
        print("--------- sum over a,b,y of C_ia,jb,ky -----------")
    for a in range(0,L*K):
        for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
            for y in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                #y and b represent the true positions of the amino acids in the sequence LK
                mask_a=torch.tensor(data_per_col[a%K,a//K]!=1,device=device)
                mask_b=torch.tensor(data_per_col[b%K,b//K]!=1,device=device)
                mask_y=torch.tensor(data_per_col[y%K,y//K]!=1,device=device)
                #in our batch the index 0 start by indexes_batch[0,0]
                y_batch=y-indexes_batch[0,0]
                b_batch=b-indexes_batch[0,0]
                sum_aby[a,b_batch,y_batch]=torch.where(mask_a & mask_b & mask_y,torch.sum(sum_by[a:a+K,b_batch,y_batch],axis=0),0)

    if Progression%5==0:
        print("--------- computing the new_couplings C_ia,jb,ky-----------")
    for a in range(L*K):
        #if a%((L*K)/8)==0:
        #    print("Progression:",a/(L*K)*100,"%")
        for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
            for y in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                #y and b represent the true positions of the amino acids in the sequence
                K_a=np.sum(data_per_col[:,a//K]==0)
                K_b=np.sum(data_per_col[:,b//K]==0)
                K_y=np.sum(data_per_col[:,y//K]==0)
                #in our batch the index 0 start by indexes_batch[0,0]
                y_batch=y-indexes_batch[0,0]
                b_batch=b-indexes_batch[0,0]
                new_couplings_3d[a,b_batch,y_batch]=new_couplings_3d[a,b_batch,y_batch]-sum_a[a,b_batch,y_batch]/K_a-sum_b[a,b_batch,y_batch]/K_b-sum_y[a,b_batch,y_batch]/K_y+sum_ab[a,b_batch,y_batch]/(K_a*K_b)+sum_ay[a,b_batch,y_batch]/(K_a*K_y)+sum_by[a,b_batch,y_batch]/(K_b*K_y)-sum_aby[a,b_batch,y_batch]/(K_a*K_b*K_y)
    #DIMENSION new_couplings_3d=(LK,batch_size*K,batch_size*K)
    return new_couplings_3d
    
######################################### 2D #########################################################
def ising_gauge(couplings, original_shape, data_per_col,length_prot1) :
    """
    apply Ising gauge on the coupling coefficients
    Inputs:
            couplings       ->      the coupling coefficients
                                    type: numpy array, shape: (L*K,L*K)
            original_shape  ->      the original shape of the couplings
                                    type: tuple, shape: (L,K)
            data_per_col    ->      the data per column
                                    type: numpy array, shape: (K,L) 
            length_prot1    ->      the length of the first protein (if we used a cross-linear model)
    Ouputs:
            new_couplings   ->      the coupling coefficients after applying the Ising gauge
                                    type: numpy array, shape: (L*K,L*K)


    """
    (L,K) = original_shape
    print("original_shape:",original_shape)

    
    new_couplings = np.copy(couplings)
    
    #verify that couplings.shape=(L*K,L*K)
    try:
        assert couplings.shape==(L*K,L*K)
    except:
        print(f"Error: couplings.shape should be ({L*K},{L*K}) but is {couplings.shape}")
        return
    
    print("--------- sum over beta of C_ialpha,jbeta and sum over aalpha of C_ialpha,jbeta -----------")
    sum_alpha_fixed_beta_from_0_to_k=np.zeros((L*K,L)) #in row the position alpha, in column the position j
    sum_beta_fixed_alpha_from_0_to_k=np.zeros((L*K,L)) #in row the position beta, in column the position i
    for alpha in range(0, L * K):  #new 22mai
        for j in range(0, L):
            sum_alpha_fixed_beta_from_0_to_k[alpha,j]=np.sum(couplings[alpha,j*K:(j*K+K)]) #sum over b for a fixed a
            #we have also the inverse for alpha -> beta and j -> i
            sum_beta_fixed_alpha_from_0_to_k[alpha,j]=np.sum(couplings[j*K:(j*K+K),alpha]) #sum over a for a fixed b
            
    print("--------- sum over alpha,beta of C_ialpha,jbeta -----------")
    sum_alpha_from_0_to_k_beta_from_0_to_k=np.zeros((L,L)) #in row the position i, in column the position j
    for i in range(0, L):
        for j in range(0,L):
            sum_alpha_from_0_to_k_beta_from_0_to_k[i,j]=np.sum(sum_alpha_fixed_beta_from_0_to_k[i*K:(i*K+K),j]) #sum over alpha of every sum_alpha_fixed_beta_from_0_to_k

    print("--------- computing the new_couplings -----------")
    if length_prot1==0:
        for alpha in range(L * K):
            for beta in range(L * K):
                i=alpha//K
                j=beta//K
                K_alpha = np.sum(data_per_col[:, i] == 0)
                K_beta = np.sum(data_per_col[:, j] == 0)
                if data_per_col[alpha % K, i] == 0 and data_per_col[beta % K, j] == 0:
                    new_couplings[alpha, beta] = couplings[alpha,beta]-sum_alpha_fixed_beta_from_0_to_k[alpha,j]/K_alpha-sum_beta_fixed_alpha_from_0_to_k[beta,i]/K_beta+sum_alpha_from_0_to_k_beta_from_0_to_k[i,j]/(K_alpha*K_beta)
                else:
                    new_couplings[alpha, beta] = 0         
    else:
        for alpha in range(0,length_prot1*K):
            for beta in range(length_prot1*K,L):
                i=alpha//K
                j=beta//K
                K_alpha = np.sum(data_per_col[:, i] == 0)
                K_beta = np.sum(data_per_col[:, j] == 0)
                if data_per_col[alpha % K, i] == 0 and data_per_col[beta % K, j] == 0:
                    new_couplings[alpha, beta] = couplings[alpha,beta]-sum_alpha_fixed_beta_from_0_to_k[alpha,j]/K_alpha-sum_beta_fixed_alpha_from_0_to_k[beta,i]/K_beta+sum_alpha_from_0_to_k_beta_from_0_to_k[i,j]/(K_alpha*K_beta)
                    new_couplings[beta, alpha] = new_couplings[alpha, beta]
                else:
                    new_couplings[alpha, beta] = 0
                    new_couplings[beta, alpha] = 0
                                 
    return new_couplings
#####################################################################################################
#####################################################################################################


#####################################################################################################
####################### functions to correct the final couplings ####################################
#####################################################################################################
@jit(nopython=True, parallel=True) #parallelise using numba
def average_product_correction(f,length_prot1) :
    """
    apply the average product correction on the couplings
    input:
            f           ->      the couplings
                                type: numpy array, shape: (L*K,L*K)
            length_prot1->      the length of the first protein (if we used a cross-linear model)
                                type: int
    """
    sum_on_i=np.sum(f,0)
    sum_on_j=np.sum(f,1)
    sum_on_j_and_i=np.sum(f)
    if length_prot1==0:
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                f[i,j]=f[i,j]-sum_on_i[j]*sum_on_j[i]/sum_on_j_and_i
    else:
        for i in range(0,length_prot1):
            for j in range(length_prot1,f.shape[1]):
                f[i,j]=f[i,j]-sum_on_i[j]*sum_on_j[i]/sum_on_j_and_i
                f[j,i]=f[j,i]-sum_on_i[i]*sum_on_j[j]/sum_on_j_and_i
    print("verify that average product is symmetric")
    assert np.allclose(f,f.T)
    return f
#####################################################################################################
#####################################################################################################
    
###########################################################################################################################################
#********************************************** MAIN FUNCTION *****************************************************************************
########################################################################################################################################### 
def couplings(model_name, length_prot1=0, number_model=1, type_average='average_couplings', output_name='/', figure=False, data_per_col='/', model_type="linear",L=0,K=0) :
    print("-------- Welcome in the couplings process :) --------")
    print("-----------------------------------------------------")
    use_cuda=False #we don't use cuda for the moment
    device = torch.device("cuda" if use_cuda else "cpu")
    ###################################################################################
    ################# EXTRACTION OF THE MODEL(S) PATH(S) ##############################
    ############ & data_per_col (the same for every model(s)) #########################
    ########################## & output_name ##########################################
    ###################################################################################
    print("-----------------------------------------------------")
    print("Extraction of the parameters...")
    number_model=int(number_model)
    #extract the folder name (first part)
    path_folder=os.path.dirname(model_name)
    if number_model==1:
        print("one model")
        index_model=os.path.basename(model_name).split('_')[-1]
    #load the data_per_col .txt file and convert it into a numpy array
    if data_per_col=="/":
        #take the path of the model_name and add the name 'data_per_col.txt' to it
        data_per_col = path_folder
        data_per_col = os.path.join(data_per_col, 'data_per_col.txt')
        print("take the default data_per_col file: ", data_per_col)
    data_per_col=np.loadtxt(data_per_col)
    if K==0:
        print("data_per_col.shape:",data_per_col.shape)
        K=data_per_col.shape[0]
        if K>21:
            name_info='INFOS_with_tax.txt'
        else:
            name_info='INFOS_no_tax.txt'
    elif K!=data_per_col.shape[0]:
        print("The data_per_col has a wrong shape, please check it (K should be",data_per_col.shape[0],")")
        return
    
    if output_name=="/":
        #take the path of the model_name and add the name 'couplings' to it
        output_name = os.path.dirname(model_name)
        output_name = os.path.join(output_name, 'couplings')
    ###################################################################################
    ###################################################################################

    ###################################################################################
    ######################### EXTRACTION OF L AND K ###################################
    ###################################################################################
    if L==0:
        try:
            #find the element in model_name.split('/') starting with prep:
            threshold_preprocessed=[x for x in model_name.split('/') if x.startswith('preprocessing')][0]
            threshold_preprocessed=threshold_preprocessed.split('-')[-1]
            path_info=model_name.split('/preprocessing')[0]
            #convert the list into path
            path_info=os.path.join(path_info, name_info)
            
            #check if endswith "gaps"
            if threshold_preprocessed.endswith("gaps"):
                threshold_preprocessed=threshold_preprocessed[:-4] #remove the "gaps" at the end
        except:
            pass
        try:
            #check that threshold_preprocessed is composed of digits and a dot
            
            threshold_preprocessed=float(threshold_preprocessed)
            
        except:
            threshold_preprocessed=input("Please enter the threshold preprocessed used:  ")
            threshold_preprocessed=float(threshold_preprocessed)
            path_info=input("Please enter the path where to find the file with the information about N,L,K:  ")

        #we don't know if it is with tax or not
        with open(os.path.join(path_info), 'r') as file:
            print("in the file:", path_info)
            lines = file.readlines()
            print(f"preprocessing with gap threshold of {threshold_preprocessed*100} %")
            for i, line in enumerate(lines):
                
                if line==f"preprocessing with gap threshold of {threshold_preprocessed*100} %\n":
                    L=lines[i+1].split(',')[3]
                    L=int(L)
                    
                    K_good=lines[i+1].split(',')[4]
                    K_good=K_good.split(')')[0]
                    K_good=int(K_good)
                    if K_good!=K:
                        print(f"The data_per_col has a wrong shape, please check it (K should be {K_good})")
                        return
                    break
    print("L,K:(",L,",",K,")")
    if K>21:
        print("K=",K,"-> we have the taxonomy for each sequence. In this case the L is the length of a sequence +1.")
    
    ###################################################################################
    ###################################################################################
    
    ###################################################################################
    ################# EXTRACTION OF LENGTH_PROT1 AND LENGHT_PROT2 #####################
    ############################# (if two proteins) ###################################
    ###################################################################################
    if length_prot1!=0:
        length_prot2=L-length_prot1
        print("---------------------------- TWO PROTEINS --------------------------------")
        print("Length of the first protein:",length_prot1)
        print("Length of the second protein:",length_prot2)
        print("--------------------------------------------------------------------------")
    ###################################################################################
    ###################################################################################
    
    
    
    ###################################################################################
    ################# WEIGHT EXTRACTION AND ISING GAUGE FOR THE MODEL(S) ##############
    ################### (depend if we consider the taxonomy or not)####################
    ######################## (depend if it is linear or not)###########################
    ###################################################################################
    #createthe directory
    couplings_path = os.path.dirname(output_name) # Path for the couplings directory without the last 
    name_couplings_before = os.path.join(couplings_path, 'couplings_before_ising')
    name_couplings_after = os.path.join(couplings_path, 'couplings_after_ising')
    name_frobenius = os.path.join(couplings_path, 'couplings_frobenius')
    name_average_product = os.path.join(couplings_path, 'average_product')
    # saved the last part as the name of the file
    output_name = os.path.basename(output_name)

   

    if number_model==1:
        #add the name 'average-models' to the output_directory path
        output_directory = os.path.join(couplings_path, 'average-models/')
    else:
        if type_average=="average_couplings_frob":
            output_directory = os.path.join(couplings_path, 'average-models-and-frob/')
        elif type_average=="average_couplings":
            output_directory = os.path.join(couplings_path, 'average-couplings/')

    # Create the directory and its parent directories if they don't exist
    os.makedirs(name_couplings_before, exist_ok=True)
    os.makedirs(name_couplings_after, exist_ok=True)
    os.makedirs(name_frobenius, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(name_average_product, exist_ok=True)
    #########################################################################
    ############################ error calculation ##########################
    if number_model==1:
        path_error_Wij_before=os.path.join(name_couplings_before, "error_Wij_"+index_model+"_before_ising.txt")
        path_error_Wij_after= os.path.join(couplings_path, "couplings_after_ising/error_Wij_after_ising_"+index_model+".txt")
    else:
        path_error_Wij_before=os.path.join(name_couplings_before, "error_Wij_0-"+str(number_model-1)+"_before_ising.txt")
        path_error_Wij_after= os.path.join(couplings_path, "couplings_after_ising/error_Wij_after_ising_0-"+str(number_model-1)+".txt")
    if os.path.isfile(path_error_Wij_after) and os.path.isfile(path_error_Wij_before):
        print("error_Wij_after_ising and error_Wij_before_ising already exist we don't compute them again")
        print("path:",path_error_Wij_before)
        print("path:",path_error_Wij_after)
        average_error_after_Wij=np.loadtxt(path_error_Wij_after)
    else:
        find_average_error_Wij=False
        if number_model==1 or type_average=="average_couplings":
            if os.path.isfile(path_error_Wij_before):
                print("error_Wij_before_ising already exists we don't compute it again")
                average_error_Wij=np.loadtxt(path_error_Wij_before)
                average_error_after_Wij=ErrorWij_after_ising(average_error_Wij,data_per_col,L,K,length_prot1)
                print("shape of error_Wij_after:",average_error_after_Wij.shape)
                find_average_error_Wij=True
        if find_average_error_Wij==False:
            error_Wij_to_do=[]
            if number_model==1:
                name_err_before=os.path.join(name_couplings_before, "error_Wij_"+index_model+"_before_ising.txt")
                name_err_after=os.path.join(name_couplings_after, "error_Wij_"+index_model+"_after_ising.txt")
            for step in range(number_model):
                if number_model>1:
                    name_err_before=os.path.join(name_couplings_before, "error_Wij_"+str(step)+"_before_ising.txt")
                    name_err_after=os.path.join(name_couplings_after, "error_Wij_"+str(step)+"_after_ising.txt")
                    #if after not done we have to do before and after:
                    print(f"----------------------- step {step} -----------------------")
                    if not os.path.isfile(name_err_after):
                        print("error_Wij_after_ising not done yet")
                        
                    if not os.path.isfile(name_err_before):
                        print("error_Wij_before_ising not done yet")

                    if not os.path.isfile(name_err_after) or not os.path.isfile(name_err_before):
                        error_Wij_to_do.append(step)
                    else:
                        print("error_Wij_before_ising already exists we don't compute it again")
                        print("path:",name_err_before)
                        print("error_Wij_after_ising already exists we don't compute it again")
                        print("path:",name_err_after)

                    print("-------------------------------------------------------")
                else: #only one model and we know that it does not exist
                    error_Wij_to_do.append(0)

            if K>21:
                print("K=",K,"-> we have the taxonomy for each sequence")
                average_error_Wij=np.zeros(((L-1)*K,(L-1)*K))
                average_error_after_Wij=np.zeros(((L-1)*K,(L-1)*K))
            else:
                print("K=",K,"-> we don't have the taxonomy for each sequence")
                average_error_Wij=np.zeros((L*K,L*K))
                average_error_after_Wij=np.zeros((L*K,L*K))
            for step in range(number_model):
                if step in error_Wij_to_do:
                    if number_model>1:
                        name_err_before=os.path.join(name_couplings_before, "error_Wij_"+str(step)+"_before_ising.txt")
                        name_err_after=os.path.join(name_couplings_after, "error_Wij_"+str(step)+"_after_ising.txt")
                    else:
                        name_err_before=os.path.join(name_couplings_before, "error_Wij_"+index_model+"_before_ising.txt")
                        name_err_after=os.path.join(name_couplings_after, "error_Wij_"+index_model+"_after_ising.txt")
                    print("extraction of the errors for the model: ", step+1, "/", number_model)
                    if number_model>1:
                        model=torch.load(model_name+'_' + str(step))
                    else:
                        model=torch.load(model_name)
                    error_Wij=ErrorWij_before_ising(model,L,K,device,length_prot1)
                    error_Wij_after=ErrorWij_after_ising(error_Wij,data_per_col,L,K,length_prot1)
                    if K>21:
                        #with tax we need to remove the last blocs of size K
                        print("treatment of errors: remove the last column corresponding to the taxonomy type")
                        error_Wij=error_Wij[:-K,:-K]
                        error_Wij_after=error_Wij_after[:-K,:-K]
                        print("shape of error_Wij_after:",error_Wij_after.shape)
                    #save the couplings in the file couplings_before_ising.txt
                    if number_model==1:
                        np.savetxt(path_error_Wij_before, error_Wij)
                        np.savetxt(path_error_Wij_after, error_Wij_after)
                        print("error_Wij before ising gauge saved in the file: ", path_error_Wij_before)
                        print("error_Wij after ising gauge saved in the file: ", path_error_Wij_after)
                    else:
                        np.savetxt(name_err_before, error_Wij)
                        np.savetxt(name_err_after, error_Wij_after)
                        print("error_Wij before ising gauge saved in the file: ", name_err_before)
                        print("error_Wij after ising gauge saved in the file: ", name_err_after)
                    average_error_Wij += error_Wij
                    average_error_after_Wij += error_Wij_after
                    del error_Wij
                    del error_Wij_after
                    del model
                    gc.collect()
                else:
                    average_error_Wij += np.loadtxt(name_err_before)
                    average_error_after_Wij += np.loadtxt(name_err_after)
        average_error_Wij=average_error_Wij/number_model
        average_error_after_Wij=average_error_after_Wij/number_model
        np.savetxt(path_error_Wij_before, average_error_Wij)
        np.savetxt(path_error_Wij_after, average_error_after_Wij)
        print("average error_Wij before ising gauge saved in the file: ", path_error_Wij_before)     
        print("average error_Wij after ising gauge saved in the file: ", path_error_Wij_after)   
    #########################################################################
    #########################################################################


    #check if the couplings after ising already exist
    if number_model==1:
        name_to_check= os.path.join(couplings_path, "couplings_after_ising/couplings_after_ising_"+index_model+".txt")
    else:
        name_to_check= os.path.join(couplings_path, "couplings_after_ising/couplings_after_ising_0-"+str(number_model-1)+".txt")
 
    if os.path.isfile(name_to_check):
        print("couplings_after_ising already exists we don't compute it again")
        average_couplings=np.loadtxt(name_to_check)
        if K>21:
            L=L-1
    
    else:
        #check if it exist
        find_average_couplings=False
        if number_model==1:
            path_couplings_before=os.path.join(name_couplings_before, "couplings_"+index_model+"_before_ising.txt")
        else:
            path_couplings_before=os.path.join(name_couplings_before, "couplings_0-"+str(number_model-1)+"_before_ising.txt")
        if number_model==1 or type_average=="average_couplings":
            if os.path.isfile(path_couplings_before):
                print("couplings_before_ising.txt, we don't compute it again")
                average_couplings=np.loadtxt(path_couplings_before)
                find_average_couplings=True
                #print("shape of couplings before ising:", average_couplings.shape) #oke
                if K>21:
                    L=L-1 #lose a dimension with class
                    data_per_col=data_per_col[:,:-1] #remove the last column corresponding to the class type
                print('L,K:(',L,',',K,')')
                print("------------------------------------")
                
        if find_average_couplings==False: #can be type_average=average_couplings_frob (we need all couplings) or we don't have the couplings_before_ising.txt
            # Create the directory and its parent directories if they don't exist
            os.makedirs(name_couplings_before, exist_ok=True)
            # Check which couplings we need to compute
            couplings_ising_to_do=[]
            if number_model==1:
                name_couplings_before_step=os.path.join(name_couplings_before, "couplings_", index_model, "_before_ising.txt")
                if os.path.isfile(name_couplings_before_step):
                    print("couplings_", index_model, "_before_ising.txt already exists we don't compute it again")
                else:
                    print("couplings_", index_model, "_before_ising.txt does not exist, we have to compute it")
                    couplings_ising_to_do.append(0)
            else:
                for step in range(number_model):
                    name_couplings_before_step=os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt")
                    if os.path.isfile(name_couplings_before_step):
                        print(f"couplings_{step}_before_ising.txt already exist, we don't compute it again")
                    else:
                        print(f"couplings_{step}_before_ising.txt does not exist, we have to compute it")
                        os.makedirs(name_couplings_after, exist_ok=True)
                        couplings_ising_to_do.append(step)
                    print("------------------------------------")

            if K>21:
                print("K=",K,"-> we have the taxonomy for each sequence")
                average_couplings=np.zeros(((L-1)*K,(L-1)*K))
            else:
                print("K=",K,"-> we don't have the taxonomy for each sequence")
                average_couplings=np.zeros((L*K,L*K))
            #for step,model in enumerate(models):
            for step in range(number_model):
                if step in couplings_ising_to_do:
                    if number_model==1:
                        print("extraction of the couplings for the models: ", index_model)
                        model=torch.load(model_name)
                    else:
                        print("extraction of the couplings for the model: ", step+1, "/", number_model)
                        model=torch.load(model_name+'_' + str(step))
                    
                        
                    couplings=extract_couplings(model, model_type, (L,K),data_per_col,couplings_path,device,length_prot1)
                    ###################################################################################
                    if K>21:
                        #with tax we need to remove the last blocs of size K
                        print("treatment of couplings: remove the last column corresponding to the taxonomy type")
                        couplings=couplings[:-K,:-K]
                    
                    #save the couplings in the file couplings_before_ising.txt
                    if number_model>1:
                        np.savetxt(os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt"), couplings)
                        print("couplings before ising gauge saved in the file: ", os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt"))
                    else:
                        np.savetxt(os.path.join(name_couplings_before, "couplings_"+index_model+"_before_ising.txt"), couplings)
                        print("couplings before ising gauge saved in the file: ", os.path.join(name_couplings_before, "couplings_"+index_model+"_before_ising.txt"))
                    average_couplings += couplings
                    del couplings
                    del model
                    gc.collect()
                else:
                    if number_model==1:
                        name_coup=os.path.join(name_couplings_before, "couplings_"+index_model+"_before_ising.txt")
                    else:
                        name_coup=os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt")
                    average_couplings += np.loadtxt(name_coup)

            if K>21:
                L=L-1 #lose a dimension with class
                data_per_col=data_per_col[:,:-1] #remove the last column corresponding to the class type

            average_couplings=average_couplings/number_model
            np.savetxt(path_couplings_before, average_couplings)
            
            print("average couplings before ising gauge saved in the file: ", path_couplings_before)
        
        ###################################################################################
        ###################################################################################

        ###################################################################################
        ################# PLOT OF THE COUPLINGS BEFORE ISING ##############################
        ###################################################################################
        print("--------------------------------------------------------------------------")
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
        del average_couplings
        gc.collect()

        ##################################################################################
        ##################################################################################
            
        ##################################################################################
        ################# COMPUTATION OF THE ISING GAUGE #################################
        ##################################################################################
        print("--------------------------------------------------------------------------")
        print("------------ computation of ising gauge ------------")
        if number_model>1:
            #step=1
            if type_average=="average_couplings_frob" or type_average=="average_couplings":
                print(f"You have chosen the type_average={type_average}")
                if type_average=="average_couplings_frob":
                    print("we apply the average product correction on each couplings before averaging them")
                else:
                    print("we will average the couplings before applying the average product correction")
                
                
                average_couplings=np.zeros((L*K,L*K))
                #ALL_couplings_ising=[]
                print("Treatment of the ising gauge on each couplings...")
                #for couplings,model in zip(ALL_couplings,models):
                for step in range(number_model):
                    print("gauge process on model : ", step+1, "/", number_model)
                    #look if the file with path couplings_path and name couplings_ising_step.txt exists
                    #if it exists, load it and don't do ising_gauge again
                    #if it doesn't exist, do ising_gauge and save it
                    if os.path.isfile(os.path.join(name_couplings_after, "couplings_after_ising_" + str(step) + ".txt")):
                        print("couplings_after_ising_" + str(step) + ".txt already exists, we don't compute it again")
                        couplings = np.loadtxt(os.path.join(name_couplings_after, "couplings_after_ising_" + str(step) + ".txt"))
                        
                    else:
                        couplings=np.loadtxt(os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt"))
                        couplings = ising_gauge(couplings, (L,K), data_per_col,length_prot1)
                        np.savetxt(os.path.join(name_couplings_after, "couplings_after_ising_" + str(step) + ".txt"), couplings)
                        print("couplings after ising gauge saved in the file: ", os.path.join(name_couplings_after, "couplings_after_ising_" + str(step) + ".txt"))
                    #ALL_couplings_ising.append(couplings)
                    average_couplings += couplings
                    del couplings
                    gc.collect()
                    #step+=1
                average_couplings=average_couplings/number_model
                np.savetxt(os.path.join(name_couplings_after, "couplings_after_ising_0-"+str(number_model-1)+".txt"), average_couplings)
                print("average couplings after ising gauge saved in the file: ", os.path.join(name_couplings_after, "couplings_after_ising_0-"+str(number_model-1)+".txt"))
                
            else:
                print("error with type_average")
        else: # we do ising on the average of the models (number_model=1)
            print("You have chosen to treat only one model")

            #check if it exist
            if os.path.isfile(os.path.join(name_couplings_after, "couplings_after_ising_" + index_model + ".txt")):
                print("couplings_after_ising_" + index_model + ".txt already exists, we don't compute it again")
                average_couplings=np.loadtxt(os.path.join(name_couplings_after, "couplings_after_ising_" + index_model + ".txt"))
                #check that we have couplings[0:length_prot1,0:length_prot1] = 0 if length_prot1!=
                if length_prot1!=0:
                    assert np.all(average_couplings[0:length_prot1*K,0:length_prot1*K]==0)
                    assert np.all(average_couplings[length_prot1*K:,length_prot1*K:]==0)
            else:
                print("Treatment of the ising gauge on the couplings...")
                average_couplings=np.loadtxt(path_couplings_before)
                print("shape average_couplings before ising:", average_couplings.shape)
                average_couplings=np.copy(average_couplings)
                #print("shape data_per_col:", data_per_col.shape)
                average_couplings=ising_gauge(average_couplings,(L,K), data_per_col,length_prot1)
                # Create the directory and its parent directories if they don't exist
                os.makedirs(name_couplings_after, exist_ok=True)
                np.savetxt(os.path.join(name_couplings_after, "couplings_after_ising_" + index_model + ".txt"), average_couplings)
                print("couplings after ising gauge saved in the file: ", os.path.join(name_couplings_after, "couplings_after_ising_" + index_model + ".txt"))
                
            

    ##################################################################################
    ##################################################################################
    
    ###################################################################################
    ################# PLOT OF THE COUPLINGS AFTER ISING ###############################
    ###################################################################################
    print("--------------------------------------------------------------------------")
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
        plt.savefig(os.path.join(name_couplings_after, "couplings_after_ising.png"))
    else:
        print("No plot after the ising gauge, because figure=False")
    del average_couplings
    gc.collect()
    ##################################################################################
    ##################################################################################
    

    ##################################################################################
    ################################## FROBENIUS #####################################
    ##################################################################################
    '''
    There are 2 cases:
    (1) number_model=1:, number_model>1 and type_average="average_couplings"
        we apply the average product correction on the average_couplings
    (2) number_model>1 and type_average="average_couplings_frob":
        we apply the average product correction on each couplings before averaging them
    '''
    print("--------------------------------------------------------------------------")
    print("------------------ Frobenius and average correction ----------------------")
    
    #check if it exist
    #if os.path.isfile(os.path.join(name_frobenius,"couplings_frobenius_0-"+str(number_model-1)+".txt")):
    #    print("The average Frobenius norm already exists, we don't compute it again")
    #    print("You can find it at the path: ", os.path.join(name_frobenius,"couplings_frobenius_0-"+str(number_model-1)+".txt"))
    #else:
    n=0
    print("---------------- Frobenius ----------------")
    if number_model>1 and type_average=="average_couplings_frob":
        couplings_frobenius_to_do=[]
        for step in range(number_model):
            name_frobenius_step=os.path.join(name_frobenius, "couplings_frobenius_"+str(step)+".txt")
            if os.path.isfile(name_frobenius_step):
                print(f"couplings_{step}_before_ising.txt already exist, we don't compute it again")
            else:
                print(f"couplings_{step}_before_ising.txt does not exist, we have to compute it")
                couplings_frobenius_to_do.append(step)
            print("------------------------------------") 
        for step in range(number_model):
            if step in couplings_frobenius_to_do:
                couplings_old=np.loadtxt(os.path.join(name_couplings_after, "couplings_after_ising_"+str(step)+".txt"))
                couplings=0.5*(couplings_old + couplings_old.T)
                del couplings_old
                if length_prot1!=0:
                    assert np.allclose(couplings[0:length_prot1*K,0:length_prot1*K],np.zeros_like(couplings[0:length_prot1*K,0:length_prot1*K]))
                    assert np.allclose(couplings[length_prot1*K:,length_prot1*K:],np.zeros_like(couplings[length_prot1*K:,length_prot1*K:]))
            #reshape couplings in a L x L array where each element contains the K x K categorical couplings to apply frobenius norm on each element
                matrix = []
                print("New version 1may 2024: we don't count the gaps scores")
                for i in range(L) :
                    rows = []
                    for j in range(L) :
                        #rows.append(couplings[i*K:(i+1)*K, j*K:(j+1)*K])
                        #NEW VERSION 1TH MAY 2024: REMOVE THE GAPS -> element k=0
                        rows.append(couplings[(i*K+1):(i+1)*K, (j*K+1):(j+1)*K])
                        
                    matrix.append(rows)
                couplings = np.array(matrix)
                
                #frobenius norm
                couplings_frob = np.linalg.norm(couplings, 'fro', (2, 3)) #frobenius norm on each element of the array (2,3)
                
                np.savetxt(os.path.join(name_frobenius, "couplings_frobenius_"+str(step)+".txt"), couplings_frob)
                print("couplings frobenius saved in the file: ", os.path.join(name_frobenius, "couplings_frobenius_"+str(step)+".txt"))
            else:
                couplings_frob=np.loadtxt(os.path.join(name_frobenius, "couplings_frobenius_"+str(step)+".txt"))
            #average product correction
            couplings = average_product_correction(couplings_frob,length_prot1)
            if n==0:
                average_couplings=np.copy(couplings) #to initialise
            if n==1:
                average_couplings+=couplings

            del couplings_frob
            #reshape in form (0,1) (0,2) ... (1,2) (1,3) ...
            couplings_new = np.triu(couplings)
            tmp = []
            for i in range(L) :
                for j in range(i+1, L) :
                    tmp.append(couplings_new[i,j])
            couplings_new = np.array(tmp) 
            if n==0:
                average_couplings_new=np.copy(couplings_new) #to initialise
                n=1
            if n==1:
                average_couplings_new += couplings_new
            del couplings_new

            
            gc.collect()

        average_couplings_new = average_couplings_new/number_model
        average_couplings = average_couplings/number_model
    else: #average couplings or for number_model=1
        print("Treatment of the average product correction on the average couplings...")
        
        average_couplings=np.loadtxt(os.path.join(name_couplings_after, "couplings_after_ising_" + index_model + ".txt"))
        average_couplings=0.5*(average_couplings + average_couplings.T) #symmetrize the couplings
        print("shape of average_couplings before frobenius:", average_couplings.shape) #(L*K,L*K)
        assert np.allclose(average_couplings, average_couplings.T) #check if the average couplings before frobenius is symmetric
        #reshape couplings in a L x L array where each element contains the K x K categorical couplings to apply frobenius norm on each element
        matrix = []
        print("New version 1may 2024: we don't count the gaps scores")
        print("L:",L)
        for i in range(L) :
            rows = []
            for j in range(L) :
                #rows.append(average_couplings[i*K:(i+1)*K, j*K:(j+1)*K])
                #NEW VERSION 1TH MAY 2024: REMOVE THE GAPS -> element k=0
                rows.append(average_couplings[(i*K+1):(i+1)*K, (j*K+1):(j+1)*K])
                #if i==0 and j==4:
                #    print(average_couplings[(i*K+1):(i+1)*K, (j*K+1):(j+1)*K])
            matrix.append(rows)
               
        average_couplings = np.array(matrix)

        print("shape of average_couplings:", average_couplings.shape) #(L,L,K,K)
        average_couplings = np.linalg.norm(average_couplings, 'fro', (2, 3)) 
        print("verify that after linalg we have symmetry")
        assert np.allclose(average_couplings, average_couplings.T) #check if the matrix frobenius is symmetric


        if length_prot1!=0:
            assert np.allclose(average_couplings[0:length_prot1,0:length_prot1],np.zeros_like(average_couplings[0:length_prot1,0:length_prot1]))
            assert np.allclose(average_couplings[length_prot1:,length_prot1:],np.zeros_like(average_couplings[length_prot1:,length_prot1:]))
             
        
        np.savetxt(os.path.join(name_frobenius, "couplings_frobenius_"+index_model+".txt"), average_couplings)
        print("couplings frobenius saved in the file: ", os.path.join(name_frobenius, "couplings_frobenius_"+index_model+".txt"))
        average_couplings = average_product_correction(average_couplings,length_prot1)
        if length_prot1!=0:
            assert np.allclose(average_couplings[0:length_prot1,0:length_prot1],np.zeros_like(average_couplings[0:length_prot1,0:length_prot1]))
            assert np.allclose(average_couplings[length_prot1:,length_prot1:],np.zeros_like(average_couplings[length_prot1:,length_prot1:]))
        average_couplings_new = np.triu(average_couplings) #np.triu -> upper triangular part of the matrix
        tmp = []
        for i in range(L) : 
            for j in range(i+1, L) :
                tmp.append(average_couplings_new[i,j]) #(0,1), (0,2),...(1,2),(1,3),....(L-1,L) -> L*(L-1)/2 elements in total
        average_couplings_new = np.array(tmp)

    ##################################################################################
    ##################################################################################
    
    #################################################################################
    ################# SAVING THE COUPLINGS ##########################################
    #################################################################################
    
    #print the path of: os.path.join(output_directory, output_name)
    
    if number_model>1:
        output_name=output_name+"_0-"+str(number_model-1)+".txt"
        np.savetxt(os.path.join(name_average_product, "average_product_0-"+str(number_model-1)+".txt"), average_couplings)
        print("The average product correction is saved in the file: ", os.path.join(name_average_product, "average_product_0-"+str(number_model-1)+".txt"))
    else:
        output_name=output_name+"_"+index_model+".txt"
        np.savetxt(os.path.join(name_average_product, "average_product_"+index_model+".txt"), average_couplings)
        print("The average product correction is saved in the file: ", os.path.join(name_average_product, "average_product_"+index_model+".txt"))
    np.savetxt(os.path.join(output_directory,  output_name), average_couplings_new)
    print("The final couplings file is saved in the file: ", os.path.join(output_directory, output_name))
    print("---------------------------------- END -----------------------------------")
    print("--------------------------------------------------------------------------")
        #################################################################################
        #################################################################################
    #########################################################################
    ############################ error calculation ##########################
    if number_model==1:
        error_after_frob_path= os.path.join(name_frobenius, "error_after_frob_"+index_model+".txt")
        average_couplings=np.loadtxt(os.path.join(name_average_product, "average_product_"+index_model+".txt"))
    else:
        error_after_frob_path= os.path.join(name_frobenius, "error_after_frob_0-"+str(number_model-1)+".txt")
        average_couplings=np.loadtxt(os.path.join(name_average_product, "average_product_0-"+str(number_model-1)+".txt"))
    if os.path.isfile(error_after_frob_path):
        print("error_after_frob already exists we don't compute it again")
        average_error_after_frob=np.loadtxt(error_after_frob_path)
        #verify that average_couplings is symmetric
        assert np.allclose(average_couplings, average_couplings.T) #check if the matrix frobenius is symmetric
        error_Wab_final=ErrorWij_average_product(average_error_after_frob,average_couplings,L,length_prot1)
        if number_model==1:
            np.savetxt(os.path.join(output_directory, "error_Wab_final_"+index_model+".txt"), error_Wab_final)
            print("error_Wab_final saved in the file: ", os.path.join(output_directory, "error_Wab_final_"+index_model+".txt"))
        else:
            np.savetxt(os.path.join(output_directory, "error_Wab_final_0-"+str(number_model-1)+".txt"), error_Wab_final)
            print("error_Wab_final saved in the file: ", os.path.join(output_directory, "error_Wab_final_0-"+str(number_model-1)+".txt"))
    else:
        print("---------------- Frobenius error calculation ----------------")
        if number_model>1 and type_average=="average_couplings_frob":
            n=0
            for step in range(number_model):
                name_couplings_before=os.path.join(name_couplings_before, "error_Wij_"+str(step)+"_before_ising.txt")
                name_err_after=os.path.join(name_couplings_after, "error_Wij_"+str(step)+"_after_ising.txt")
                name_frob=os.path.join(name_frobenius, "error_after_frob_"+str(step)+".txt")
                couplings_after_ising=np.loadtxt(os.path.join(name_couplings_after, "couplings_after_ising_"+str(step)+".txt"))
                couplings_after_frob=np.loadtxt(os.path.join(name_frobenius, "couplings_frobenius_"+str(step)+".txt"))
                error_Wij_after_ising=np.loadtxt(name_err_after)
                error_Wab_step=ErrorWij_frobenius(error_Wij_after_ising,couplings_after_ising,couplings_after_frob,L,K,length_prot1)
                if n==0:
                    error_Wab=np.copy(error_Wab_step)
                    n=1
                if n==1:
                    error_Wab += error_Wab_step
                np.savetxt(name_frob, error_Wab)
                print("error after frobenius saved in the file: ", name_frob)
            error_Wab=error_Wab/number_model
            np.savetxt(error_after_frob_path, error_Wab)
        
        else:
            couplings_after_ising=np.loadtxt(os.path.join(name_couplings_after, "couplings_after_ising_" + index_model + ".txt"))
            couplings_after_frob=np.loadtxt(os.path.join(name_frobenius, "couplings_frobenius_"+index_model+".txt"))
            error_Wij_after_ising=np.loadtxt(path_error_Wij_after)
            error_Wab=ErrorWij_frobenius(error_Wij_after_ising,couplings_after_ising,couplings_after_frob,L,K,length_prot1)
            np.savetxt(error_after_frob_path, error_Wab)
            print("error after frobenius saved in the file: ", os.path.join(name_frobenius, "error_after_frob_"+index_model+".txt"))
            del couplings_after_ising
            del couplings_after_frob
            del error_Wij_after_ising
            gc.collect()

        error_Wab_final=ErrorWij_average_product(error_Wab,average_couplings,L,length_prot1)
        if number_model==1:
            np.savetxt(os.path.join(output_directory, "error_Wab_final_"+index_model+".txt"), error_Wab_final)
            print("error_Wab_final saved in the file: ", os.path.join(output_directory, "error_Wab_final_"+index_model+".txt"))
        else:
            np.savetxt(os.path.join(output_directory, "error_Wab_final_0-"+str(number_model-1)+".txt"), error_Wab_final)
            print("error_Wab_final saved in the file: ", os.path.join(output_directory, "error_Wab_final_0-"+str(number_model-1)+".txt"))


    #########################################################################
    #########################################################################
    print("------------- END OF THE COUPLINGS CALCULATION :) -------")
    print(" See you soon!")
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################




    


            

        

