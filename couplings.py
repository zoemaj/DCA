import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import gc
#still in progress, need to change the ising 

def prediction(input, model, model_type) :
    #for a linear model, the couplings are directly the weights learned
    if model_type == "linear" :
        return np.array(model.masked_linear.linear.weight.detach().cpu())
    if model_type == "non-linear" : return model.non_linear(input)
    else : print("error with model type")   


def extract_couplings_W3(model, model_type, original_shape, indexes_batch) :
    
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

def extract_couplings(model, model_type, original_shape,data_per_col,path) :
    #if torch.cuda.is_available():
    #    use_cuda = True
    #else:
    #    use_cuda = False
    use_cuda=False #we don't use cuda for the moment
    device = torch.device("cuda" if use_cuda else "cpu")
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
                    W_ialpha_jbeta_kgamma=extract_couplings_W3(model, model_type, (L,K), indexe_batch) #DIMENSION: (1,L*K) 
                    W_3_batch[:,indexe0,indexe1]=W_ialpha_jbeta_kgamma #DIMENSION: (L*K,batch_size*K,batch_size*K)
                    #now we need to complete the other half of the matrix
                    W_3_batch[:,indexe1,indexe0]=W_ialpha_jbeta_kgamma #DIMENSION: (L*K,batch_size*K,batch_size*K)
                    #and the terms not present correspond when j=k which is not possible
                
                if Progression%5==0:
                    print("Ising gauge on W3...")  
                    print("shape W_3_batch before ising :",W_3_batch.shape)  
                #now we need to apply ising gauge on W_3_batch
                W_3_batch=ising_gauge_W3(W_3_batch, (L,K),indexes_batch,batch_size,data_per_col,device,Progression) # shape of W_3_batch: (L*K,batch_size*K,batch_size*K)
                
                #for indexe_batch in indexes_batchs: #we project every [i,[j,k]] to [i,j] for every k (and simitrically)
                #    W_3_2d[:,indexe_batch[0]]=W_3_2d[:,indexe_batch[0]]+W_3_batch[:,indexe_batch[0],indexe_batch[1]]
                #    W_3_2d[:,indexe_batch[1]]=W_3_2d[:,indexe_batch[1]]+W_3_batch[:,indexe_batch[1],indexe_batch[0]]
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
                    #if a%((L*K)/8)==0:
                    #    print("Progression:",a/(L*K)*100,"%")
                    K_a=np.sum(data_per_col[:,a//K]==0)
                    for b in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                        K_b=np.sum(data_per_col[:,b//K]==0)
                        b_batch=b-indexes_batch[0,0]
                        for y in range(indexes_batch[0,0],indexes_batch[0,0]+batch_size*K):
                            K_y=np.sum(data_per_col[:,y//K]==0)
                            y_batch=y-indexes_batch[0,0]
                            W2[a,b]=W2[a,b]+sum_a[a,b_batch,y_batch]/K_a+sum_b[a,b_batch,y_batch]/K_b+sum_y[a,b_batch,y_batch]/K_y
                #convert W2 to a numpy array
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
    return W2

        
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
    

def ising_gauge(couplings, original_shape, data_per_col) :
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
    
    new_couplings = np.copy(couplings)
    # Initialize arrays outside the loops
    sum_row = np.zeros((L * K, L * K))
    sum_col = np.zeros((L * K, L * K))
    sum_rowcol = np.zeros((L * K, L * K))
    print("--------- sum over b of C_ia,jb and sum over a of C_ia,jb -----------")
    for i in range(0, L * K):  
        col_i=i//K #exemple: i=0, K=2, k=0 | i=1, K=2, k=0 | i=2, K=2, k=1 | i=3, K=2, k=1
        row_i=col_i
        k=i%K #% take the modulo of i by K. exemple: i=0, K=2, k=0 | i=1, K=2, k=1 | i=2, K=2, k=0 | i=3, K=2, k=1
        mask = (data_per_col[k, col_i] != 1) 
        sum_col[:, i] = np.where(mask, np.sum(couplings[:, col_i:col_i + K], axis=1), 0)
        sum_row[i, :] = np.where(mask, np.sum(couplings[row_i:row_i + K, :], axis=0), 0)
    print("--------- sum over a,b of C_ia,jb -----------")
    for j in range(0, L * K):
        for i in range(0, L * K):
            col_i=i//K
            k_i=i%K
            row_j=j//K
            k_j=j%K
            mask_row = (data_per_col[k_i, col_i] != 1)
            mask_col = (data_per_col[k_j, row_j] != 1)
            sum_rowcol[j, col_i:col_i+K] = np.sum(sum_row[j, col_i:col_i+K])
            sum_rowcol[j, col_i:col_i+K] = np.where(mask_row & mask_col, sum_rowcol[j, col_i:col_i+K], 0)
    print("--------- computing the new_couplings -----------")
    for j in range(L * K):
        for i in range(L * K):
            col_i=i//K
            k_i=i%K
            row_j=j//K
            k_j=j%K
            K_beta = np.sum(data_per_col[:, col_i] == 0)
            K_alpha = np.sum(data_per_col[:, row_i] == 0)
            new_couplings[j, i] = (
                new_couplings[j, i] - sum_row[j, i] / K_alpha
                - sum_col[j, i] / K_beta
                + sum_rowcol[j, i] / (K_alpha * K_beta)
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
    

def couplings(model_name, number_model=1, type_average='average_couplings', output_name='/', figure=False, data_per_col='/', model_type="linear",L=0,K=0) :

    ###################################################################################
    ################# EXTRACTION OF THE MODEL(S) ######################################
    ############ & data_per_col (the same for every model(s)) #########################
    ###################################################################################
    print("--------------------------------------------------------------------------")
    print("Welcome in the couplings step, we are extracting the parameters...")

    #models=[]
    number_model=int(number_model)
    #if number_model>1:
     #   print("------------------ several models ------------------")
     #   print("------------- extraction of the models --------------")
      #  for m in range(number_model):
      #      print("model_name:", model_name+'_' + str(m))
      #      models.append(torch.load(model_name+'_' + str(m)))
    #else:
    print("-------------------- one model ---------------------")
    #    print("------------- extraction of the model --------------")
    #    models.append(torch.load(model_name))
    
    #extract the folder name (first part)
    path_folder=os.path.dirname(model_name)
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
    ###################################################################################
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
            threshold_preprocessed=input("Please enter the threshold preprocessed used")
            path_info=input("Please enter the path where to find the file with the information about N,L,K")

        
        
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

    ###################################################################################
    

    print("L,K:(",L,",",K,")")
    if K>21:
        print("K=",K,"-> we have the taxonomy for each sequence. In this case the L is the length of a sequence +1.")
        
    
    if output_name=="/":
        #take the path of the model_name and add the name 'couplings' to it
        output_name = os.path.dirname(model_name)
        output_name = os.path.join(output_name, 'couplings')
    ###################################################################################
    ################# WEIGHT EXTRACTION AND ISING GAUGE FOR THE MODEL(S) ##############
    ################### (depend if we consider the taxonomy or not)####################
    ######################## (depend if it is linear or not)###########################
    ###################################################################################
    
    #createthe name
    couplings_path = os.path.dirname(output_name) # Path for the couplings directory without the last 
    #add the name 'couplings_before_ising' to the couplings_path
    name_couplings_before = os.path.join(couplings_path, 'couplings_before_ising/')
    ALL_couplings=[]
    #check if the couplings after ising already exist
    name_to_check= os.path.join(couplings_path, "couplings_after_ising/couplings_after_ising_0-"+str(number_model-1)+".txt")
    if os.path.isfile(name_to_check):   
        print("couplings_after_ising already exists, we don't compute it again")
        average_couplings=np.loadtxt(name_to_check)
        couplings_path=os.path.join(couplings_path, "couplings_after_ising")
    else:
        #check if it exist
        find_average_couplings=False
        path_couplings_before=os.path.join(name_couplings_before, "couplings_0"+str(number_model-1)+"_before_ising.txt")

        if number_model==1 or type_average=="average_couplings":
            if os.path.isfile(path_couplings_before):
                print("couplings_before_ising.txt already exists, we don't compute it again")
                average_couplings=np.loadtxt(path_couplings_before)
                find_average_couplings=True
                #print("shape of couplings before ising:", average_couplings.shape) #oke
                if K>21:
                    L=L-1 #lose a dimension with class
                    data_per_col=data_per_col[:,:-1] #remove the last column corresponding to the class type
                print('L,K:(',L,',',K,')')
                
        if find_average_couplings==False: #can be type_average=average_couplings_frob (we need all couplings) or we don't have the couplings_before_ising.txt
            # Create the directory and its parent directories if they don't exist
            os.makedirs(name_couplings_before, exist_ok=True)
            # Check which couplings we need to compute
            couplings_ising_to_do=[]
            for step in range(number_model):
                name_couplings_before_step=os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt")
                if os.path.isfile(name_couplings_before_step):
                    print(f"couplings_{step}_before_ising.txt already exists, we don't compute it again")
                else:
                    print(f"couplings_{step}_before_ising.txt does not exist, we will compute it")
                    os.makedirs(couplings_path, exist_ok=True)
                    couplings_ising_to_do.append(step)
            print("--------------------------------------------------------------------------")
            if K>21:
                print("K=",K,"-> we have the taxonomy for each sequence")
                average_couplings=np.zeros(((L-1)*K,(L-1)*K))
            else:
                print("K=",K,"-> we don't have the taxonomy for each sequence")
                average_couplings=np.zeros((L*K,L*K))
            #for step,model in enumerate(models):
            for step in range(number_model):
                if step in couplings_ising_to_do:
                    print("extraction of the couplings for the model: ", step+1, "/", number_model)
                    model=torch.load(model_name+'_' + str(step))
                    couplings=extract_couplings(model, model_type, (L,K),data_per_col,output_name)
                    if K>21:
                        #with tax we need to remove the last blocs of size K
                        print("treatment of couplings: remove the last column corresponding to the taxonomy type")
                        couplings=couplings[:-K,:-K]
                    #save the couplings in the file couplings_before_ising.txt
                    np.savetxt(os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt"), couplings)
                    print("couplings before ising gauge saved in the file: ", os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt"))
                    #ALL_couplings.append(couplings)
                    average_couplings += couplings
                    del couplings
                    del model
                    gc.collect()
                else:
                    name_coup=os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt")
                    #ALL_couplings.append(np.loadtxt(name_coup))
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
                # Path for the couplings directory without the last part of the output_name (to stock in the folder)
                couplings_path = os.path.dirname(output_name)
                #add the name 'couplings_models' to the couplings_path
                couplings_path = os.path.join(couplings_path, 'couplings_after_ising/')
                # Create the directory and its parent directories if they don't exist
                os.makedirs(couplings_path, exist_ok=True)
                
                average_couplings=np.zeros((L*K,L*K))
                #ALL_couplings_ising=[]
                print("Treatment of the ising gauge on each couplings...")
                #for couplings,model in zip(ALL_couplings,models):
                for step in range(number_model):
                    print("gauge process on model : ", step+1, "/", number_model)
                    #look if the file with path couplings_path and name couplings_ising_step.txt exists
                    #if it exists, load it and don't do ising_gauge again
                    #if it doesn't exist, do ising_gauge and save it
                    if os.path.isfile(os.path.join(couplings_path, "couplings_after_ising_" + str(step) + ".txt")):
                        print("couplings_after_ising_" + str(step) + ".txt already exists, we don't compute it again")
                        couplings = np.loadtxt(os.path.join(couplings_path, "couplings_after_ising_" + str(step) + ".txt"))
                    else:
                        couplings=np.loadtxt(os.path.join(name_couplings_before, "couplings_"+str(step)+"_before_ising.txt"))
                        couplings = ising_gauge(couplings, (L,K), data_per_col)
                        np.savetxt(os.path.join(couplings_path, "couplings_after_ising_" + str(step) + ".txt"), couplings)
                        print("couplings after ising gauge saved in the file: ", os.path.join(couplings_path, "couplings_after_ising_" + str(step) + ".txt"))
                    #ALL_couplings_ising.append(couplings)
                    average_couplings += couplings
                    del couplings
                    gc.collect()
                    #step+=1
                average_couplings=average_couplings/number_model
                np.savetxt(os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt"), average_couplings)
                print("average couplings after ising gauge saved in the file: ", os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt"))
                
            else:
                print("error with type_average")
        else: # we do ising on the average of the models (number_model=1)
            print("You have chosen to treat only one model")
            couplings_path = os.path.dirname(output_name)
            #add the name 'couplings_after_ising' to the couplings_path
            couplings_path = os.path.join(couplings_path, 'couplings_after_ising/')
            #check if it exist
            if os.path.isfile(os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt")):
                print("couplings_after_ising_0-"+str(number_model-1)+".txt already exists, we don't compute it again")
                average_couplings=np.loadtxt(os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt"))
            else:
                print("Treatment of the ising gauge on the couplings...")
                average_couplings=np.loadtxt(path_couplings_before)
                average_couplings=np.copy(average_couplings)
                #print("shape data_per_col:", data_per_col.shape)
                average_couplings=ising_gauge(average_couplings,(L,K), data_per_col)
                # Create the directory and its parent directories if they don't exist
                os.makedirs(couplings_path, exist_ok=True)
                np.savetxt(os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt"), average_couplings)
                print("couplings after ising gauge saved in the file: ", os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt"))
                
            

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
        plt.savefig(os.path.join(couplings_path, "couplings_after_ising.png"))
    else:
        print("No plot after the ising gauge, because figure=False")
    del average_couplings
    gc.collect()
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
    print("--------------------------------------------------------------------------")
    print("------------------------ average product correction ----------------------")
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
    if os.path.isfile(os.path.join(output_directory, output_name,"_0-"+str(number_model-1)+".txt")):
        print("The final couplings file already exists, we don't compute it again")
        print("You can find it at the path: ", os.path.join(output_directory, output_name))
    else:
        n=0
        print("---------------- Frobenius ----------------")
        if number_model>1 and type_average=="average_couplings_frob":
            print("Treatment of the average product correction on the couplings...")
            for step in range(number_model):
                couplings_old=np.loadtxt(os.path.join(couplings_path, "couplings_after_ising_"+str(step)+".txt"))
                couplings=0.5*(couplings_old + couplings_old.T)
                
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
                del couplings
                del couplings_old
                gc.collect()

            average_couplings = average_couplings/number_model
        else: #average couplings or for number_model=1
            print("Treatment of the average product correction on the average couplings...")
            average_couplings=np.loadtxt(os.path.join(couplings_path, "couplings_after_ising_0-"+str(number_model-1)+".txt"))
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
        output_name=output_name+"_0-"+str(number_model-1)+".txt"
        np.savetxt(os.path.join(output_directory,  output_name), average_couplings)
        print("The final couplings file is saved in the file: ", os.path.join(output_directory, output_name))
        print("---------------------------------- END -----------------------------------")
        print("--------------------------------------------------------------------------")
        #################################################################################
        #################################################################################
                



    


            

        

