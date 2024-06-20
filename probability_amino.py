import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FixedLocator
from matplotlib.colors import LinearSegmentedColormap

#the main function is at the end of the file and is called execute

#the other functions are used to load the data, create the datasets and plot the results

###############################################################################################
def create_datasets(data, labels, separations,nb_models=1) :
    """
    This function is used to create the training, validation and test datasets with the same repartition used during the model building and the training.
    input:
                data        ->      input data
                labels      ->      corresponding to the input data
                separations ->      list of 2 floats, the first one is the fraction of the data that will be used for training,
                                    the second one is the fraction of the data that will be used for validation
                nb_models   ->      number of models that have been trained
    output:
                sequences       ->      test dataset
                sequences_train ->      training dataset
                labels          ->      labels of the test dataset
    """
    #compute the indices of the 3 datasets
    indices = np.array(range(len(data)))
    np.random.shuffle(indices)
    train_indices, _, test_indices = np.split(indices, [int(separations[0]*len(data)), int(separations[1]*len(data))])
    if nb_models>1:
        test_indices_final=np.copy(test_indices)
        for i in range(1,nb_models):
            np.random.shuffle(indices)
            _, _, test_indices_temp = np.split(indices, [int(separations[0]*len(data)), int(separations[1]*len(data))])
            #keep only the same indices between test_indices and test_indices_temp
            test_indices_final=np.intersect1d(test_indices,test_indices_temp)
            train_indices_final=np.setdiff1d(indices, test_indices_final)
        print("The number of test sequences is:",len(test_indices))
        if len(test_indices)==0:
            print("Since it's 0 we will only take the test of the first model.")
        else:
            test_indices=test_indices_final
            train_indices=train_indices_final
    else:   
        print("The number of test sequences is:",len(test_indices))

    #create training, validation and test dataset
    sequences=torch.from_numpy(data[test_indices])
    sequences_train=torch.from_numpy(data[train_indices])
    labels=torch.from_numpy(labels[test_indices])
    return sequences,sequences_train,labels
###############################################################################################
###############################################################################################
def get_data_labels(MSA_file,K) :
  ''' This function is used to load the data and the labels from the MSA file.
    input:
            MSA_file -> path to the MSA file
            K        -> number of amino acids
    output:
            data     -> input data
            labels   -> corresponding to the input data
    '''
  
  #load data from the MSA file
  data = np.genfromtxt(MSA_file, delimiter=',').astype(int)
  #put the data in one hot encoding form of size K
  new_data = np.array(nn.functional.one_hot(torch.Tensor(data).to(torch.int64),num_classes=K))
  (N,L,K) = new_data.shape
  print("Wah there are",N,"sequences with",L,"amino acids and",K,"different values for each amino acid!")
  print("This looks interesting! Let's see what we can do with this ;)")
  labels=data #the labels are the same as the input
  new_data = np.reshape(new_data, (N, L * K)) #now we have N sequences with LK values
  print("Data and labels have been successfully obtained")
  return new_data, labels
###############################################################################################
###############################################################################################
def plot_final( scores_normalized, scores_true, aaRemapInt, list_info_amino, percentage_good_prediction_per_ai, N, K,start_id,end_id,percentage_tax_tot):
    """
    This function is used to plot the scores and the true scores for one sequence.
    input:
                scores_normalized                   ->      the K scores for each amino acid 
                                                            type numpy array, shape (K,L)
                scores_true                         ->      the true scores
                                                            type numpy array, shape (K,L)
                aaRemapInt                          ->      dictionary for the amino acid
                                                            type dictionary
                list_info_amino                     ->      list contraining four list of amino acids (positive, negative, polar, apolar)
                                                            type list, length 4
                percentage_good_prediction_per_ai   ->      percentage of good prediction for each amino acid position
                                                            type numpy array, shape (L,)
                N                                   ->      number of sequences
                                                            type int
                K                                   ->      number of different values for each amino acid
                                                            type int
                start_id                            ->      the start of the plot (index of the amino acid)
                                                            type int
                end_id                              ->      the end of the plot (index of the amino acid)
                                                            type int
                percentage_tax_tot                  ->      dictionary containing the percentage of each taxonomy
                                                            type dictionary
    output:
                None
    """
    extent=[0, end_id-start_id, 0, K] #the extent of the plot
    plt.figure(figsize=(16, 5))
    colors=['white','yellow','aquamarine','lime','green','magenta','maroon','peru','black']
    cmap_name = 'custom_cbar'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    plt.imshow(scores_normalized, cmap=cm, extent=extent) #show the scores
    ################ Plot the true scores in green ########################################
    for i in range(scores_true.shape[0]): #the k values
        for j in range(scores_true.shape[1]): #the i amino acids
            if scores_true[i, j] ==1:  # Adjust threshold as needed
                plt.plot(j + 0.5, K - i - 0.5, marker='x', color='green', markersize=6) 
    #######################################################################################
    #add the colorbar
    plt.clim(0, 1)
    cbar = plt.colorbar()
    cbar.set_label('Probability %')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '20', '40', '60', '80', '100'])
    #add the x ticks
    plt.xticks(np.arange(0.5, end_id-start_id+0.5, step=1))  
    ################ Manually adjusting x-axis tick labels ################################
    ############## (as the correction position of amino acids) ############################
    tick_labels = np.arange(start_id, end_id, step=1)
    tick_labels = [rf"$a_{{{x+1}}}$" for x in tick_labels]
    plt.gca().set_xticklabels(tick_labels, rotation=90)
    #######################################################################################
    plt.gca().yaxis.set_minor_locator(FixedLocator(np.arange(0, K, step=1))) #add minor ticks for the y-axis
    plt.gca().xaxis.set_minor_locator(FixedLocator(np.arange(0, end_id-start_id, step=1))) #add minor ticks for the x-axis
    plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5) #add grid for minor ticks
    ################ Manually adjusting y-axis tick labels ################################
    ################### (as the correction character for ai,k) ############################
    plt.yticks(np.arange( 0.5, K+0.5,step=1)) #set the y-ticks to be in the middle of the boxes
    tick_labels = np.arange(0, K, step=1)
    tick_labels = [list(aaRemapInt.keys())[list(aaRemapInt.values()).index(x)] for x in tick_labels]
    if K>21: #to add the percentage only for the taxonomies, remove the if and start the for loop to 0 if you want to add the percentage for all the amino acids
        for i in range(21,K,1):
            id_i=list(aaRemapInt.values())[list(aaRemapInt.keys()).index(tick_labels[i])]
            dictionary_for_column_i=percentage_tax_tot
            tick_labels[i]=f"{tick_labels[i]} ({int(dictionary_for_column_i[id_i]/N*100)}%)"
    plt.gca().set_yticklabels(tick_labels[::-1]) #reverse the list of tick labels -> the first one is at the bottom
    #######################################################################################
    ################## Add color to the y ticks labels ####################################
    #######################################################################################
    #add color to the y ticks labels according to the type of amino acid
    #green for positive, red for negative, blue for polar and orange for apolar
    #if the amino acid is not in the list_info_amino, the color is black
    for i in range(len(tick_labels[::-1])):
        color='black'
        for j in range(len(list_info_amino)):
            if tick_labels[::-1][i] in list_info_amino[j]:
                if j==0:
                    color='red'
                elif j==1:
                    color='green'
                elif j==2:
                    color='blue'
                elif j==3:
                    color='orange'
        plt.gca().get_yticklabels()[i].set_color(color)
    #######################################################################################
    plt.title('Comparison of Scores and True Scores for one sequence', fontsize=12)
    plt.xlabel('amino acid', fontsize=14)
    plt.ylabel('value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ############### Add the percentage of good prediction for each amino acid ############
    ####################### (just under the x-axis) ######################################
    str_line="-"*250
    plt.text(len(percentage_good_prediction_per_ai)//2, -8, f"{str_line}", ha='center', va='center', color='black', rotation=0)
    plt.text(len(percentage_good_prediction_per_ai)//2, -9, f"percentage of good prediction for a total of {N} sequences test", ha='center', va='center', color='black', rotation=0)
    for i in range(len(percentage_good_prediction_per_ai)):
        if percentage_good_prediction_per_ai[i]*100>70:
            c='black'
        else:
            c='red'
        plt.gca().get_xticklabels()[i].set_color(c)
        plt.text(i+0.5, -11, f'{int(percentage_good_prediction_per_ai[i]*100)}%', ha='center', va='center', color=c, rotation=90)
    #######################################################################################
    plt.tight_layout()
    plt.show()

###########################################################################################################################################
#********************************************** MAIN FUNCTION *****************************************************************************
########################################################################################################################################### 
def execute(model_path,MSA_file,seed,K,nb_models=1):
    print("--------- Welcome to the probability amino script :) ---------")
    ###################### initialize the seed and the device ######################
    torch.manual_seed(seed) #necessary to have exactly the same distribution of the data than the model used
    np.random.seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("The device is set to cuda.")
    else:
        device = torch.device('cpu')
        print("The device is set to cpu.")
    ####################### load the data and the model ############################
    print("load of the data and label ...")
    data, labels= get_data_labels(MSA_file,K) 
    sequences,sequences_train,labels=create_datasets(data, labels, [0.7, 0.7],nb_models)
    print("load of the model ...")
    model=torch.load(model_path,map_location=device) #load the model
    path_folder=model_path.split("/")[:2] #get the two first elements of the path
    path_folder="/".join(path_folder)
    
    ###############################################################################
    ######################## get the predictions for ##############################
    ######################## each amino acid position #############################
    print("-------------------- Start the predictions --------------------")
    print("get the predictions for each amino acid position ...")
    L=labels.shape[1] #number of amino acids
    ALL_scores=np.zeros((len(sequences),K,L)) #initialize the scores
    ALL_scores_true=np.zeros((len(sequences),K,L))
    percentage_good_prediction_per_ai=np.zeros(L) #initialize the percentage of good prediction for each amino acid position
    percentage_good_prediction_per_ai_2=np.zeros(L)
    percentage_good_prediction_per_ai_3=np.zeros(L)
    predicted_aminos=model(sequences) #shape (nb_sequences,number amino acids*K)
    predicted_aminos=predicted_aminos.detach().numpy() #convert to numpy array
    predicted_aminos=np.reshape(predicted_aminos,(len(sequences),L,K)) #shape (nb_sequences,number amino acids,K)
    #convert the labels into one hot encoding for every sequences
    labels_one_hot=nn.functional.one_hot(torch.Tensor(labels).to(torch.int64),num_classes=K) #shape (nb_sequences,L,K)
    ################################################################################

    ################################################################################
    ################## find the percentage of good prediction ######################
    ################################################################################
    print("find the percentage of good prediction ...")
    for amino_i in tqdm(range(0,L)):#for each amino acid position 
        predicted_amino=predicted_aminos[:,amino_i,:] #get the K values for all the sequences
        predicted_amino=np.reshape(predicted_amino,(len(sequences),K)) #shape (nb_sequences,K)
        ALL_scores[:,:,amino_i]=predicted_amino #attribute this score to the corresponding amino acid position
        true_labels=labels_one_hot[:,amino_i,:] #get the true labels for all the sequences
        true_labels=np.reshape(true_labels,(len(sequences),K)) #shape (nb_sequences,K)
        ALL_scores_true[:,:,amino_i]= true_labels #attribute this score to the corresponding amino acid position
        index_max_1=np.argmax(predicted_amino,axis=1)  #find the index of the maximum value for each sequence
        #find when index_max_1==labels[amino_i] for each sequence
        find_match_1=[True if index_max_1[seq]==labels[seq][amino_i] else False for seq in range(len(sequences))]
        percentage_good_prediction_per_ai[amino_i]=sum(find_match_1)/len(sequences)
        predicted_amino_2=np.copy(predicted_amino)
        predicted_amino_2[np.arange(len(sequences)),index_max_1]=0
        index_max_2=np.argmax(predicted_amino_2,axis=1)
        find_match_2=[True if index_max_1[seq]==labels[seq][amino_i] or index_max_2[seq]==labels[seq][amino_i] else False for seq in range(len(sequences))]
        percentage_good_prediction_per_ai_2[amino_i]=sum(find_match_2)/len(sequences)
        predicted_amino_3=np.copy(predicted_amino_2)
        predicted_amino_3[np.arange(len(sequences)),index_max_2]=0
        index_max_3=np.argmax(predicted_amino_3,axis=1)
        find_match_3=[True if index_max_1[seq]==labels[seq][amino_i] or index_max_2[seq]==labels[seq][amino_i] or index_max_3[seq]==labels[seq][amino_i] else False for seq in range(len(sequences))]
        percentage_good_prediction_per_ai_3[amino_i]=sum(find_match_3)/len(sequences)
    #################################################################################

    
    ############################# get the dictionary for the amino acid ################################
    aaRemapInt={'-/X/Z/B/O/U':0,'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8,
                'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}
    amino_negatif=['D','E']
    amino_positif=['K','R','H']
    amino_polar=['Q','N','S','T','Y']
    amino_apolaire=['G','A','V','C','P','L','I','M','F','W']
    list_info_amino=[amino_negatif,amino_positif,amino_polar,amino_apolaire]

    path_folder=model_path.split("weights")[0]
    while path_folder[-1]=='/' or path_folder[-1]=='\\': #remove the last character if it's a / or a \ (linux or windows)
        path_folder=path_folder[:-1]
    if K>21:
        print("load the taxonomies distribution ...")
        path_file_with_tax=path_folder+"/distribution-tax.txt"
        # add theses values to the dictionary aaRemapInt
        with open(path_file_with_tax) as f:
            lines=f.readlines()
            for line in lines:
                tax=line.split(":")[0].strip() #strip() remove the spaces at the beginning and the end of the string
                value=int(line.split(":")[1].strip())
                aaRemapInt[tax]=value
    ####################################################################################################

    #convert sequences in numpy array of shape (nb_sequences,L,K)
    sequences_train=sequences_train.detach().numpy()
    sequences_train=np.reshape(sequences_train,(len(sequences_train),L,K))
    sequences_test=sequences.detach().numpy()
    sequences_test=np.reshape(sequences_test,(len(sequences_test),L,K))
    #################################################################################
    ################### PLOTS TAXONOMIES DISTRIBUTION ###############################
    #################################################################################
    print("-------------------- Start the plots --------------------")
    percentage_tax_tot={}
    percentage_tax_test={}
    if K>21: #keep the percentage of taxonomies for the test dataset and the train dataset
        for k in range(21,K,1):
            count_k=sum(np.argmax(sequences_train,axis=2)==k) #count how many sequences have k (only one k>21 for each sequences)
            count_k_test=sum(np.argmax(sequences_test,axis=2)==k) #count how many sequences have k (only one k>21 for each sequences)
            percentage_tax_tot[k]=count_k[-1]
            percentage_tax_test[k]=count_k_test[-1]
    n = len(percentage_tax_tot)

    r1 = np.arange(n)
    r2 = [x + 0.25 for x in r1]
    plt.figure(figsize=(5, 6))
    plt.bar(r1, percentage_tax_tot.values(), color='aqua', width=0.5, align='center',label='train 70%')
    plt.bar(r2, percentage_tax_test.values(), color='violet', width=0.5, align='center',label='test 30%')
    plt.title('Percentage of each taxonomies', fontsize=12)
    plt.xlabel('taxonomy', fontsize=15)
    list_tax_names=[list(aaRemapInt.keys())[list(aaRemapInt.values()).index(k)] for k in percentage_tax_tot.keys()]
    plt.xticks([r + 0.125 for r in range(n)], list_tax_names, fontsize=13)
    # Add percentage labels above the bars
    for i, key in enumerate(percentage_tax_tot.keys()):
        plt.text(r2[i], -0.1, f"{int(percentage_tax_tot[key]/len(sequences_train) * 100)}%", ha='center', va='bottom', color='black')
    plt.ylabel('percentage', fontsize=15)
    plt.legend(fontsize=12,loc='upper right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid()
    plt.show()
    #################################################################################
    

    #################################################################################
    ################# Pepare the parameters for one #################################
    ########## arbitrary sequence to show in the colormap ###########################
    el_id=453 #random choice, you can change it
    scores=ALL_scores[el_id,:,:] #take the first element of the list 
    scores=np.reshape(scores,(K,L))
    #we will not do a plot for all the sequences but only for the first one
    scores_true=ALL_scores_true[el_id,:,:]
    scores_true=np.reshape(scores_true,(K,L))
    scores_normalized = scores #SINCE THE MODEL apply a softmax, the scores are already between 0 and 1
    ###################################################################################

    ###############################################################################################
    ################ Plot the first, second and third good prediction percentage ##################
    ###############################################################################################
    plt.figure(figsize=(5, 6))
    plt.plot(np.arange(0, len(percentage_good_prediction_per_ai)), percentage_good_prediction_per_ai*100,color='blue',label='1st prediction correct')
    plt.plot(np.arange(0, len(percentage_good_prediction_per_ai_2)), percentage_good_prediction_per_ai_2*100,color='purple',label='1st or 2nd prediction correct')
    plt.plot(np.arange(0, len(percentage_good_prediction_per_ai_3)), percentage_good_prediction_per_ai_3*100,color='magenta',label='1st or 2nd or 3rd prediction correct')
    plt.title(str(len(sequences))+' sequences test', fontsize=12)
    plt.xlabel('amino acid', fontsize=14)
    xticks_labels = np.arange(1, len(percentage_good_prediction_per_ai)+1, step=100)
    xticks_labels = [rf"$a_{{{x}}}$" for x in xticks_labels]
    plt.xticks(np.arange(0, len(percentage_good_prediction_per_ai), step=100), xticks_labels)
    plt.ylabel('error %', fontsize=15)
    plt.legend(fontsize=12,loc='lower center')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid()
    plt.tight_layout()
    plt.show()
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    ###############################################################################################
    ############################## Plot the scores and the true scores ############################
    ###############################################################################################
    N=len(sequences)
    L=labels.shape[1]
    several_plot=input("Do you want to separate the plot in several part of the sequences? (yes/no)")
    while several_plot not in ["yes","no"]:
        several_plot=input("Please enter yes or no.")

    if several_plot=="yes":
        how_many_plot=int(input("In how many part?"))
        while how_many_plot<=0:
            how_many_plot=int(input("Please enter a positive number."))
    
        for i in range(0,how_many_plot):
            if i==0:
                start=0
                end = input("Enter the end of the first part.")
            elif i==how_many_plot-1:
                start=end
                end=L
            else:
                start=end
                end = input("Enter the end of the next part.")
            start=int(start)
            end=int(end)
            while end<=start or end>L:
                end = input("Please enter a number greater than the start and less than the number of sequences.")
                end=int(end)
            ########## plot the scores and the true scores for the part of the sequences ###########
            scores_normalized_part=scores_normalized[:,start:end] #take only the part of the scores that we want to plot
            scores_true_part=scores_true[:,start:end]
            percentage_good_prediction_per_ai_part=percentage_good_prediction_per_ai[start:end]
            plot_final(scores_normalized_part, scores_true_part, aaRemapInt, list_info_amino, percentage_good_prediction_per_ai_part, N, K,start,end,percentage_tax_test)
            ########################################################################################
    else:
        plot_final(scores_normalized, scores_true, aaRemapInt, list_info_amino, percentage_good_prediction_per_ai, N, K,0,L,percentage_tax_test)
    ################################################################################################
    ################################################################################################
    print("--------- End of the probability amino script ---------")
    print("--------- See you soon! ---------")

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

    




