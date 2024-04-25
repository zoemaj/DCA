import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import psutil
import subprocess
from tqdm import tqdm
from itertools import chain
import gc

# Function to print CPU usage percentage -> not used for now but can be useful in case of debugging
def print_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)  # Get CPU usage percentage for the last 1 second
    print("CPU Usage:", cpu_percent, "%")

class SquareActivation(nn.Module) :
  'custom square activation'
  def __init__(self) :
    super().__init__()
  def forward(self, input) :
    return input.square()

####################################################################################################################################
########################## CREATION OF THE MASKED   ################################################################################
####################################################################################################################################
def generate_indices_mask(L,K,data_per_col,path,length_prot1=0):
    ''' 
    This function is used to generate the indices that will be used to mask the model.
    The indices are written in a txt file that will be saved in the folder.
      (like this we do not need to stock the list of indices and we don't need to compute them again at each time)
    for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
    we mask also the links between the amino acids that are not possible at a position (identities)
    input:
        L: length of the sequences
        K: number of different amino acids
        data_per_col: for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
        path: path where the file will be saved
    '''
    if length_prot1!=0:
      length_prot2=L-length_prot1
      if K>21:
         length_prot2=length_prot2-1
      print("-------------- PAIR OF PROTEINS ----------------")
      print("You are using a pair of proteins, the first protein A has a length of ",length_prot1," and the second protein B has a length of ",length_prot2)
      print("We will masks the links between the amino acids of the same protein")
      print("A will be predicted with B and vice versa...")

    #check if the file already exists
    if os.path.exists(path+"/indice_mask.txt"):
      #in this case, we will not compute it again
      print("The file already exists, we will not compute it again")
      print("You can find it at: ",path+"/indice_mask.txt")
      return
    else: 
      print("Writing the indices_mask...")
      with open(path+"/indice_mask.txt", "w") as file:
        if length_prot1==0:
          for j in range(0, L * K, K):
              for a in range(j, j + K):
                  for b in range(j, j + K):
                      file.write(str(a) + " " + str(b) + "\n")
        else:
          
          for a in range(0, length_prot1 * K):
            for b in range(0, length_prot1 * K):
              file.write(str(a) + " " + str(b) + "\n")
          for a in range(length_prot1 * K,L*K):
            for b in range(length_prot1 * K,L*K):
              file.write(str(a) + " " + str(b) + "\n")
        for col_i in range(L):
          for amino in range(K):
            if data_per_col[amino, col_i] == 1:
              index_amino = col_i * K + amino
              for b in range(col_i * K):
                  file.write(str(index_amino) + " " + str(b) + "\n")
                  file.write(str(b) + " " + str(index_amino) + "\n")
              for b in range((col_i + 1) * K, L * K):
                  file.write(str(index_amino) + " " + str(b) + "\n")
                  file.write(str(b) + " " + str(index_amino) + "\n")
      print("The file has been successfully created : ",path+"/indice_mask.txt")
    return

def generate_indices_mask_non_linear(L,K,nb_hidden_neurons,data_per_col,path):
    ''' 
    This function is used to generate the indices that will be used to mask the model.
    The indices are written in a txt file that will be saved in the folder.
      (like this we do not need to stock the list of indices and we don't need to compute them again at each time)
    for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
    we mask also the links between the amino acids that are not possible at a position (identities)
    input:
        L: length of the sequences
        K: number of different amino acids
        data_per_col: for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
        path: path where the file will be saved
    '''
    #check if the file already exists
    path1=path+"/indice_mask1.txt" #disconnection between input and hidden 
    path2=path+"/indice_mask2.txt" #disconnection between hidden and output
    if os.path.exists(path1) and os.path.exists(path2):
      #in this case, we will not compute it again
      print(f"The files {path1} and {path2} already exists, we will not compute it again")
      print(f"You can find it at: {path1} and {path2}")
    
    else: 
      print("Writing the indices_mask...")
      with open(path1, "w") as file1:
        with open(path2, "w") as file2:
          for j in range(0,L):
            for a in range(j*nb_hidden_neurons,(j+1)*nb_hidden_neurons): #for each hidden layer
              for b in range(j*K,(j+1)*K): #for each amino acid j
                file1.write(str(a) + " " + str(b) + "\n") #disconnection between the input and the hidden layer
              if (j+1)<L: #if we are not at the last hidden layer
                for b in chain(range(0,j*K),range((j+1)*K,L*K)): #for each amino acid except amino acid j
                  file2.write(str(b) + " " + str(a) + "\n") #disconnection between the the hidden layer j and all the other amino acids except amino acid j
              else: #if we are at the last hidden layer
                for b in range(0,j*K):
                  file2.write(str(b) + " " + str(a) + "\n") 
          for col_i in range(L): #need to mask the links between the amino acids that are not possible at a position
              for amino in range(K):
                  if data_per_col[amino, col_i] == 1:
                      index_amino = col_i * K + amino #index of the amino acid in the input layer
                      #deconnect this index_amino with all the hidden layer except the one in position col_i
                      if (col_i+1)<L:
                        for b in chain(range(0,col_i*nb_hidden_neurons),range((col_i+1)*nb_hidden_neurons,L*nb_hidden_neurons)):
                            file1.write( str(b)+ " "+ str(index_amino) + "\n")
                      else:
                        for b in range(0,col_i*nb_hidden_neurons):
                            file1.write( str(b)+ " " +str(index_amino) + "\n")
                      #now deconnect the output coresponding to this amino acid with the hidden layer
                      for b in range(col_i*nb_hidden_neurons +amino, (col_i+1)*nb_hidden_neurons):
                          file2.write( str(index_amino)+ " " + str(b) + "\n")
                      
                        
      print(f"The files {path1} and {path2} have been successfully created")
    return



class MaskedLinear(nn.Module):
  """
  This function is usefull to remove links between some amino acids a_m,k
  build a non fully connected layer by putting a mask on the weights that must be null
  """
  def __init__(self, in_dim, out_dim, indices_mask):
    """
    in_dim: number of input features
    out_dim: number of output features
    indices_mask: list of tuples of int
    """
    #print cpu usage
    super(MaskedLinear, self).__init__()
    self.linear = nn.Linear(in_dim, out_dim) #MaskedLinear is made of a linear layer
    #Force the weights indicated by indices_mask to be zero by use of a mask
    self.mask = torch.zeros([out_dim, in_dim]).bool()
    #for a, b in indices_mask : self.mask[(a, b)] = 1
    #now indices_mask is a txt file that contains the indices of the weights that must be null
    with open(indices_mask, "r") as file:
        for line in file:
            a, b = line.split()
            a, b = int(a), int(b)
            self.mask[a, b] = 1
    self.linear.weight.data[self.mask] = 0 # zero out bad weights
    #modify backward_hood to prevent changes to the masked weights
    def backward_hook(grad):
      # Clone due to not being allowed to modify in-place gradients
      out = grad.clone()
      out[self.mask] = 0
      return out
    self.linear.weight = nn.Parameter(self.linear.weight)
    self.linear.weight.register_hook(backward_hook)
  def forward(self, input):
    input = input.to(self.linear.weight.device).float()
    # Use torch.nn.functional.linear to apply the modified weights
    return nn.functional.linear(input, self.linear.weight, self.linear.bias)
####################################################################################################################################
####################################################################################################################################


class LinearNetwork(nn.Module):
  'linear model with softmax activation on the output layer applied on residue positions'
  def __init__(self, indices_mask, in_dim, original_shape):
    """
    indices_mask: list of input and output neurons that must be disconnected
    in_dim: dimension of input layer
    original_shape: original shape of the MSA (N,L,K)
    """
    #print cpu usage
    super(LinearNetwork, self).__init__()
    self.masked_linear = MaskedLinear(in_dim, in_dim, indices_mask) #The masked linear layer to remove some connections
    self.softmax = nn.Softmax(dim=2) #the softmax activation function
    (_,L,K) = original_shape #original shape of the MSA
    self.L = L #length of the sequences
    self.K = K #number of different amino acids
  def forward(self, x):
    x = self.masked_linear(x) #apply the masked linear layer to remove some connections
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K)) #reshape 
    x = self.softmax(x) #apply softmax
    x = torch.reshape(x, (len(x), self.L*self.K)) #reshape to have the same shape as the input
    return x

class NonLinearNetwork(nn.Module):
  def __init__(self,indices_mask1,indices_mask2,in_dim,hidden_dim,original_shape,activation="square"):
    super(NonLinearNetwork, self).__init__()
    if activation=="square":
      activation_function=SquareActivation()
    elif activation=="tanh":
      activation_function=nn.Tanh()
    else:
      print("invalid activation function, square taken instead")
      activation_function=SquareActivation()
    
    self.non_linear=nn.Sequential(MaskedLinear(in_dim,hidden_dim,indices_mask1),activation_function,MaskedLinear(hidden_dim,in_dim,indices_mask2))
    self.softmax=nn.Softmax(dim=2)

    (_,L,K)=original_shape
    self.L=L
    self.K=K
  def forward(self,x):
    x=self.non_linear(x)
    x=torch.reshape(x,(len(x),self.L,self.K))
    x=self.softmax(x)
    x=torch.reshape(x,(len(x),self.L*self.K))
    return x

####################################################################################################################################
###### The following classes have not been modified from Aude Maier code (2022), is not used in this version ######################
###### there are still there if one wants to adapt them and use them in the future ######################################################
class MixNetwork(nn.Module) :
  'network mixing linear model and model with hidden layer and square/tanh activation'
  def __init__(self, indices_mask1, indices_mask2, indices_mask_linear, in_dim, hidden_dim, original_shape, activation="square"):
    """
    :param indices_mask1: list of input and hidden neurons that must be disconnected for the non-linear model
    :type indices_mask1: list of tuples of int
    :param indices_mask2: list of hidden and output neurons that must be disconnected for the non-linear model
    :type indices_mask2: list of tuples of int
    :param indices_mask_linear: list of input and output neurons that must be disconnected for the linear model
    :type indices_mask_linear: list of tuples of int
    :param in_dim: dimension of input layer
    :type in_dim: int
    :param hidden_dim: dimension of hidden layer
    :type hidden_dim: int
    :param original_shape: original shape of the MSA (N,L,K)
    :type original_shape: tuple of int
    :param activation: activation for the hidden layer, must be "square" or "tanh" otherwise square is taken by default
    :type activation: string
    """
    super(MixNetwork, self).__init__()
    #define activation function
    if activation == "square" : activation_function = SquareActivation()
    elif activation == "tanh" : activation_function = nn.Tanh()
    else :
      print("invalid activation function, square taken instead")
      activation_function = SquareActivation()
    #elements of the network
    self.linear = MaskedLinear(in_dim, in_dim, indices_mask_linear)
    self.non_linear = nn.Sequential(MaskedLinear(in_dim, hidden_dim, indices_mask1), activation_function, MaskedLinear(hidden_dim, in_dim, indices_mask2))
    self.softmax = nn.Softmax(dim=2)   
    (N,L,K) = original_shape
    self.L = L
    self.K = K
  def forward(self, x):
    #combine linear and non-linear models
    x = self.linear(x) + self.non_linear(x)
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K))
    x = self.softmax(x)
    x = torch.reshape(x, (len(x), self.L*self.K))   
    return x
####################################################################################################################################
####################################################################################################################################

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, indices, data, labels):
    """
    indices: indices of the data and labels that will be used to create the dataset
    data: input data
    labels: labels corresponding to the input data
    """
    self.indices = indices
    self.data = torch.from_numpy(data[indices])
    self.labels = torch.from_numpy(labels[indices])
  def __len__(self):
    #Denotes the total number of samples
    return len(self.indices)
  def __getitem__(self, index):
    #Generates one sample of data
    return self.data[index], self.labels[index]
  
def loss_function(output, labels) : #modification 26.02.24
  """
  loss_function that will be used to train the model
  it corresponds cross entropy loss with one hot encoded inputs and labels
  """
  # Ensure both output and labels are on the same device
#  output_device = output.device
 # labels_device = labels.device

  #if output_device != labels_device:
      # Move output to the device of labels
  # Flatten both output and labels
   #   output = output.to(labels_device)
  #flat_output = torch.flatten(output).float()  # Convert to float
  #flat_labels = torch.flatten(labels).float()  # Convert to float
  #loss = -torch.dot(flat_labels, torch.log(flat_output))
  #return loss
  return nn.functional.cross_entropy(output, labels)

def train(train_points, train_labels, model, loss_function):
  'train function of the neural networks'
  pred = model(train_points)
  loss = loss_function(torch.flatten(pred), torch.flatten(train_labels)) 
  return loss 

def error(data, labels, model, original_shape) :
  """
  classification error achieved by the model on the given data
  labels: labels corresponding to points
  model: model to be evaluated
  original_shape: original shape of the MSA (N,L,K)
  """
  #CUDA
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  labels = labels.to(device)
  data = data.to(device)
  N = len(data)#number of sequences
  (_,L,K) = original_shape
  model.eval()
  with torch.no_grad() :
    #prediction of points given by model, reshaped to remove one-hot encoding
    _, pred = torch.max(torch.reshape(model(data), (N, L, K)), 2)
    #labels reshaped to remove one-hot encoding
    _, labels = torch.max(torch.reshape(labels, (N, L, K)), 2)
    #total number of predicted amino acids (#sequences * length of sequences)
    total = torch.numel(pred)
    #number of correctly predicted amino acids
    correct = (pred == labels).sum().item()
    #return the fraction of uncorrect predictions
    return(1 - correct / total)
  
def error_positions(data, labels, model, original_shape) :
  """
  labels: labels corresponding to points
  model: model to be evaluated
  original_shape: original shape of the MSA (N,L,K)
  """
  #CUDA
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  labels = labels.to(device)
  data = data.to(device)
  N = len(data)
  (_,L,K) = original_shape
  model.eval()
  with torch.no_grad() :
    #prediction of points given by model, reshaped to remove one-hot encoding
    _, pred = torch.max(torch.reshape(model(data), (N, L, K)), 2)
    #labels reshaped to remove one-hot encoding
    _, labels = torch.max(torch.reshape(labels, (N, L, K)), 2)
    #error rate per position
    errors_positions = []
    for position in range(L) :
      #number of correctly predicted amino acids for this position 
      correct = (pred[:, position] == labels[:, position]).sum().item()
      errors_positions.append(1 - correct / N)
    return errors_positions
  
def get_data_labels(MSA_file, weights_file, device, max_size = None) :
  """
  This function is used to get the input data and the labels from the MSA (N,L). It also returns the shape of the MSA (N,L,K)
  and a new file of shape (K,L) data_per_col that will be used to mask the model when an amino acid from 1 to K is not possible at a position
  input:
        MSA_file        ->  name of the file containing the preprocessed MSA (only numbers, from 0 to K, separated by commas)
                            shape (N,L) with N the numbers of sequences and L the length of the sequences
        weights_file    ->  name of the file containing the weights of the sequences 
                            shape (N,1) with N the numbers of sequences  
                            If the sequence has a lot of similar sequences (80% same a.a) in the MSA, the weight is low
        max_size        ->  maximum number of sequences to be used, if None all the sequences will be used
  output:
        new_data        ->  input data in one hot encoding form
                            shape (N, L*K) with N the numbers of sequences and L the length of the sequences
        labels          ->  labels corresponding to the input data
                            shape (N, L*K) with N the numbers of sequences and L the length of the sequences
        (N,L,K)         ->  original shape of the MSA
        data_per_col    ->  for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
                            shape (K,L) with K the number of different amino acids and L the length of the sequences
  """
  #load data and weights
  data = np.genfromtxt(MSA_file, delimiter=',').astype(int)
  weights = np.loadtxt(weights_file)
  weights = weights.reshape((len(data), 1, 1))
  if max_size is not None and max_size < len(data) :
    data = data[:max_size]
    weights = weights[:max_size]
  #put the data in one hot encoding form
  new_data = np.array(nn.functional.one_hot(torch.Tensor(data).to(torch.int64)))
  (N,L,K) = new_data.shape
  #data_per_col: for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
  data_per_col=np.ones((K,data.shape[1])) 
  print("The shape of data_per_col is ",data_per_col.shape)
  #check for each column which number is present and put 0 in the corresponding column of data_per_col
  for i,col in enumerate(data.T):
      for k in range(K):
          if k in col:
              data_per_col[k,i]=0
  #data_per_col = torch.tensor(data_per_col, device=device)

  #the labels are the weighted data
  labels = weights * new_data
  #reshape such that each sequence has only one dimension
  new_data = np.reshape(new_data, (N, L * K))
  labels = np.reshape(labels, (N, L * K))
  print("Data and labels have been successfully obtained")
  return new_data, labels, (N,L,K), data_per_col

def create_datasets(data, labels, separations) :
  """
  This function is used to create the training, validation and test datasets
  data: input data
  labels: labels corresponding to the input data
  separations: list of 2 floats, the first one is the fraction of the data that will be used for training, the second one is the fraction of the data that will be used for validation
  """
  #compute the indices of the 3 datasets
  indices = np.array(range(len(data)))
  np.random.shuffle(indices)
  train_indices, validation_indices, test_indices = np.split(indices, [int(separations[0]*len(data)), int(separations[1]*len(data))])
  #create training, validation and test dataset
  training_set = Dataset(train_indices, data, labels)
  validation_set = Dataset(validation_indices, data, labels)
  test_set = Dataset(test_indices, data, labels)
  
  return training_set, validation_set, test_set
def write_the_optimizer(model, optimizer):
    ''' 
    write the optimizer according to the name of the optimizer and the different parameters to initialize
    '''
    print("optimizer writting...")
    optimizer_name=optimizer["name"]
    if optimizer_name=="Adam":
          return torch.optim.__dict__[optimizer_name](model.parameters(), lr=optimizer["lr"], betas=(optimizer["beta1"], optimizer["beta2"]), eps=optimizer["epsilon"], weight_decay=optimizer["weight_decay"], amsgrad=optimizer["amsgrad"])
    elif optimizer_name=="SGD":
          return torch.optim.__dict__[optimizer_name](model.parameters(), lr=optimizer["lr"], momentum=optimizer["momentum"], dampening=optimizer["dampening"], weight_decay=optimizer["weight_decay"], nesterov=optimizer["nesterov"])
    elif optimizer_name=="AdamW":
          return torch.optim.__dict__[optimizer_name](model.parameters(), lr=optimizer["lr"], betas=(optimizer["beta1"], optimizer["beta2"]), eps=optimizer["epsilon"], weight_decay=optimizer["weight_decay"], amsgrad=optimizer["amsgrad"])
    elif optimizer_name=="Adagrad":
        return torch.optim.__dict__[optimizer_name](model.parameters(), lr=optimizer["lr"], lr_decay=optimizer["lr_decay"], weight_decay=optimizer["weight_decay"], initial_accumulator_value=optimizer["initial_accumulator_value"], eps=optimizer["eps"])
    elif optimizer_name=="Adadelta":
        return torch.optim.__dict__[optimizer_name](model.parameters(), lr=optimizer["lr"], rho=optimizer["rho"], eps=optimizer["eps"], weight_decay=optimizer["weight_decay"])
    else:
        print('optimizer name', optimizer_name)
        print("Error: optimizer list is not correct or need to be defined in file model.py")
        return
    
def build_and_train_model(data,labels, original_shape, separation, model_type,activation, nb_hidden_neurons, max_epochs, batch_size, validation, test, optim, device, use_cuda, path):
  #print the memory usage of the cpu and the gpu
  print_cpu_usage()
  
  list_train_err = []
  list_val_err = []
  list_test_err = []
  print("Training duration : ", max_epochs, "epochs")
  print("Model type : ", model_type)
  print("creation datasets...")
  training_set, validation_set, test_set = create_datasets(data, labels, separation)
  print("size training_set",training_set.data.shape)
  print("size validation_set",validation_set.data.shape)
  print("size test_set",test_set.data.shape)
  print("the datasets have been successfully created")
  (N,L,K) = original_shape
  params = {'batch_size': batch_size, #modif
            'shuffle': True,
            'num_workers':1,
            'pin_memory': True if use_cuda else False} #change 28.02.24 for the cluster
    #        'num_workers': 2}
    #create data loader for the training_set
  training_generator = torch.utils.data.DataLoader(training_set, **params)
   #define the model according to model_type (linear, non-linear or mix) and, if not linear, activation (square or tanh)
  print("-----------------------------")
  ##########################################################################
  ################# CREATE THE LIST OF MODELS AND OPTIMIZERS ###############
  ##########################################################################
  if model_type == "linear" :
    print("Initialisation of the linear model...")   
    model=LinearNetwork(path+"/indice_mask.txt", training_set.labels.shape[1], original_shape)
    model=model.to(device)
  elif model_type == "non-linear" :
    print(f"Initialisation of the non linear model with activation {activation}...")
    model=NonLinearNetwork(path+"/indice_mask1.txt", path+"/indice_mask2.txt", L*K, L*nb_hidden_neurons, original_shape)
    model=model.to(device)
  else:
    print("For now only the linear and non linear model are implemented")
  
  print("-----------------------------")
  print("Initialisation of the optimizer...")   
  optimizer= write_the_optimizer(model, optim)
  print("-----------------------------")
  print("-----------------------------")
  print("Training...")  
  # Loop over epochs
  for epoch in range(max_epochs):
    model.train()
    optimizer=optimizer #get the optimizer for the current model
    for local_batch, local_labels in training_generator:
        optimizer.zero_grad()
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        loss = train(local_batch, local_labels, model, loss_function) #get the loss for the current model
        loss.backward() #backpropagation
        optimizer.step() #update the weights
    # Compute and store the errors
    train_err = error(training_set.data, training_set.labels, model, original_shape)
    list_train_err.append(train_err)
    if epoch%10==0:
      print("epoch: ", epoch)
      print("train error model: ", train_err)
    if validation==True:
        val_err = error(validation_set.data, validation_set.labels, model, original_shape)
        list_val_err.append(val_err)
        if epoch%10==0:
          print("test error model: ", test_err)
    if test==True:
        test_err = error(test_set.data, test_set.labels, model, original_shape)
        list_test_err.append(test_err)
        if epoch%10==0:
          print("test error model: ", test_err)
    #print(" Epoch {} complete".format(epoch))
  ##################################################################################
  ##################################################################################
    
  #put all the errors in a list
  errors = [list_train_err]
  if validation == True :
    errors.append(list_val_err)
  if test == True :
    errors.append(list_test_err)
  errors_positions = [error_positions(training_set.data, training_set.labels, model, original_shape)]
  if validation == True : errors_positions.append(error_positions(validation_set.data, validation_set.labels, model, original_shape))
  if test == True : errors_positions.append(error_positions(test_set.data, test_set.labels, model, original_shape)) 
  #delete the datasets to free memory
  del training_set
  del validation_set
  del test_set
  gc.collect()
  return model, errors, errors_positions
  
def execute(MSA_file, weights_file, model_params, length_prot1, path="", output_name='/',errors_computation=False) :
  
  """
  MSA_file: name of the file containing the preprocessed MSA
  weights_file: name of the file containing the weights of the sequences
  model_params: name of the file containing the parameters of the model
  activation: if model_type is "non-linear" or "mix", activation will be the activation function of the hidden layer, can be "square"
      or "tanh", if model_type is "linear" this parameter will be ignored
  path: path to the folder where the model will be saved
  output_name: name of the file where the model will be saved

  """

  print("----------------------------") 
  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  # Additional check to print GPU details if available
  if use_cuda:
      print(torch.cuda.get_device_name(0))  # Print the name of the GPU
      print(torch.cuda.memory_allocated(0))  # Print the current GPU memory usage
      print(torch.cuda.memory_cached(0))  # Print the current GPU memory cache
  else:
      print("No GPU available, using the CPU instead.")
  torch.backends.cudnn.benchmark = True
  print("----------------------------")
  

  ###################################################################################
  ################# LOAD OF THE LEARNING PARAMETERS #################################
  ###################################################################################
  # Open the file in read mode
  with open(model_params, 'r') as file:
      # Read lines from the file
      lines = file.readlines()
  # Initialize an empty dictionary to store the arguments
  arguments = {}
  # Iterate through each line and extract key-value pairs
  for line in lines:
      # Split each line into key and value using ": "
      key, value = map(str.strip, line.split(': '))
        # If the value contains a comma, split it further
      if ',' in value:
          value = [v.strip() for v in value.split(',')]
      else:
         # Convert numeric values to int or float
        value = int(value) if value.isdigit() else float(value) if '.' in value else value
      # Store key-value pair in dictionary
      arguments[key] = value
  #learning parameters : 
  print("-------- load of the learning parameters --------")
  max_epochs=int(arguments.get("epochs"))
  print("max_epochs: ",max_epochs)
  batch_size=int(arguments.get("batchs"))
  print("batch_size: ",batch_size)
  model_type=arguments.get("model_type")
  print("model_type: ",model_type)
  n_models=int(arguments.get("n_models"))
  print("n_models: ",n_models)
  activation=arguments.get("activation")
  nb_hidden_neurons=int(arguments.get("nb_hidden_neurons"))
  if model_type=="non-linear":
    print("activation: ",activation)
    print("nb_hidden_neurons: ",nb_hidden_neurons)
  validation=str(arguments.get("Validation"))#is "True" or "False"
  number_seed=int(arguments.get("seed"))
  np.random.seed(number_seed)
  torch.manual_seed(number_seed)
  #put the type of validation as boolean:
  if validation=="True":
      validation=True
  else:
      validation=False
  print("validation: ",validation)
  test=arguments.get("Test")
  #put the type of validation as boolean:
  if test=="True":
      test=True
  else:
      test=False
  print("test: ",test)
  separation=arguments.get("separation")
  #we have separation=['(0.7', '0.8)'] we want separation=(0.7,0.8) with 0.7 and 0.8 double numbers:
  separation=separation[0]+","+separation[1]
  separation=separation.replace("(","")
  separation=separation.replace(")","")
  separation=separation.split(",")
  separation=(float(separation[0]),float(separation[1]))
  print("separation: ",separation)
  #it is special for optimizer because we have a list of the different values of the keys of a dictionary. For example we can have:
  #name=SGD, lr=0.01, momentum=0.001, dampening=False, weight_decay=0, nesterov=False
  #so we want to stock them as a dictionary:
  #optimizer={"name":"SGD", "lr":0.01, "momentum":0.001, "dampening":False, "weight_decay":0, "nesterov":False}
  optimizer={}
  optimizer["name"]=str(arguments.get("optimizer")[0])
  print("-------- load of the optimizer parameters --------")
  if optimizer["name"]=="Adam":
      optimizer["lr"]=float(arguments.get("optimizer")[1])
      optimizer["beta1"]=float(arguments.get("optimizer")[2])
      optimizer["beta2"]=float(arguments.get("optimizer")[3])
      optimizer["epsilon"]=float(arguments.get("optimizer")[4])
      optimizer["weight_decay"]=float(arguments.get("optimizer")[5])
      optimizer["amsgrad"]=arguments.get("optimizer")[6]
  elif optimizer["name"]=="AdamW":
      optimizer["lr"]=float(arguments.get("optimizer")[1])
      optimizer["beta1"]=float(arguments.get("optimizer")[2])
      optimizer["beta2"]=float(arguments.get("optimizer")[3])
      optimizer["epsilon"]=float(arguments.get("optimizer")[4])
      optimizer["weight_decay"]=float(arguments.get("optimizer")[5])
      optimizer["amsgrad"]=arguments.get("optimizer")[6]
  elif optimizer["name"]=="SGD":
      optimizer["lr"]=float(arguments.get("optimizer")[1])
      optimizer["momentum"]=float(arguments.get("optimizer")[2])
      optimizer["dampening"]=float(arguments.get("optimizer")[3])
      optimizer["weight_decay"]=float(arguments.get("optimizer")[4])
      optimizer["nesterov"]=arguments.get("optimizer")[5]
  elif optimizer["name"]=="Adagrad":
      optimizer["lr"]=float(arguments.get("optimizer")[1])
      optimizer["lr_decay"]=float(arguments.get("optimizer")[2])
      optimizer["weight_decay"]=float(arguments.get("optimizer")[3])
      optimizer["initial_accumulator_value"]=float(arguments.get("optimizer")[4])
      optimizer["eps"]=float(arguments.get("optimizer")[5])
  elif optimizer["name"]=="Adadelta":
      optimizer["lr"]=float(arguments.get("optimizer")[1])
      optimizer["rho"]=float(arguments.get("optimizer")[2])
      optimizer["eps"]=float(arguments.get("optimizer")[3])
      optimizer["weight_decay"]=float(arguments.get("optimizer")[4])
  for key in optimizer:
    print(key,": ",optimizer[key])
  ###################################################################################
  ###################################################################################
  if path=="":#default value
    path_folder=weights_file.split("/")[:-1] #take the path of the input file
    path_folder="/".join(path_folder)
    path=path_folder+"/model_"+model_type+"-"+str(max_epochs)+"epochs-"+str(batch_size)+"batch_size/seed"+str(number_seed)
    print("The folder path (where the model(s) and data_per_col.txt will be saved) is: ",path)
  #check if the path to the folder exist if not create it
  if not os.path.exists(path):
    os.makedirs(path)
  ###################################################################################
  ################# LOAD OF THE DATA AND LABELS #####################################
  ################################################################################### 
  print("-------- load of the data and label --------")
  data, labels, original_shape, data_per_col = get_data_labels(MSA_file, weights_file, device)
  np.savetxt(path+"/data_per_col.txt", data_per_col, fmt="%d")
  print("MSA shape : ", original_shape)
  ##################################################################################
  ##################################################################################

  ##################################################################################
  ################# BUILD AND TRAIN THE MODEL ######################################
  ##################################################################################
  
  (_,L,K) = original_shape
  if model_type=="linear":
    generate_indices_mask(L, K, data_per_col,path, length_prot1)
  elif model_type=="non-linear":
    generate_indices_mask_non_linear(L, K, nb_hidden_neurons,data_per_col,path)
  else:
    print("For now only the linear and non linear model are implemented")


  ########################################################
  ######## initatialisation average model ################
  ########################################################
  # Initialize the average model with the same architecture as the individual models
  average_model=None
  #The model will be initialized with the right architecture
  #and we will be able to compute the average model
  print("initatialisation average model...") #DONT FORGET TO PUT BACK AFTER
  if model_type=="linear":
    average_model = LinearNetwork(path+"/indice_mask.txt", L*K, original_shape)
  elif model_type=="non-linear":
    average_model = NonLinearNetwork(path+"/indice_mask1.txt", path+"/indice_mask2.txt", L*K, L*nb_hidden_neurons, original_shape, activation)
  else:
    print("For now only the linear and non linear model are implemented")
  average_model = average_model.to(device)
  average_parameters = {key: torch.zeros_like(value).to(device) for key, value in average_model.state_dict().items()}
  #########################################################
  #########################################################

  #########################################################
  ############### initialisation errors ###################
  #########################################################
  avg_train_errors = []
  avg_train_errors_positions = []
  if validation==True:
      avg_validation_errors = []
      avg_validation_errors_positions = []
  if test==True:
      avg_test_errors = []
      avg_test_errors_positions = []

  #########################################################
  #########################################################

  #########################################################
  ################ compute the model(s) ################### 
  #########################################################
  ALLmodel=[]
  ALLerrors=[]
  ALLerrors_positions=[]
  for i in tqdm(range(n_models)):
    '''
    (1) write the name of the model and the errors to save them later
    '''
    if output_name=="/": #the user doesn't want a specific name
      #take only <path>/model_<i>
      model_name = path+"/model_" + str(i)
      errors_name = path +"/errors_" + str(i) + ".txt"
      errors_positions_name = path+ "/errors_positions_" + str(i) + ".txt"
    else: #the user wants a specific name: <path>/model_<output_name><i>
      model_name = path+"/model_" + output_name + "_"+str(i)
      errors_name = path +"/errors_" + output_name + "_"+str(i) + ".txt"
      errors_positions_name = path+ "/errors_positions_" + output_name + "_"+ str(i) + ".txt"
    ALLmodel.append(model_name) #save the name of the model(s) in a list
    ALLerrors.append(errors_name) #save the name of the errors(s) in a list
    ALLerrors_positions.append(errors_positions_name) #save the name of the errors_positions(s) in a list
    ''' 
    (2) build and train the model if it doesn't already exist
    '''
    #look before to compute and save the model if it already exist
    if os.path.exists(model_name):       
      print("model ", model_name, "already exist, we will not overwrite it")
      #we need to ensure the next models will not be trained on the same data
      #by removing the data used for the previous model with the specific probability
      training_set, validation_set, test_set = create_datasets(data, labels, separation) 
      #remove them from the memory
      del training_set
      del validation_set
      del test_set
      

    else:
      #print the seed location
      print("seed location: ",np.random.get_state()[1][0])
      print("-------- model "+str(i)+"--------")
      model, errors, errors_positions = build_and_train_model(data,labels, original_shape, separation, model_type,activation, nb_hidden_neurons, max_epochs, batch_size, validation, test, optimizer, device, use_cuda, path)
      torch.save(model, model_name)
      if errors_computation==True:
        np.savetxt(errors_name, errors)
        np.savetxt(errors_positions_name, errors_positions)
      print("model saved:", model_name)
      print("errors saved:", errors_name)
      print("errors_positions saved:", errors_positions_name)  
      del model #don't keep the model in memory
      del errors
      del errors_positions
      
    gc.collect()
      

  #########################################################
  #########################################################
  
  #########################################################
  ################ compute the average model ##############
  #########################################################
  print("-------- model average --------")
  # Loop over models and accumulate the model parameters saved
  if n_models>1:
  ######################################################### don't forget to put them again after
    for model_n in ALLmodel:
        model=torch.load(model_n)
        model_parameters = model.state_dict()
        for key in model_parameters:
            average_parameters[key] += model_parameters[key]
    # Compute the average parameters for the average model
    for key in average_parameters:
        average_parameters[key] /= n_models
    # Load the average parameters into the average model
  # average_model.load_state_dict(average_parameters)
  #########################################################
    # Compute the average of the train, val,and test errors at each epochs
    #avg_error(epoch_i)=somme(epoch_i)/n_models
    #print("the dtype of errors is ",np.loadtxt(ALLerrors[0]).dtype)
    #print("the shape of errors is ",np.loadtxt(ALLerrors[0]).shape)
    #print("the shape of ALLerrors is ",np.array(ALLerrors).shape)
    if errors_computation==True:
      for i in range(max_epochs):
          avg_train=0
          avg_test=0
          avg_val=0
          comptage=0
          if validation==False and test==False: #errors[i] is composed of only one list
              for errors in ALLerrors:
                  avg_train+=np.loadtxt(errors)[i]
              avg_train_errors.append(avg_train/n_models)
          elif validation==True and test==False: #errors[i] is composed of two lists
              for errors in ALLerrors:
                  avg_train+=np.loadtxt(errors)[0][i]
                  avg_val+=np.loadtxt(errors)[1][i]
              avg_train_errors.append(avg_train/n_models)
              avg_validation_errors.append(avg_val/n_models)
          elif validation==False and test==True:
              for errors in ALLerrors:
                  avg_train+=np.loadtxt(errors)[0][i]
                  avg_test+=np.loadtxt(errors)[1][i]
              avg_train_errors.append(avg_train/n_models)
              avg_test_errors.append(avg_test/n_models)           
              
          else: #validation and test are true
              for errors in ALLerrors:
                  avg_train+=np.loadtxt(errors)[0][i]
                  avg_val+=np.loadtxt(errors)[1][i]
                  avg_test+=np.loadtxt(errors)[2][i]
              avg_train_errors.append(avg_train/n_models)
              avg_validation_errors.append(avg_val/n_models)
              avg_test_errors.append(avg_test/n_models)
      print("-------- plot the curve --------")
      #plot learning curve
      plt.plot(range(len(avg_train_errors)), avg_train_errors, label="train error")
      plt.plot(range(len(avg_test_errors)), avg_test_errors, label="test error")
      plt.ylabel("categorical error")
      plt.xlabel("epoch")
      plt.legend()
      plt.grid()
      #save the plot in the folder
      plt.savefig(path+"/learning_curve.png")
      print("-------- end --------")
    else:
      print("only one model, the average model is the model")
      average_model=torch.load(ALLmodel[0])
      avg_train=0
      avg_test=0
      avg_val=0
      if validation==False and test==False:
          for errors in ALLerrors:
              avg_train+=np.loadtxt(errors)[0]
          avg_train_errors.append(avg_train)
      elif validation==True and test==False:
          for errors in ALLerrors:
              avg_train+=np.loadtxt(errors)[0][0]
              avg_val+=np.loadtxt(errors)[1][0]
          avg_train_errors.append(avg_train)
          avg_validation_errors.append(avg_val)
      elif validation==False and test==True:
          for errors in ALLerrors:
              avg_train+=np.loadtxt(errors)[0][0]
              avg_test+=np.loadtxt(errors)[1][0]
          avg_train_errors.append(avg_train)
          avg_test_errors.append(avg_test)
      else: #validation and test are true
          for errors in ALLerrors:
              avg_train+=np.loadtxt(errors)[0][0]
              avg_val+=np.loadtxt(errors)[1][0]
              avg_test+=np.loadtxt(errors)[2][0]
          avg_train_errors.append(avg_train)
          avg_validation_errors.append(avg_val)
          avg_test_errors.append(avg_test)
  
  
    #put the errors in a list
    #avg_train_errors_list = [list(errors) for errors in avg_train_errors]
    #print it
    print("avg_train_errors_list: ",avg_train_errors)
    #avg_train_errors_positions_list = [list(errors) for errors in avg_train_errors_positions]
    avg_errors = [avg_train_errors]
    #avg_errors_positions = [avg_train_errors_positions_list]
    if validation == True :
      avg_validation_errors_list = [list(errors) for errors in avg_validation_errors]
      #avg_validation_errors_positions_list = [list(errors) for errors in avg_validation_errors_positions]
      avg_errors.append(avg_validation_errors_list)
      #avg_errors_positions.append(avg_validation_errors_positions_list)
    if test == True :
      #avg_test_errors_list = [list(errors) for errors in avg_test_errors]
      #avg_test_errors_positions_list = [list(errors) for errors in avg_test_errors_positions]
      avg_errors.append(avg_test_errors)
      #avg_errors_positions.append(avg_test_errors_positions_list)
  
  
  #save the average model
  if output_name=="/":
    model_name = path+"/model_" + "average_0-" + str(n_models-1) 
    if errors_computation==True:
      avg_errors_name = path +"/errors_0-" + str(n_models-1) + ".txt"
      #avg_errors_positions_name = path+ "/errors_positions_0-" + str(n_models-1) + ".txt"
  else:
    model_name = path+ "/model_" + output_name + "average_0-" + str(n_models-1)
    if errors_computation==True:
      avg_errors_name = path +"/errors_" + output_name + "_0-" + str(n_models-1) + ".txt"
      #avg_errors_positions_name = path+ "/errors_positions_" + output_name + "_0-" + str(n_models-1) + ".txt"
  if os.path.exists(model_name):       
    print("model ", model_name, "already exist, we will not overwrite it")
  else:
    torch.save(average_model, model_name)
    print("average model saved:", model_name)
    if errors_computation==True:
      np.savetxt(errors_name, avg_errors)
      #np.savetxt(errors_positions_name, avg_errors_positions)
      
      print("average errors saved:", avg_errors_name)
    #print("average errors_positions saved:", avg_errors_positions_name)

  ##########################################################################
  ##########################################################################
##################################################################################
##################################################################################
  if errors_computation==True:
    ##################################################################################
    print("-------- plot the curve --------")
    #plot learning curve
    plt.plot(range(len(avg_train_errors)), avg_train_errors, label="train error")
    plt.plot(range(len(avg_test_errors)), avg_test_errors, label="test error")
    plt.ylabel("categorical error")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    #save the plot in the folder
    plt.savefig(path+"/learning_curve.png")
    print("-------- end --------")
    ##################################################################################
    ##################################################################################
