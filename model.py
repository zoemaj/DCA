import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
#nohapƒ


class SquareActivation(nn.Module) :
  'custom square activation'
  def __init__(self) :
    super().__init__()
  def forward(self, input) :
    return input.square()
  


class MaskedLinear(nn.Module):
  """
  linear model of one non fully connected layer

  build a non fully connected layer by putting a mask on the weights that must be null
  """
  def __init__(self, in_dim, out_dim, indices_mask):
    """
    in_dim: number of input features
    out_dim: number of output features
    indices_mask: list of tuples of int
    """
    super(MaskedLinear, self).__init__()
        
    self.linear = nn.Linear(in_dim, out_dim) #MaskedLinear is made of a linear layer
    #Force the weights indicated by indices_mask to be zero by use of a mask
    self.mask = torch.zeros([out_dim, in_dim]).bool()
    for a, b in indices_mask : self.mask[(a, b)] = 1

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




class LinearNetwork(nn.Module):
  'linear model with softmax activation on the output layer applied on residue positions'

  def __init__(self, indices_mask, in_dim, original_shape):
    """
    indices_mask: list of input and output neurons that must be disconnected
    in_dim: dimension of input layer
    original_shape: original shape of the MSA (N,L,K)
    """
    super(LinearNetwork, self).__init__()

    self.masked_linear = MaskedLinear(in_dim, in_dim, indices_mask) 
    self.softmax = nn.Softmax(dim=2) 

    (_,L,K) = original_shape
    self.L = L
    self.K = K

  def forward(self, x):
    x = self.masked_linear(x)
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K))
    x = self.softmax(x)
    x = torch.reshape(x, (len(x), self.L*self.K))

    return x




class NonLinearNetwork(nn.Module):
  'model with hidden layer and square / tanh activation on the hidden layer'
  def __init__(self, indices_mask1, indices_mask2, in_dim, hidden_dim, original_shape, activation="square"):
    """
    indices_mask1: list of input and hidden neurons that must be disconnected
    indices_mask2: list of hidden and output neurons that must be disconnected
    in_dim: dimension of input layer
    hidden_dim: dimension of hidden layer
    original_shape: original shape of the MSA (N,L,K)
    activation: activation for the hidden layer, must be "square" or "tanh" otherwise square is taken by default
    """
    super(NonLinearNetwork, self).__init__()

    #define activation function
    if activation == "square" : activation_function = SquareActivation()
    elif activation == "tanh" : activation_function = nn.Tanh()
    else :
      print("invalid activation function, square taken instead")
      activation_function = SquareActivation()

    #elements of the network
    self.non_linear = nn.Sequential(MaskedLinear(in_dim, hidden_dim, indices_mask1), activation_function, MaskedLinear(hidden_dim, in_dim, indices_mask2))
    self.softmax = nn.Softmax(dim=2)
    
    (_,L,K) = original_shape
    self.L = L
    self.K = K

  def forward(self, x):
    x = self.non_linear(x)
    #apply softmax on residues
    x = torch.reshape(x, (len(x), self.L, self.K))
    x = self.softmax(x)
    x = torch.reshape(x, (len(x), self.L*self.K))
    return x
  



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




def get_data_labels(MSA_file, weights_file, max_size = None) :
  """
  MSA_file: name of the file containing the preprocessed MSA
  weights_file: name of the file containing the weights of the sequences
  max_size: maximum number of sequences to be used, if None all the sequences will be used
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

  #the labels are the weighted data
  labels = weights * new_data

  #reshape such that each sequence has only one dimension
  new_data = np.reshape(new_data, (N, L * K))
  labels = np.reshape(labels, (N, L * K))

  print("Data and labels have been successfully obtained")
  return new_data, labels, (N,L,K), data_per_col




def create_datasets(data, labels, separations) :
  """
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
  
  print("size training_set",training_set.data.shape)
  print("size validation_set",validation_set.data.shape)
  print("size test_set",test_set.data.shape)
  print("the datasets have been successfully created")
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
    


def build_and_train_model(data, labels, original_shape, data_per_col, model_type, n_models, optimizer, activation, batch_size, max_epochs,nb_hidden_neurons, validation,test, separation) :
  """
  data: input data
  labels: labels corresponding to the input data
  original_shape: original shape of the MSA (N,L,K)
  data_per_col: for each colomn if there is a one at an amino position it means that it is not possible to have this position! -> need to be masked in the model
  model_type: type of model, can be "linear", "non-linear" or "mix"
  n_models: number of models to be trained, if n_models > 1, the average model will also be trained
  optimizer: dictionary containing the parameters of the optimizer
  activation: if model_type is "non-linear" or "mix", activation will be the activation function of the hidden layer, can be "square"
      or "tanh", if model_type is "linear" this parameter will be ignored
  batch_size: size of the batch
  max_epochs: number of epochs
  nb_hidden_neurons: number of neurons in the hidden layer
  validation: if True, the validation error will be computed
  test: if True, the test error will be computed
  separation: list of 2 floats, the first one is the fraction of the data that will be used for training, the second one is the fraction of the data that will be used for validation

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
  print("Training duration : ", max_epochs, "epochs")
  print("Model type : ", model_type)
  if model_type == "non-linear" or model_type == "mix" : print("Activation function : ", activation)
  if n_models>1:
    ALLmodel=[]
    ALLtraining_set=[]
    ALLvalidation_set=[]
    ALLtest_set=[]
    average_model=None

    for i in range(n_models): 
      print("-------- model "+str(i)+"--------")
      training_set, validation_set, test_set = create_datasets(data, labels, separation)
      ALLtraining_set.append(training_set)
      ALLvalidation_set.append(validation_set)
      ALLtest_set.append(test_set)
    separation_big_model=(0.1,0.1) #almost only test data
    print("-------- model average --------")
    training_set, validation_set, test_set = create_datasets(data, labels, separation_big_model)
    
    ALLtraining_set.append(training_set)
    ALLvalidation_set.append(validation_set)
    ALLtest_set.append(test_set)
      

    
    (N,L,K) = original_shape
    ALLtraining_generator=[]


    for training_set in ALLtraining_set:
      params = {'batch_size': batch_size, #modif
              'shuffle': True,
              'num_workers':1} #change 28.02.24 for the cluster
      #        'num_workers': 2}
      

      #create data loader for the training_set
      training_generator = torch.utils.data.DataLoader(training_set, **params)
      ALLtraining_generator.append(training_generator)
        

    #define the model according to model_type (linear, non-linear or mix) and, if not linear, activation (square or tanh)
    if model_type == "linear" :
      #list of tuples (input, output) that we want to be disconnected in the model
      indices_mask = []
      print("writing the indices_mask")
      for j in range(0, L*K, K) :
        for a in range(j, j+K) :
          indices_mask += [(a, b) for b in range(j, j+K)]
      for col_i in range(L):
        for amino in range(K):
          if data_per_col[amino,col_i]==1:
            #add (col_i*K+amino,b) with b is all the others possibilities in the others columns
            current_bloc=col_i*K
            index_amino=col_i*K+amino
            #remove the connection between index_amino and all the amino in the blocks before and the reciprocal also
            indices_mask += [(index_amino, b) for b in range(current_bloc)]
            indices_mask += [(b, index_amino) for b in range(current_bloc)]
            #remove the connection between index_amino and all the amino in the blocks after and the reciprocal also
            indices_mask += [(index_amino, b) for b in range(current_bloc+K,L*K)]
            indices_mask += [(b, index_amino) for b in range(current_bloc+K,L*K)]
      
        
    
      print("original_shape",original_shape)
      print("-----------------------------")
      print("Initialisation of the models...")   
      for training_set in ALLtraining_set:
          model = LinearNetwork(indices_mask, training_set.labels.shape[1], original_shape)
          model=model.to(device)
          ALLmodel.append(model)
      
      print("-----------------------------")
      print("Initialisation of the optimizer...")   
      optimizer_list = [ write_the_optimizer(model, optimizer) for model in ALLmodel]
      
      print("-----------------------------")
      print("Initialisation of the average model...")   
      # Initialize the average model with the same architecture as the individual models
      average_model = LinearNetwork(indices_mask, training_set.labels.shape[1], original_shape)
      average_model = average_model.to(device)
      # Initialize the parameters for averaging
      # Initialize the parameters for averaging
      average_parameters = {key: value.to(device) for key, value in average_model.state_dict().items()}



      # Initialize lists to store errors and errors_positions for each model
      avg_train_errors = []
      if validation==True:
          avg_validation_errors = []
      if test==True:
          avg_test_errors = []


      # List to accumulate the gradients of all models
      #grad_accumulator = [torch.zeros_like(param) for param in ALLmodel[0].parameters()]
      print("-----------------------------")
      print("Training...")
    
      # Loop over epochs
      for epoch in range(max_epochs):
        # Display the current epoch
        print("epoch =", epoch)
        for i, model in enumerate(ALLmodel):
          model.train()
          optimizer=optimizer_list[i]
          training_generator = ALLtraining_generator[i]
          training_set = ALLtraining_set[i]
          for local_batch, local_labels in training_generator:
              optimizer.zero_grad()
              local_batch, local_labels = local_batch.to(device), local_labels.to(device)
              loss = train(local_batch, local_labels, model, loss_function)
              loss.backward()
              optimizer.step() 

      # Compute the average model based on the accumulated gradients
        # Initialize the parameters for averaging
        average_parameters = {key: torch.zeros_like(value).to(device) for key, value in average_model.state_dict().items()}

            # Loop over models and accumulate the model parameters
        for model in ALLmodel:
            model_parameters = model.state_dict()
            for key in model_parameters:
                average_parameters[key] += model_parameters[key]

        # Compute the average parameters
        for key in average_parameters:
            average_parameters[key] /= n_models
        # Load the average parameters into the average model
        average_model.load_state_dict(average_parameters)
        # Compute and store the errors
        training_set=ALLtraining_set[n_models]
        validation_set=ALLvalidation_set[n_models]
        test_set=ALLtest_set[n_models]

        train_err = error(training_set.data, training_set.labels, average_model, original_shape)
        avg_train_errors.append(train_err)
        print("train error average model: ", train_err)

        if validation==True:
            val_err = error(validation_set.data, validation_set.labels, average_model, original_shape)
            avg_validation_errors.append(val_err)
        if test==True:

            test_err = error(test_set.data, test_set.labels, average_model, original_shape)
            avg_test_errors.append(test_err)
            print("test error average model: ", test_err)

        print(" Epoch {} complete".format(epoch))

      #put all the errors in a list

      errors = [avg_train_errors]
      if validation == True :
        errors.append(avg_validation_errors)
      if test == True :
        errors.append(avg_test_errors)

      #compute the final error per position

      training_set=ALLtraining_set[n_models]
      validation_set=ALLvalidation_set[n_models]
      test_set=ALLtest_set[n_models]

      errors_positions = [error_positions(training_set.data, training_set.labels, average_model, original_shape)]
      if validation == True : errors_positions.append(error_positions(validation_set.data, validation_set.labels, average_model, original_shape))
      if test == True : errors_positions.append(error_positions(test_set.data, test_set.labels, average_model, original_shape))
  
  else:
    #create the datasets
    print("Only one model")
    training_set, validation_set, test_set = create_datasets(data, labels, separation)
    (N,L,K) = original_shape

    params = {'batch_size': batch_size, #modif
              'shuffle': True,
              'num_workers':1} #change 28.02.24 for the cluster
      #        'num_workers': 2}
    #create data loader for the training_set
    training_generator = torch.utils.data.DataLoader(training_set, **params)


    #define the model according to model_type (linear, non-linear or mix) and, if not linear, activation (square or tanh)
    if model_type == "linear" :
      #list of tuples (input, output) that we want to be disconnected in the model
      indices_mask = []
      print("writing the indices_mask")
      for j in range(0, L*K, K) :
        for a in range(j, j+K) :
          indices_mask += [(a, b) for b in range(j, j+K)]
      for col_i in range(L):
        for amino in range(K):
          if data_per_col[amino,col_i]==1:
            #add (col_i*K+amino,b) with b is all the others possibilities in the others columns
            current_bloc=col_i*K
            index_amino=col_i*K+amino
            #remove the connection between index_amino and all the amino in the blocks before and the reciprocal also
            indices_mask += [(index_amino, b) for b in range(current_bloc)]
            indices_mask += [(b, index_amino) for b in range(current_bloc)]
            #remove the connection between index_amino and all the amino in the blocks after and the reciprocal also
            indices_mask += [(index_amino, b) for b in range(current_bloc+K,L*K)]
            indices_mask += [(b, index_amino) for b in range(current_bloc+K,L*K)]
    
    average_model = LinearNetwork(indices_mask, training_set.labels.shape[1], original_shape)
    average_model=average_model.to(device)
    optimizer = write_the_optimizer(average_model, optimizer)

    # Initialize lists to store errors and errors_positions for each model
    avg_train_errors = []
    if validation==True:
        avg_validation_errors = []
    if test==True:
        avg_test_errors = []
    print("-----------------------------")
    print("Training...")
    # Loop over epochs
    for epoch in range(max_epochs):
      # Display the current epoch
      if (epoch+1)%10==0:
        print("epoch =", epoch+1)
      average_model.train()
      for local_batch, local_labels in training_generator:
          optimizer.zero_grad()
          local_batch, local_labels = local_batch.to(device), local_labels.to(device)
          loss = train(local_batch, local_labels, average_model, loss_function)
          loss.backward()
          optimizer.step() 
      # Compute and store the errors
      train_err = error(training_set.data, training_set.labels, average_model, original_shape)
      avg_train_errors.append(train_err)
      print("train error average model: ", train_err)

      if validation==True:
          val_err = error(validation_set.data, validation_set.labels, average_model, original_shape)
          avg_validation_errors.append(val_err)
      if test==True:
          test_err = error(test_set.data, test_set.labels, average_model, original_shape)
          avg_test_errors.append(test_err)
          print("test error average model: ", test_err)

      ALLmodel=average_model
          
      #print(" Epoch {} complete".format(epoch))


      #put all the errors in a list
      errors = [avg_train_errors]
      if validation == True :
        errors.append(avg_validation_errors)
      if test == True :
        errors.append(avg_test_errors)


      errors_positions = [error_positions(training_set.data, training_set.labels, average_model, original_shape)]
      if validation == True : errors_positions.append(error_positions(validation_set.data, validation_set.labels, average_model, original_shape))
      if test == True : errors_positions.append(error_positions(test_set.data, test_set.labels, average_model, original_shape))

    
  return ALLmodel, average_model, errors, errors_positions





def execute(MSA_file, weights_file, model_params, activation, path, output_name) :
  
  """
  MSA_file: name of the file containing the preprocessed MSA
  weights_file: name of the file containing the weights of the sequences
  model_params: name of the file containing the parameters of the model
  activation: if model_type is "non-linear" or "mix", activation will be the activation function of the hidden layer, can be "square"
      or "tanh", if model_type is "linear" this parameter will be ignored
  path: path to the folder where the model will be saved
  output_name: name of the file where the model will be saved

  """

  #check if the path to the folder exist if not create it
  if not os.path.exists(path):
      os.makedirs(path)

  #######################################################
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
  nb_hidden_neurons=int(arguments.get("nb_hidden_neurons"))
  print("nb_hidden_neurons: ",nb_hidden_neurons)
  validation=str(arguments.get("Validation"))#is "True" or "False"
  np.random.seed(int(arguments.get("seed")))
  torch.manual_seed(int(arguments.get("seed")))
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

   
  print("-------- load of the data and label --------")

  data, labels, original_shape, data_per_col = get_data_labels(MSA_file, weights_file)
  np.savetxt(path+"/data_per_col.txt", data_per_col, fmt="%d")
  print("MSA shape : ", original_shape)

  ALLmodel, model, errors, errors_positions = build_and_train_model(data, labels, original_shape, data_per_col, model_type,n_models,optimizer, activation=None, batch_size=batch_size, max_epochs=max_epochs,  nb_hidden_neurons=nb_hidden_neurons, validation=validation, test=test, separation=separation) 
  print("-------- plot the curve --------")
  #plot learning curve
  plt.plot(range(len(errors[0])), errors[0], label="train error")
  plt.plot(range(len(errors[1])), errors[1], label="test error")
  plt.ylabel("categorical error")
  plt.xlabel("epoch")
  plt.legend()
  plt.grid()
  plt.show()


  #save model and errors
  #the models from the list ALLmodel:
  if n_models>1:
    for i in range(len(ALLmodel)-1): #write the name with the number of the model
      if output_name=="/":
        model_name = path+ "/model_" + str(i)
      else:
        model_name = path+"/model_" + output_name + str(i)
      print(model_name)
      torch.save(ALLmodel[i], model_name)
    #average model:
    if output_name=="/":
      model_name = path+"/model_" + "average_0-" + str(n_models-1) 
      errors_name = path +"/errors_0-" + str(n_models-1) + ".txt"
      errors_positions_name = path+ "/errors_positions_0-" + str(n_models-1) + ".txt"
    else:
      model_name = path+ "/model_" + output_name + "average_0-" + str(n_models-1)
      errors_name = path +"/errors_" + output_name + "_0-" + str(n_models-1) + ".txt"
      errors_positions_name = path+ "/errors_positions_" + output_name + "_0-" + str(n_models-1) + ".txt"
  else:
    if output_name=="/":
      model_name = path+"/model" 
      errors_name = path +"/errors" + ".txt"
      errors_positions_name = path+ "/errors_positions"  + ".txt"
    else:
      model_name = path+ "/model_" + output_name
      errors_name = path +"/errors_" + output_name + ".txt"
      errors_positions_name = path+ "/errors_positions_" + output_name + ".txt"


  print(model_name)
  
  torch.save(model, model_name)
  np.savetxt(errors_name, errors)
  np.savetxt(errors_positions_name, errors_positions)