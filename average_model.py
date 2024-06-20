import torch
from torch import nn


def compute_average(model_name, n_models) :
    '''
        compute the model average from n_models with name model_name
        input:
            model_name: name of the models to average (without the index number _i)
            L,K: size of the data
            n_models: number of models to average
        output:
            average_model: model average with name model_name_average_0-number_model     
    '''
    print("-------- model average --------")
    models=[]
    n_models=int(n_models)
    for m in range(n_models):          
        models.append(torch.load(model_name+'_' + str(m)))
        print("model ", m, " loaded")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("device: ", device)
    torch.backends.cudnn.benchmark = True
     # Additional check to print GPU details if available
    if use_cuda:
        print(torch.cuda.get_device_name(0))  # Print the name of the GPU
        print(torch.cuda.memory_allocated(0))  # Print the current GPU memory usage
        print(torch.cuda.memory_cached(0))  # Print the current GPU memory cache
    else:
        print("No GPU available, using the CPU instead.")

    #Initialize the average model with the same architecture as the individual models by doing a copy of one model
    average_model = models[0]
    average_model = average_model.to(device)
    # Initialize the parameters for averaging
    average_parameters = average_model.state_dict() #model.state_dict() -> model_parameters

    # List to accumulate the gradients of all models
    #grad_accumulator = [torch.zeros_like(param) for param in ALLmodel[0].parameters()]
    print("-------- accumulation models --------")
    # Loop over models and accumulate the model parameters
    #no the models[0] that is already in average_model
    for model in models[1:]:
        model_parameters = model.state_dict()
        for key in model_parameters:
            average_parameters[key] += model_parameters[key]
    # Compute the average parameters
    for key in average_parameters:
        average_parameters[key] /= n_models
      # Load the average parameters into the average model
    average_model.load_state_dict(average_parameters)
    #save the model 
    output_name= model_name+"_average_0-"+str(n_models-1)
    print(output_name)
  
    torch.save(average_model, output_name)
    
    