import argparse
import learning_param

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("epochs",help="The number of epochs for example 15", type=str)
    parser.add_argument("batchs", help="the number of batchs for example 32", type=str)
    parser.add_argument("model_type", help="the type of model in string for example linear", type=str)
    parser.add_argument("n_models", help="the number of models to trains in order to do an average", type=str)
    parser.add_argument("seed", help="seed for the random number generator", type=str)
    parser.add_argument("output_name", help="Name for the output file.", type=str)
    parser.add_argument("-activation", help="activation function of hidden layer if model_type is 'non-linear' or 'mix', can be 'square' or 'tanh', if model_type is 'linear' the parameter will be ignored. default='square'", default="square", type=str)
    parser.add_argument("-nb_hidden_neurons", help="Number of hidden neurons. Not used if linear type.", default=32, type=int)
    parser.add_argument("-optimizer", metavar='N', type=str, nargs='+', help="list with the optimizer name and its different parameters, for example SGD,0.01,1.e-2,False. Default=SGD,0.008,0.01,0,0,False", default="/")
    parser.add_argument("-separation", metavar='N', type=str, nargs='+', help="gives the proporition of training, validation and test (0.7,0.7) will keep 70% of training, 0% of validation and 30% for the test, (0.7,0.8) will keep 70% of training, 10% of validation and 20% for the test. Default=0.7,0.7", default="/")
    args=parser.parse_args()

    learning_param.execute(args.epochs, args.batchs, args.model_type, args.n_models,args.seed,args.output_name,args.activation, args.nb_hidden_neurons, args.optimizer,args.separation)
