import argparse
import learning_param

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("epochs",help="The number of epochs for example 15")
    parser.add_argument("batchs", help="the number of batchs for example 32")
    parser.add_argument("model_type", help="the type of model in string for example linear")
    parser.add_argument("n_models", help="the number of models to trains in order to do an average")
    parser.add_argument("separation", help="gives the proporition of training, validation and test (0.7,0.7) will keep 70% of training, 0% of validation and 30% for the test, (0.7,0.8) will keep 70% of training, 10% of validation and 20% for the test ")
    parser.add_argument("nb_hidden_neurons", help="Number of hidden neurons")
    parser.add_argument("optimizer", help="list with the optimizer name and its different parameters, for example '[SGD,0.01, 1.e-2, False]' ")
    parser.add_argument("seed", help="seed for the random number generator")
    parser.add_argument("output_name", help="Name for the output file.")
    args=parser.parse_args()
    
    learning_param.execute(args.epochs, args.batchs, args.model_type, args.n_models,args.separation,  args.optimizer, args.nb_hidden_neurons,args.seed,args.output_name)