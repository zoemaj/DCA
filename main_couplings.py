import argparse
import couplings

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    #parser.add_argument("triplets_name",help="File containing the triplets of the NN obtained using model.py.")
    parser.add_argument("model_name", help="Name of the file where the trained model was saved", type=str)
    parser.add_argument("-number_models", help="number of model to train. Default=1", default=1, type=int)
    parser.add_argument("-type_average", help= "if number_models this is neglected. Otherwise you need to choose between 'average_couplings' or 'average_couplings_frob' with respectively do the average for each couplings of each model, do the average for each couplings and frobenius of each model. Default='average_couplings'", default="average_couplings", type=str)
    parser.add_argument("-output_name", help="Name for the output file that will containg the couplings. Default=path(model_name)/<type_average>/couplings", default="/", type=str)
    parser.add_argument("-figure", help="Boolean to decide if we want to plot the couplings or not (before and after ising). True or False. by default: False", default=False, action="store_true")
    parser.add_argument("-data_per_col", help="number of possible a.a per column in the MSA (only for couplings_version3). Default=path(model_name)/data_per_col.txt", default="/", type=str)
    parser.add_argument("-model_type", help="type of the model, can be 'linear', 'non-linear' or 'mix'", default="linear", type=str)
    parser.add_argument("-L", help="length of the sequence, if we have not the INFOS file", default=0, type=int)
    parser.add_argument("-K", help="value of K, if we have not the INFOS file", default=0, type=int)
    args=parser.parse_args()
    
    couplings.couplings(args.model_name, args.number_models, args.type_average, args.output_name, args.figure, args.data_per_col, args.model_type, args.L, args.K)