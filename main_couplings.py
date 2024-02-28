import argparse
import couplings

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    #parser.add_argument("triplets_name",help="File containing the triplets of the NN obtained using model.py.")
    parser.add_argument("model_name", help="Name of the file where the trained model was saved")
    parser.add_argument("model_type", help="type of the model, can be 'linear', 'non-linear' or 'mix'")
    parser.add_argument("L", help="length of the sequences in the MSA")
    parser.add_argument("K", help="number of categories in the MSA")
    parser.add_argument("data_per_col", help="number of possible a.a per column in the MSA (only for couplings_version3)")
    parser.add_argument("number_models", help="number of model to train")
    parser.add_argument("type_average", help= "if number_models this is neglected. Otherwise you need to choose between 'average_couplings' or 'average_couplings_frob' with respectively do the average for each couplings of each model, do the average for each couplings and frobenius of each model")
    parser.add_argument("output_name", help="Name for the output file that will containg the couplings.")
    args=parser.parse_args()
    
    couplings.couplings(args.model_name, args.model_type, args.L, args.K, args.data_per_col, args.number_models, args.type_average, args.output_name)