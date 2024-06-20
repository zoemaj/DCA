import argparse
import probability_amino

if __name__=="__main__":    
    parser=argparse.ArgumentParser()
    parser.add_argument("model_path",help="The path to the model file.", type=str)
    parser.add_argument("MSA_file",help="The path to the MSA file. (after preprocessing)", type=str)
    parser.add_argument("seed",help="The seed for the random number generator. (used during the model training)", type=int)
    parser.add_argument("K",help="Number values possible per amino acid.", type=int)
    parser.add_argument("-nb_models",help="to complete.", type=int, default=1)
    args=parser.parse_args()
    probability_amino.execute(args.model_path,args.MSA_file,args.seed,args.K,args.nb_models)
    
    