import argparse
import weights

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_name",help="Thie input MSA preprocessed using the file execute_preprocessing.", type=str)
    parser.add_argument("-output_name", help="Name for the output file. Default=path(input_name)/weights-(threshold)/weights-(threshold).txt", Default="",type=str)
    parser.add_argument("-threshold", help="The percentage of simulutude accepted. Default=0.8, we keep all the sequences", default=0.8, type=float)
    args=parser.parse_args()
    
    weights.weights(args.input_name, args.output_name, args.threshold)