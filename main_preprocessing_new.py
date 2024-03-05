import argparse
import preprocessing_new

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_name",help="The input MSA in fasta format or csv.")
    parser.add_argument("threshold",help="The threshold for the percentage of gaps in a sequence.")
    parser.add_argument("output_name", help="Name for the output file.")
    args=parser.parse_args()
    
    preprocessing_new.preprocessing(args.input_name, args.threshold, args.output_name)