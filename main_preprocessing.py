import argparse
import preprocessing

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_name",help="The input MSA in fasta format or csv.", type=str)
    parser.add_argument("-output_name", help="Name for the output file. Default=path(input_name)/preprocessing-(threshold)gaps/preprocessed-(threshold)gaps.csv", default="",type=str)
    parser.add_argument("-threshold",help="The threshold for the percentage of gaps in a sequence. Default=1.0", default=1.0, type=float)
    args=parser.parse_args()
    
    preprocessing.preprocessing(args.input_name, args.output_name, args.threshold)