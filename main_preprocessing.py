import argparse
import preprocessing

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_name",help="The input MSA in fasta format or csv.")
    parser.add_argument("data_type",help="Only for csv file (will be ignored if your file is in fasta). Please give the type of data 'full_prot' or 'JD' if you want to consider the data_sequences class. If you don't want to consider the class please write 'full_prot_no_class' or 'JD_no_class'.")
    parser.add_argument("threshold",help="The threshold for the percentage of gaps in a sequence.")
    parser.add_argument("output_name", help="Name for the output file.")
    args=parser.parse_args()
    
    preprocessing.preprocessing(args.input_name,args.data_type, args.threshold, args.output_name)