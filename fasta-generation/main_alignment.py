import argparse
import alignment

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("seq_base",help="The sequence in fasta format to use as reference")
    parser.add_argument("file",help="The file in fasta or stockholm format to adjust in function of the reference seq_base")
    args=parser.parse_args()
    alignment.transformation(args.seq_base,args.file)