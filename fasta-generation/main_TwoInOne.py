import argparse
import TwoInOne

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("file1",help="The first file fasta to combine")
    parser.add_argument("file2",help="The second file fasta to combine")
    args=parser.parse_args()
    TwoInOne.TwoInOne(args.file1,args.file2)