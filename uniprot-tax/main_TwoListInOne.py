import argparse
import TwoListInOne

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("path_file_1",help="The path where to find the list 1")
    parser.add_argument("path_file_2",help="The path where to find the list 2")
    args=parser.parse_args()
    
    TwoListInOne.execute(args.path_file_1,args.path_file_2)