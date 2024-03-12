import argparse
import write_list

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("path_file",help="The path where to find the list from uniprot. Can be in xls or tsv format.")
    args=parser.parse_args()
    
    write_list.execute(args.path_file)
