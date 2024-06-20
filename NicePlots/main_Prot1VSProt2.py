import argparse
import Prot1VSProt2

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("fasta_file_1",help="File containing the fasta file for the first protein", type=str)
    parser.add_argument("fasta_file_2",help="File containing the fasta file for the second protein", type=str)
    parser.add_argument("uniprot_file_1",help="File containing the uniprot file for the first protein", type=str)
    parser.add_argument("uniprot_file_2",help="File containing the uniprot file for the second protein", type=str)
    parser.add_argument("-remove_label_type",type=str,help="Name of the unclassified and environmental samples organism. Default='unclassified, environmental samples'. If you want nothings please juste write None",default='unclassified, environmental samples')
    parser.add_argument("-output_directory",help="Directory for the output files. Default=...Results/",default=None,type=str)
    parser.add_argument("-output_name",help="Name for the output files. Default=None -> <name_fasta>.png",default=None,type=str)
    parser.add_argument("-dictionary_for_colors",help="Dictionary containing the colors for each level of taxonomy. Default=None -> will take the most recent in DictionaryColors",default=None,type=str)
    args=parser.parse_args()
    Prot1VSProt2.execute(args.fasta_file_1,args.fasta_file_2,args.uniprot_file_1,args.uniprot_file_2,args.remove_label_type,args.output_directory,args.output_name,args.dictionary_for_colors)