import argparse
import SimBtwTax

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("fasta_file",help="File containing the fasta file", type=str)
    parser.add_argument("file_taxonomy",help="File containing the taxonomy of the fasta file", type=str)
    parser.add_argument("-nb_phylum", help="Number of phylum to compare (will take the ones with maximum of sequences). Default=4",default=4,type=int)
    parser.add_argument("-remove_label_type",type=str,help="Name of the unclassified and environmental samples organism. Default='unclassified, environmental samples'. If you want nothings please juste write None",default='unclassified, environmental samples')
    parser.add_argument("-output_directory",help="Directory for the output files. Default=...Results/",default=None,type=str)
    parser.add_argument("-output_name1",help="Name for the output figure for the mean. Default=None -> <name_fasta>.png",default=None,type=str)
    parser.add_argument("-output_name2",help="Name for the output figure for the mean. Default=None -> <name_fasta>.png",default=None,type=str)
    parser.add_argument("-dictionary_for_colors",help="Dictionary containing the colors for each level of taxonomy. Default=None -> will take the most recent in DictionaryColors",default=None,type=str)
    args=parser.parse_args()
    SimBtwTax.execute(args.fasta_file,args.file_taxonomy,args.nb_phylum,args.remove_label_type,args.output_directory,args.output_name1,args.output_name2,args.dictionary_for_colors)
