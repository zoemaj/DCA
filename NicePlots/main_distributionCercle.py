import argparse
import distributionCercle

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("fasta_file",help="File containing the fasta file", type=str)
    parser.add_argument("file_taxonomy",help="File containing the taxonomy of the fasta file", type=str)
    parser.add_argument("-removes", type=str,help="Name of the unclassified and environmental samples organism. Default='unclassified, environmental samples'. If you want nothings please juste write None",default='unclassified, environmental samples')
    parser.add_argument("-seed",help="Seed for the random generator. Default=0",default=0,type=int)
    parser.add_argument("-sub_plots",help="Number of subplots. Default=0",default=0,type=int)
    parser.add_argument("-nb_biggest_organism",help="Number of biggest organism to display. Default=0 -> will display every organisms with a percentage > 5",default=0,type=int)
    parser.add_argument("-dictionary_for_colors",help="Dictionary containing the colors for each level of taxonomy. Default=None -> will take the most recent in DictionaryColors",default=None,type=str)
    parser.add_argument("-output_directory",help="Directory for the output files. Default=...Results/",default=None,type=str)
    parser.add_argument("-output_name",help="Name for the output files. Default=None -> <name_fasta>.png",default=None,type=str)
    args=parser.parse_args()

    distributionCercle.execute(args.fasta_file,args.file_taxonomy,args.removes,args.seed,args.sub_plots,args.nb_biggest_organism,args.dictionary_for_colors,args.output_directory,args.output_name)