from Bio import SeqIO
import os
from tqdm import tqdm

import random
import matplotlib.pyplot as plt
import numpy as np
import datetime

################################################################################################
def find_indices_header(headers, *args):
    ''' 
    This function will find the indices of the headers in the csv file
    input:
            headers     ->      list, the headers of the csv file
            *args       ->      list of strings, the headers to find
    '''
    indices = {}
    for arg in args:
        if arg in headers:
            indices[arg] = headers.index(arg)
        else:
            print(f"Header {arg} not found in the header of the csv file")
            return None
    return indices
################################################################################################



################################################################################################
def find_OX_in_seq(sequence):
    ''' 
    This function will find the OX in the sequence description
    input:
            sequence    ->      string, the sequence description
    output:
            OX          ->      string, the OX number
    '''
    header_parts = sequence.description.split() #split the fasta header by spaces
    OX=None
    for el in header_parts:
        try:
            OX=el.split('OX=')[1]
            if "-" in OX:
                OX=OX.split('-')[0]
            pass
        except:
            pass
    return OX
################################################################################################

################################################################################################
def Infos_in_uniprot(file_taxonomy):
    ''' 
    This function will read the csv file containing the taxonomy information and return the lines and the indices of the OX and phylum headers
    input:
            file_taxonomy    ->      string, the path to the csv file
    output:
    lines           ->      list, the lines of the csv file
    OX_index        ->      int, the index of the OX header
    phylum_index    ->      int, the index of the phylum header
    '''
    with open(file_taxonomy, "r") as file: 
        lines = file.readlines()
        lines = [line.strip().split(",") for line in lines]
        #consider each line as a list of elements separated by a comma
        headers = lines[0]  #.strip() -> ensures that any leading or trailing whitespace characters (such as spaces, tabs, or newline characters) are removed 
        print(f"Read the tax file {file_taxonomy}...")
        required_headers = ["Organism_name", "OX", "phylum"]
        header_indices = find_indices_header(headers, *required_headers)
        phylum_index = header_indices["phylum"]
        OX_index = header_indices["OX"]
        Organism_name = header_indices["Organism_name"]
        return [lines, OX_index, phylum_index, Organism_name]
################################################################################################


################################################################################################
def find_phylum_and_organism_name(OX,infos):
    ''' 
    This function will find the phylum name corresponding to the OX number
    input:
            lines           ->      list, the lines of the csv file
            OX              ->      string, the OX number
            OX_index        ->      int, the index of the OX header
            phylum_index    ->      int, the index of the phylum header
    output:
            phylum_name     ->      string, the phylum name
    '''
    lines=infos[0]
    OX_index=infos[1]
    phylum_index=infos[2]
    organism_name_index=infos[3]
    line_with_OX = next((line for line in lines[1:] if int(line[OX_index]) == int(OX)), None) #find the line with the same OX number, if not found return None
    if line_with_OX!=None: 
        phylum_name= line_with_OX[int(phylum_index)]
        organism_name=line_with_OX[int(organism_name_index)]
        #remove spaces at the beginning and end of the phylum name
        phylum_name=phylum_name.strip()
        organism_name=organism_name.strip()
    else:
        phylum_name="Other"
        organism_name="Other"
    return phylum_name, organism_name
################################################################################################



################################################################################################
def distinct_phylum(fasta_file,infos,removes_label_type,nb_phylum):

    #read the fasta file
    print(f"Read the fasta file {fasta_file}...")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    organism={}
    for sequence in tqdm(sequences):
        OX=find_OX_in_seq(sequence) #find the OX of the sequence in the fasta file to find the phylum in uniprot (in lines)
        seq=sequence.seq
        phylum_name,organism_name=find_phylum_and_organism_name(OX,infos)
        if phylum_name=="Other":
            continue
        for remove_label_type in removes_label_type:
            if remove_label_type != None:
                remove_label_type=str(remove_label_type).strip().lower()
                if remove_label_type in str(organism_name).lower():
                    continue #if the organism_name contains the remove_label_type, we do not consider it, we go to the next sequence

        #look if it is already a key in the dictionary
        if phylum_name in organism:
            organism[phylum_name]+=[seq]
        else: #if not create it
            organism[phylum_name]=[seq]
    #sorted by the number of sequences in each phylum, max first
    organism={k: v for k, v in sorted(organism.items(), key=lambda item: len(item[1]),reverse=True)}
    #keep only the nb_phylum first phylum
    organism_to_keep={}
    i=0
    for phylum in organism:
        if i<nb_phylum:
            organism_to_keep[phylum]=organism[phylum]
            i+=1
        else:
            break
    print("Phylum to keep: ",list(organism_to_keep.keys()))
    return organism_to_keep
################################################################################################

################################################################################################
def comparison_btw_two_seqs(list_seq1,list_seq2):
    list_percentage=[]
    for seq1 in tqdm(list_seq1):
        for seq2 in list_seq2:
            total_sim=0
            for c1,c2 in zip(seq1,seq2):
                if c1==c2:
                    total_sim+=1
            total_sim=total_sim/len(seq1)*100
            list_percentage.append(total_sim)
    return list_percentage
################################################################################################


################################################################################################
def find_similarities_with_other_phylum(organism_ref,organisms_to_compare):
    seqs=[list(seq.split(',')[0]) for seq in organism_ref]
    phylum_names=list(organisms_to_compare.keys())
    All_list_sim=[]
    print("-----------------------------------")
    print("-----------------------------------")
    for phylum in phylum_names:
        print(f"Similarities with {phylum} in process...")
        list_sim=[]
        seqs_to_compare=[list(seq.split(',')[0]) for seq in organisms_to_compare[phylum]]
        list_sim.append(comparison_btw_two_seqs(seqs,seqs_to_compare))
        #sort the list with min first
        list_sim.sort(reverse=False)
        All_list_sim.append(list_sim)
        print("-----------------------------------")
        print("-----------------------------------")
    return All_list_sim
################################################################################################






################################################################################################
def execute(fasta_file, uniprot_file, nb_phylum=4,remove_label_type="unclassified, environmental samples", output_directory=None, output_name1=None,output_name2=None,dictionary_for_colors=None):
    '''
    This function will separate the sequences in the fasta file according to their taxonomy. It will also remove the sequences that have a label in their organism name that contain remove_label_type.
    The sequences will be saved in a dictionary with the taxonomy as key and the sequences as values.
    '''
    print("-------- WELCOME TO THE SIMILARITY BETWEEN TAXONOMY FUNCTION --------")
    
    fasta_name=os.path.basename(fasta_file)
    figure1_name=f"{fasta_name}-mean_sim_btw_tax.png"
    figure2_name=f"{fasta_name}-distribution_sim_btw_tax.png"

    if remove_label_type!="None":
        remove_label_type=remove_label_type.split(",")
        remove_label_type=[str(el).strip() for el in remove_label_type]
    else:
        remove_label_type=[None]
    
    if output_directory!=None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if output_name1==None:
            output_name1=figure1_name
        if output_name2==None:
            output_name2=figure2_name
        output1=os.path.join(output_directory,output_name1)
        output2=os.path.join(output_directory,output_name2)
    else:
        output1=os.path.join(os.path.dirname(__file__), "Results", figure1_name)
        output2=os.path.join(os.path.dirname(__file__), "Results", figure2_name)

    print("The output figures will be saved in: ", output1," and ",output2)


    

    #create the dictionary with the sequences and organisms names for each phylum_name
    organism=distinct_phylum(fasta_file,Infos_in_uniprot(uniprot_file),remove_label_type,nb_phylum)
    phylum_names=list(organism.keys())

    if dictionary_for_colors==None: #take the  one in DictionaryColors with the most recent date
        directory = os.path.join(os.path.dirname(__file__), "DictionaryColors")
        files = os.listdir(directory)
        if not files:
            print("No file in DictionaryColors, we will create a new one.")
            dictionary_for_colors={}
            for phyl in phylum_names:
                color_potential=(random.random(), random.random(), random.random())
                while color_potential in dictionary_for_colors.values(): #if the color is already used
                    color_potential=(random.random(), random.random(), random.random())
                dictionary_for_colors[str(phyl)]=color_potential #attribute a color for each phylum
        else:
            files = [file.split(".txt")[0] for file in files if file.endswith(".txt")]
            dates = [file.split("_")[1] for file in files] #return for each file the date [2023-05-12, 2024-05-13, ...]
            dates.sort(reverse=True)
            file_name = os.path.join(directory, f"colors_{dates[0]}.txt")
            print("take as colors the one in the file: ", file_name)
            #load it
            dictionary_for_colors = {}
            with open(file_name, "r") as file:
                lines = file.readlines()
                for line in lines:
                    name_phyl,colors_phyl= line.split(":")
                    name_phyl=name_phyl.strip()
                    colors_phyl=colors_phyl.split("(")[1]
                    colors_phyl=colors_phyl.split(")")[0]
                    colors_phyl=colors_phyl.split(",")
                    color_phyl=(float(colors_phyl[0]),float(colors_phyl[1]),float(colors_phyl[2]))
                    if name_phyl in phylum_names:
                        dictionary_for_colors[name_phyl] = color_phyl
    colors=[] #list of colors for each phylum (because if dictionary is given, we need to know exactly which phylum is used)
    for phyl in phylum_names:
        try:
            color=dictionary_for_colors[str(phyl)] #if the phylum exist in the dictionary
            colors.append(color)
        except: #if the phylum is not in the dictionary
            color_potential=(random.random(), random.random(), random.random()) #create a new color
            while color_potential in colors: #if the color is already used
                color_potential=(random.random(), random.random(), random.random())
            colors.append(color_potential)
            dictionary_for_colors[str(phyl)]=color_potential #attribute a color for the new phylum


    SIM=[]
    #plot an histogram of the similarities, with different color for each list in list_similarities
    fig, ax = plt.subplots(nb_phylum, 1, figsize=(10, 8))
    print("Organism:",list(organism.keys()))
    for i,phylum in enumerate(organism):
        print(f"Phylum {phylum} contains {len(organism[phylum])} sequences")
        organisms_to_compare={}
        list_names_to_compare=""
        list_names=[]
        k=0
        for phylum_to_compare in organism:
            if phylum_to_compare!=phylum:
                organisms_to_compare[phylum_to_compare]=organism[phylum_to_compare]
                list_names_to_compare+=str(phylum_to_compare)
                list_names.append(phylum_to_compare)
                k+=1
                if k<nb_phylum-1:
                    list_names_to_compare+=", and "
        

        print(f"Similarities between {phylum} and {list_names_to_compare}")
        print("phylum: ",phylum)
        print("organisms_to_compare: ",list(organisms_to_compare.keys()))
        list_similarities=find_similarities_with_other_phylum(organism[phylum],organisms_to_compare)
        print("list_similarities: ",len(list_similarities))
        #SIM.append(np.mean(list_similarities))
        color_values=[]
        for phylum_comparison in list_names:
            color_values.append(dictionary_for_colors[phylum_comparison])
        for s,sim in enumerate(list_similarities):
            ax[i].hist(sim, bins=40, color=color_values[s], alpha=0.7, label=list_names[s], edgecolor="black")
        ax[i].set_title(f"{phylum}")
        ax[i].set_xlabel(f"Similarity (%)",fontsize=14)
        ax[i].set_ylabel(f"$N_{{seq}}$",fontsize=14)
        ax[i].legend(loc="upper right", fontsize=11)
        ax[i].grid()
    plt.tight_layout()
    plt.savefig(output2)
    
    plt.show()
        

        

    #plot the similarities as bar plot, the axis x corresponds to the phylum and the axis y to the percentage
    #we will have for each phylum nb_phylum-1 bars

################################################################################################




