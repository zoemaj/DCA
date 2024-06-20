
from Bio import SeqIO
import os
from tqdm import tqdm

import random
import matplotlib.pyplot as plt
import numpy as np

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
def plot_differenceSeq(fasta_file,file_taxonomy,seed,output,dictionary_for_colors):
    ''' 
    This function compare each sequence of the fasta file with the first one and compute the similary (character per character)
    The results are ploted in an histogram as percentage of similarity
    input:
            fasta_file      ->      str, the fasta file
            file_taxonomy    ->      str, the file containing the taxonomy of the fasta file
            output          ->      str, the output path to save the plot
    '''
    if dictionary_for_colors==None:
        dictionary_for_colors={}
    ######################## read the file taxonomy ##########################
    random.seed(seed)
    with open(file_taxonomy, "r") as file:     
        lines = file.readlines()
        lines = [line.strip().split(",") for line in lines]
        headers = lines[0]
        print(f"Read the tax file {file_taxonomy}...")
        required_headers = ["OX", "phylum"]
        header_indices = find_indices_header(headers, *required_headers)
        phylum_index = header_indices["phylum"]
        OX_index = header_indices["OX"]
    ##########################################################################

    ########################### read the fasta file ##########################
    #read the fasta file
    print(f"Read the fasta file {fasta_file}...")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    #take the first sequence as the reference
    reference = sequences[0].seq
    similarities = {}
    #for each sequence in the fasta file
    for sequence in tqdm(sequences[1:]):
        OX=find_OX_in_seq(sequence)
        line_OX=next((line for line in lines[1:] if int(line[OX_index]) == int(OX)), None)
        if line_OX!=None:
            phylum=line_OX[int(phylum_index)]
            phylum=phylum.strip()
        else:
            print("OX doesn't correspond")
            phylum='Other'
        #compute the difference with the reference
        sim_i=sum([1 for i in range(len(reference)) if reference[i] == sequence.seq[i]])/len(reference)*100
        #add to the list of similarity for the phylum
        try:
            similarities[phylum].append(sim_i)
        except:
            
            similarities[phylum]=[sim_i]
    ##########################################################################

    
    ############ decide which phylum to keep ###############################
    print("the different phylum in the fasta and their percentage are:")
    phylum_names_to_show=list(similarities.keys())
    #sort the phylum names by length of similarities[phylum]
    phylum_names_to_show.sort(key=lambda x: len(similarities[x]), reverse=False) #reverse=True to have the biggest first
    #tot is the total number of similiarities [phylum]
    tot=sum([len(similarities[phylum]) for phylum in phylum_names_to_show])
    for phylum in phylum_names_to_show:
        print(f"{phylum}: {len(similarities[phylum])/tot:.2f}%")
    R=input('Do you want to keep all the phylum? (yes/no)')
    while  R!='yes' and R!='no':
        R=input('Please write yes or no:')
    if R=='no':
        R4=input("Please enter the elements you want to keep (separate by a comma and no space, no need to take 'Other' since it will be considered)")
        R4=R4.split(",")
        while set(R4).issubset(phylum_names_to_show)==False:  #if the elements are not in the list
            for element in R4:
                if element not in phylum_names_to_show:
                    print("the type ", element, " is not in the list.")
            R4=input("Please enter valid elements ")
            R4=R4.split(",")
        phylum_names=R4

        
    #######################################################################
    #change the keys of the similarities dictionary to keep only the phylum_names and 'Other' if it is not in the phylum_names
    similarities_new={phylum: similarities[phylum] for phylum in phylum_names}
    if 'Other' not in phylum_names:
        similarities_new['Other']=[sim for phylum in phylum_names_to_show if phylum not in phylum_names for sim in similarities[phylum]]
    phylum_names.append('Other')
    similarities=similarities_new

    ########################### compute the colors ##########################
    ########################### for each phylum ##############################
    if dictionary_for_colors=={}: #if no dictionary is given
        for phylum in phylum_names:
            phylum=phylum.strip()
            color_potential=(random.random(), random.random(), random.random())
            while color_potential in dictionary_for_colors.values(): #if the color is already used
                color_potential=(random.random(), random.random(), random.random())
            dictionary_for_colors[phylum]=color_potential #attribute a color for each phylum
    colors=[] #list of colors for each phylum (because if dictionary is given, we need to know exactly which phylum is used)
    for phylum in phylum_names:
        try:
            color=dictionary_for_colors[phylum] #if the phylum exist in the dictionary
            colors.append(color)
        except: #if the phylum is not in the dictionary
            color_potential=(random.random(), random.random(), random.random()) #create a new color
            while color_potential in colors: #if the color is already used
                color_potential=(random.random(), random.random(), random.random())
            colors.append(color_potential)
            dictionary_for_colors[phylum]=color_potential #attribute a color for the new phylum
    ##########################################################################
    
    ############################ plot the histogram ##########################
    n_sequences=len(sequences)
    n_bins=20
    #need to do an histogram per color and plot them together on the same graph
    _, ax = plt.subplots(figsize=(6, 6))
    min_sim=100.0
    max_sim=0.0
    similarities_i=[] #list composed of all the list for the different phylum
    
    for color,phyl_name in zip(colors,phylum_names):
        list_sim=similarities[phyl_name]
        ###################### find the min_sim and max_sim ######################
        if min_sim>min(list_sim):
            min_sim=min(list_sim)
        if max_sim<max(list_sim):
            max_sim=max(list_sim)
        ##########################################################################
        ax.hist(list_sim, bins=n_bins, color=color, alpha=0.7, label=phyl_name)
    plt.xlabel('Percentage of similarity', fontsize=16)
    plt.ylabel('Number of sequences', fontsize=16)
    plt.xlim(min_sim,max_sim)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #put the legend at the left up of the graph
    plt.legend(loc='upper left', fontsize=12)
    plt.grid()
    #add one box text with two text: the max and min difference, with white background and black border
    #if max_sim=100 and min_sim=0 don't put the .2f
    if max_sim==100.0 and min_sim==0.0:
        text = f"Max similarity: {int(max_sim)}%\nMin similarity: {(min_sim)}%"
    else:
        text = f"Max similarity: {max_sim:.2f}%\nMin similarity: {min_sim:.2f}%"
    plt.text(0.1, 0.45, text, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'), transform=plt.gca().transAxes,fontsize=14)
    #plt.title(f'total sequences: {len(sequences)}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output)
    plt.show()
################################################################################################

def execute(fasta_file,file_taxonomy,seed=24,output_directory=None,output_name=None,dictionary_for_colors=None):
    print("---------- WELCOME TO THE SimWithFirst ----------")

    fasta_name=fasta_file.split('/')[-1]
    figure_name=fasta_name.split('.')[0]+"_SimWithFirst.png"
    if output_directory!=None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if output_name==None:
            output_name=figure_name
        output=os.path.join(output_directory,output_name)
    else:
        output=os.path.join(os.path.dirname(__file__), "Results", figure_name)
    print("The output figure will be saved in: ", output)

    print(" ---------- Find the dictionary for colors .... ----------")
    if dictionary_for_colors==None: #take the  one in DictionaryColors with the most recent date
        directory = os.path.join(os.path.dirname(__file__), "DictionaryColors")
        files = os.listdir(directory)
        if not files:
            print("No file in DictionaryColors, we will create a new one.")
            dictionary_for_colors={}
        else:
            files = [file.split(".txt")[0] for file in files if file.endswith(".txt")]
            dates = [file.split("_")[1] for file in files] #return for each file the date [2023-05-12, 2024-05-13, ...]
            dates.sort(reverse=True)
            file_name = os.path.join(directory, f"colors_{dates[0]}.txt")
            print("take as colors the one in the file: ", file_name)
            #load it
            dictionary_for_colors={}
            with open(file_name, "r") as file:
                lines = file.readlines()
                for line in lines:
                    key, value = line.strip().split(": ")
                    key=key.strip()
                    color=value.split("(")[1]
                    color=color.split(")")[0]
                    color=color.split(",")
                    color=(float(color[0]),float(color[1]),float(color[2]))
                    dictionary_for_colors[key] = color
    
    print(" -------------- Plot the histogram .... ------------------")
    plot_differenceSeq(fasta_file,file_taxonomy,seed,output,dictionary_for_colors)
    
