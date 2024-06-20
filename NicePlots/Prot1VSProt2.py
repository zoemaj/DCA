
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
        phylum_name="Unknown"
        organism_name="Unknown"
    return phylum_name, organism_name
################################################################################################



################################################################################################
def numbers_of_distinct_organisms(fasta_file,infos,remove_label_types):
    #read the fasta file
    print(f"Read the fasta file {fasta_file}...")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    organism={}
    for sequence in tqdm(sequences):
        OX=find_OX_in_seq(sequence) #find the OX of the sequence in the fasta file to find the phylum in uniprot (in lines)
        phylum_name,organism_name=find_phylum_and_organism_name(OX,infos)
        remove_in_organism=False
        for remove_label_type in remove_label_types:
            if remove_label_type != None:
                remove_label_type=remove_label_type.strip().lower()
                
                if remove_label_type in organism_name.lower():
                    remove_in_organism=True
                    continue #if the organism_name contains the remove_label_type, we do not consider it, we go to the next sequence
        if remove_in_organism:
            continue
        #look if it is already a key in the dictionary
        if organism_name in organism:
            organism[organism_name][0]+=1
        else: #if not create it
            organism[organism_name]=[1,phylum_name]
    #organise the dictionary by the number of sequences
    organism = dict(sorted(organism.items(), key=lambda item: item[1], reverse=False))
    return organism
################################################################################################




def execute(fasta_file_1,fasta_file_2,uniprot_1,uniprot_2,remove_label_type="unclassified, environemental samples",output_directory=None,output_name=None,dictionary_for_colors=None):
    print("-------- WELCOME TO THE PROTEIN 1 VS PROTEIN 2 ----------")
    fasta_name_1=fasta_file_1.split('/')[-1]
    fasta_name_2=fasta_file_2.split('/')[-1]
    figure_name=fasta_name_1.split('.')[0]+"-"+fasta_name_2.split('.')[0]+".png"
    if output_directory!=None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if output_name==None:
            output_name=figure_name
        output=os.path.join(output_directory,output_name)
    else:
        output=os.path.join(os.path.dirname(__file__), "Results", figure_name)
    print("The output figure will be saved in: ", output)


    if remove_label_type=="None":
        remove_label_type=[None]
        print("No label type to remove")
    else:
        remove_label_type=remove_label_type.split(",")
        remove_label_type=[el.strip() for el in remove_label_type]
        print("Remove the following label types: ", remove_label_type)

    organism1=numbers_of_distinct_organisms(fasta_file_1,Infos_in_uniprot(uniprot_1),remove_label_type)
    organism2=numbers_of_distinct_organisms(fasta_file_2,Infos_in_uniprot(uniprot_2),remove_label_type)
    print(f"Number of distinct organisms in the first file: {len(organism1)}")
    print(f"Number of distinct organisms in the second file: {len(organism2)}")
    
    #Figure with the different organism1 on axis x and organism2 on axis y
    #a different color of dots for each organism, the graduation of the axis correspond to the numbers for the organisms
    list_organism1=list(organism1.keys())
    list_organism2=list(organism2.keys())

    #labels is all the organisms in list_organism1 and list_organism2 (no repetition)
    labels=list(set(list_organism1+list_organism2))
    #make a list of pairs for same organism, if one organism is not in the list, we put 0
    numbers=[]
    max_numbers=0
    
    for label in labels:
        try:
            x=int(organism1[label][0])
            z=organism1[label][1]
        except:
            x=0
        try:
            y=int(organism2[label][0])
            z=organism1[label][1]
        except:
            y=0

        max_of_x_y=max(x,y)
        if max_of_x_y>max_numbers:
            max_numbers=max_of_x_y
        #look if there are already the exact same pair x,y -> if yes "add" a little bit to x or y to avoid superposition on plots
        step=0.05
        x_step=1
        y_step=0
        n=8
        while (x,y,z) in numbers or n<8:
            numbers[numbers.index((x,y,z))]=(x+step*x_step,y+step*y_step,z)
            (x,y,z)=(x-step*x_step,y-step*y_step,z)
            #permute x_step and y_step
            x_step,y_step=y_step,x_step
            n+=1
        numbers.append((x,y,z))
    #compute the proportion of each unique z in numbers
    Unique_z=list(set([el[2] for el in numbers])) #list of unique phylum names
    proportion_z=[sum([1 for el in numbers if el[2]==z])/len(numbers) for z in Unique_z]
    perc_limite=0.07
    #only display the phylum z with a proportion > perc_limite and keep the percentage in percentage_phylum
    #labels_to_display=[Unique_z[i] for i in range(len(Unique_z)) if proportion_z[i]>perc_limite]
    labels_to_display=[Unique_z[i] for i in range(len(Unique_z)) if proportion_z[i]>perc_limite]
    percentage_phylum=[round(proportion_z[i]*100,2) for i in range(len(Unique_z)) if proportion_z[i]>perc_limite]
    print("The phylum to display are: ", labels_to_display)
    print("The percentage of each phylum to display are: ", percentage_phylum)

    
    if dictionary_for_colors==None: #take the  one in DictionaryColors with the most recent date
        directory = os.path.join(os.path.dirname(__file__), "DictionaryColors")
        files = os.listdir(directory)
        if not files:
            print("No file in DictionaryColors, we will create a new one.")
            dictionary_for_colors={}
            for phyl in Unique_z:
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
                    if name_phyl in Unique_z:
                        dictionary_for_colors[name_phyl] = color_phyl
    colors=[] #list of colors for each phylum (because if dictionary is given, we need to know exactly which phylum is used)
    for phyl in Unique_z:
        try:
            color=dictionary_for_colors[str(phyl)] #if the phylum exist in the dictionary
            colors.append(color)
        except: #if the phylum is not in the dictionary
            color_potential=(random.random(), random.random(), random.random()) #create a new color
            while color_potential in colors: #if the color is already used
                color_potential=(random.random(), random.random(), random.random())
            colors.append(color_potential)
            dictionary_for_colors[str(phyl)]=color_potential #attribute a color for the new phylum

    #plot the figure
    fig, ax = plt.subplots((1), figsize=(8,8))
    #provisoir #
    #fig, ax = plt.subplots((1), figsize=(4,8))
    ##########
    already_displayed=[] #list of z already displayed
    nb_max_diff_xy=4
    list_max_diff=[np.zeros(4) for i in range(nb_max_diff_xy)]
    list_max_numbers=[np.zeros(4) for i in range(nb_max_diff_xy)]
    max_numbers=0
    for i in range(len(labels)):
        x=numbers[i][0]
        y=numbers[i][1]
        if x>max_numbers:
            max_numbers=x
        if y>max_numbers:
            max_numbers=y
        z=numbers[i][2]
        if z in labels_to_display and z not in already_displayed: #cross and not circles
            label_z=f"{z} ({percentage_phylum[labels_to_display.index(z)]}%)"
            ax.scatter(x, y, c=[colors[Unique_z.index(z)]],label=label_z,marker="x",linewidths=1.0)
            already_displayed.append(z)
        else:
            ax.scatter(x, y, c=[colors[Unique_z.index(z)]],marker="x",linewidths=1.0)
        diff=abs(x-y)
        if diff>list_max_diff[-1][0]:
            list_max_diff[-1]=[diff,x,y,labels[i]]
            list_max_diff.sort(key=lambda x:x[0],reverse=True)
        max_btw_xy=max(x,y)
        if max_btw_xy>list_max_numbers[-1][0]:
            list_max_numbers[-1]=[max_btw_xy,x,y,labels[i]]
            list_max_numbers.sort(key=lambda x:x[0],reverse=True)

    #list_annotation is a the list of list_max_diff and list_max_numbers (without repetition)
    list_annotation=list_max_diff
    for el in list_max_numbers:
        if el not in list_annotation:
            list_annotation.append(el)

    #add a text with the name of the organism just above the point (only for the ones in max_diff)
    for el in list_annotation:
        x=el[1]
        y=el[2]
        label=el[3]
        ##provisoir##
        #if x>110 or y>110:
        #    continue
        #############
        ax.text(x,y+0.1,label,fontsize=10)
    #plot a diagonal line going from (0-5,0-5) to (max_numbers+5,max_numbers+5)
    ax.plot([0,max_numbers+2],[0,max_numbers+2],c="black",linestyle="--")
    plt.xlabel(f"Number of sequences \n in {fasta_name_1.split('.')[0]} per organism",fontsize=14)
    plt.ylabel(f"Number of sequences \n in {fasta_name_2.split('.')[0]} per organism",fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.xlim(0,max_numbers+2)
    plt.ylim(0,max_numbers+2)
    ##provisoir##
    #plt.xlim(0,110)
    #plt.ylim(0,110)
    #xticks = np.arange(0, 110,10)
    #yticks = np.arange(0, 110,10)
    #############
    xticks = np.arange(0, max_numbers+2,50)
    yticks = np.arange(0, max_numbers+2,50)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid()
    
    #add the legend centering top and outside the plot, only one column for fig 4,6
    plt.legend(fontsize=13,loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=1) #0.5 fixed to the center
    plt.tight_layout()
    plt.savefig(output)
    plt.show()

    