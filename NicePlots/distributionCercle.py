
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
def removing_type_organism(lines,organism_id,type_to_remove):
    ''' 
    This function will remove the lines with the type_to_remove in the organism name
    input:
            lines       -> list of lists, each list is a line of the csv file
            organism_id -> int, the index of the column with the organism name
            type_to_remove -> string, the type of organism to remove (in lower case)
            
    output:
            lines       -> list of lists, each list is a line of the csv file,
                        but the lines with the type_to_remove in the organism name are removed
    '''

    print(f"Removing the lines with {type_to_remove} in the organism name...")
    print("Number sequence before removing: ", len(lines))
    lines=[line for line in lines if type_to_remove not in line[organism_id].lower()]
    print("Number sequence after removing: ", len(lines))
    return lines
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

def population_organism(fasta_file,file_taxonomy,removes=['unclassified','environmental samples']):
    '''  
    This function will count the number of sequences for each organism in the fasta file
    and return a dictionary with the organism name as key and the number of sequences as value
    input:
            fasta_file      ->      string, the path to the fasta file
            file_taxonomy   ->      string, the path to the taxonomy file
            remove          ->      string, the type of organism to remove, if None, no organism is removed
    output:
            dictionary_organism     ->      dictionary, the keys are the organism names and the values are the number of sequences
    '''
    dictionary_organism={} 
    ################################ extract the lines  #####################################
    ############################### from the taxonomy file ##################################
    ########################### and remove undefined organisms ##############################
    with open(file_taxonomy, "r") as file: 
        lines = file.readlines()
        lines = [line.strip().split(",") for line in lines]
        #consider each line as a list of elements separated by a comma
        headers = lines[0]  #.strip() -> ensures that any leading or trailing whitespace characters (such as spaces, tabs, or newline characters) are removed 
        print(f"Read the tax file {file_taxonomy}...")
        required_headers = ["Organism_name", "Strain", "OX", "phylum"]
        header_indices = find_indices_header(headers, *required_headers)
        organism_id = header_indices["Organism_name"]
        Strain_index = header_indices["Strain"]
        phylum_index = header_indices["phylum"]
        OX_index = header_indices["OX"]
        for remove in removes:
            if remove!=None:
                lines=removing_type_organism(lines,organism_id,remove)
    #########################################################################################


    ############################## find the info from uniprot for ###########################
    ############################# each sequence of the fasta file ###########################
    ############################### + creation of the dictionary ############################
    for sequence in tqdm(SeqIO.parse(fasta_file, "fasta")):
        
        OX=find_OX_in_seq(sequence) #find the OX of the sequence in the fasta file
        #find the line in the file with taxonomy corresponding to the OX number
        line_with_OX = next((line for line in lines[1:] if int(line[OX_index]) == int(OX)), None) #find the line with the same OX number, if not found return None
        if line_with_OX!=None: #if None we will just not take it
            #we need to look if the dictionary already contains the organism name or not. If not, we add it, if yes, we add 1 to the number of sequences.
            #the DICTIONARY dictionary_organism:
            #       -> the keys are the phylum names
            #       -> the values are a list of dictionary with the organism name as key and the number of sequences as value
            phylum_name= line_with_OX[int(phylum_index)] #get the phylum name
            phylum_name=phylum_name.strip()
            organism_name=line_with_OX[int(organism_id)] #get the organism name
            if phylum_name in dictionary_organism: #if the phylum name is already in the dictionary
                keys_organism=[] #list of organism name in the phylum
                #dictionary_organism[phylum_name] contains a list of dictionary. Each dictionary has the organism name as key and the number of sequences as value
                for organism_list in dictionary_organism[phylum_name]: #look the different organism list in the phylum
                    key=organism_list.keys() #take the organism name for each list
                    keys_organism.append(list(key)[0]) #add the organism name to the list
                if organism_name not in keys_organism:  #it is the first time that we have this kind of organism for this type of phylum!
                    organism_name_dict={} #creation of the dictionary with the organism name as key and the number of sequences as value
                    organism_name_dict[organism_name]=1 #first sequence
                    dictionary_organism[phylum_name].append(organism_name_dict) #add this dictionary to the list of dictionary for this phylum
                else: #the organism name is already in the dictionary
                    dictionary_organism[phylum_name][keys_organism.index(organism_name)][organism_name]+=1 #increase the number of sequences for this organism
            else: #we have never seen this phylum before, we need to create the key and the dictionary for the organism corresponding to this phylum
                organism_name_dict={}
                organism_name_dict[organism_name]=1 #dictionary of the organism
                dictionary_organism[phylum_name]=[] #new phylum key
                dictionary_organism[phylum_name].append(organism_name_dict)
    #########################################################################################
    print("Number of organism: ", len(dictionary_organism.keys()))

    return dictionary_organism
################################################################################################


################################################################################################
def PLOT_different_phylum_in_cercle(dictionary_organism,seed,sub_plots,nb_biggest_organism,dictionary_for_colors,output):
    '''  

    This function plot the proportion of sequences for each phylum in a cercle
    Only the biggest proportion are labeled (and the percentage is written)
    input:
            dictionary_organism     ->      dictionary, the keys are the phylum names and the values are a list of dictionary
                                            each dictionary has the organism name as key and the number of sequences as value
            seed                    ->      int, the seed for the random function
            sub_plots               ->      boolean, if True, plot the proportion of sequences for the 3 biggest phylum
            nb_biggest_organism     ->      int, the number of biggest organism to label, if 0, all organism with a percentage bigger than perc_limit% are labeled
            dictionary_for_colors   ->      dictionary, the keys are the phylum names and the values are the color to use for the phylum
    output:
            dictionary_for_colors   ->      the one given with new colors if new categories or a new one with the color for each phylum (if not given)
    '''

    random.seed(seed)
    ################################ look for dictionary #####################################
    if dictionary_for_colors==None:
        dictionary_for_colors={}
    ##########################################################################################
    
    ############################### compute the proportion of ################################
    ############################### sequences for each phylum ################################
    nb_seq_tot_per_phyl=[] #will be a list of size len(dictionary_organism.keys())
    #with each position: the number of sequences for each phylum defined in dictionary_organism
    for phylum in dictionary_organism.keys():
        phylum=phylum.strip()
        #phylum is a list of dictionary, each dictionary is an organism with key OS and value the number of OS
        nb_seq_per_org = []
        for organism in dictionary_organism[phylum]: #organism is a dictionary   
            nb_seq = organism[list(organism.keys())[0]] #get the number of sequences (position 0)
            nb_seq_per_org.append(nb_seq) #add the number of sequences for the organism
        nb_seq_tot_per_phyl.append(sum(nb_seq_per_org)) #add the number of sequences for the phylum
    #convert the numbers into proportions
    nb_seq_tot_per_phyl_prop = nb_seq_tot_per_phyl / np.sum(nb_seq_tot_per_phyl) * 100
    ############################################################################################


    ################################ extraction of the colors ##################################
    ################################## for disctinct phylum ####################################
    labels = list(dictionary_organism.keys()) #name of the different phylum
    colors=[] #list of colors for each phylum (because if dictionary is given, we need to know exactly which phylum is used)
    
    if dictionary_for_colors=={}: #if no dictionary is given
        for label in labels:
            label=label.strip()
            color_potential=(random.random(), random.random(), random.random())
            while color_potential in dictionary_for_colors.values(): #if the color is already used
                color_potential=(random.random(), random.random(), random.random())
            dictionary_for_colors[str(label)]=color_potential #attribute a color for each phylum
            colors.append(color_potential)
    
    else:
        for label in labels:
            label=label.strip()
            if label in dictionary_for_colors.keys():
                colors.append(dictionary_for_colors[label])
            else:
                color_potential=(random.random(), random.random(), random.random())
                while color_potential in dictionary_for_colors.values():
                    color_potential=(random.random(), random.random(), random.random())
                dictionary_for_colors[label]=color_potential
                colors.append(color_potential)

    ############################# extraction of the biggest phylum #############################
    ################################## and prepare the labels ##################################
    names_biggest_percentage=[]
    if nb_biggest_organism>0: #take the nb_biggest_organism biggest percentage
        biggest_percentage = sorted(nb_seq_tot_per_phyl_prop, reverse=True)[:nb_biggest_organism] #sort the percentage and take the nb_biggest_organism biggest
        id_biggest_percentage=[i for i in range(len(nb_seq_tot_per_phyl_prop)) if nb_seq_tot_per_phyl_prop[i] in biggest_percentage] #get the index of these phylum
        [names_biggest_percentage.append(labels[i]) for i in id_biggest_percentage] #get the name of the phylum and add it to the list
    else: #take all percentage bigger than 5%
        perc_limit=5
        biggest_percentage = [el for el in nb_seq_tot_per_phyl_prop if el>perc_limit] #take all percentage bigger than perc_limit
        id_biggest_percentage=[i for i in range(len(nb_seq_tot_per_phyl_prop)) if nb_seq_tot_per_phyl_prop[i] in biggest_percentage] #get the index of these phylum
        [names_biggest_percentage.append(labels[i]) for i in id_biggest_percentage] #get the name of the phylum and add it to the list
    labels_masked = [label if nb_seq_tot_per_phyl_prop[i] in biggest_percentage else '' for i, label in enumerate(labels)] #keep only the name of the biggest phylum
    ############################################################################################


    ############################## function to write the percentage #############################
    ################################ (only for the biggest phylum) ##############################
    def my_autopct(pct):
        #only write the percentage if it belongs to the list biggest_percentage
        #return ('%1.1f%%' % pct) if pct in biggest_percentage else ''
        pct_to_print=''
        if nb_biggest_organism>0: #if we want to take the nb_biggest_organism biggest percentage
            for i in range(len(biggest_percentage)):
                if pct>=biggest_percentage[i]:
                    pct_to_print='%1.1f%%' % pct
        else: #if we want to take all percentage bigger than perc_limit
            for i in range(len(biggest_percentage)):
                if pct>=perc_limit:
                    pct_to_print='%1.1f%%' % pct
        return pct_to_print
    ############################################################################################
    

    ###################################### plot the figure #####################################
    _, ax = plt.subplots(figsize=(10, 6))
    ax.axis('equal') 
    #use ax.pie to have the wanted plot 
    _, _, pcts =ax.pie(nb_seq_tot_per_phyl_prop, labels=labels_masked, colors=colors, autopct=my_autopct, textprops={'fontsize': 17})
    #have white background for pcts:
    for pct in pcts:
        pct.set_bbox({'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'white'})
    #add the title
    plt.title(f'{np.sum(nb_seq_tot_per_phyl)} sequences', fontsize=18)
    plt.savefig(output)
    plt.show()
    
    ############################################################################################

    ###################################### sub_plots ############################################
    if sub_plots>0:
        for phylum_max in names_biggest_percentage[0:sub_plots]:
            labels=[]
            nb_seq_per_org = []
            for organism in dictionary_organism[phylum_max]: #organism_list is a list of dictionary
                nb_seq_per_org.append(organism[list(organism.keys())[0]])
                labels.append(list(organism.keys())[0])
            #keep only the percentage and labels 
            perc_limit=1
            nb_seq_per_org_prop = np.array(nb_seq_per_org) / np.sum(nb_seq_per_org) * 100
            biggest_percentage_phyl = [el for el in nb_seq_per_org_prop if el>perc_limit]
            print("biggest_percentage_phyl: ", biggest_percentage_phyl)
            id_biggest_percentage_phyl=[i for i in range(len(nb_seq_per_org_prop)) if nb_seq_per_org_prop[i] in biggest_percentage_phyl]
            print(id_biggest_percentage_phyl)
            labels_masked = [label if nb_seq_per_org_prop[i] in biggest_percentage_phyl else '' for i, label in enumerate(labels)]
            colors=[]
            for i in range(len(labels)):
                colors.append((random.random(), random.random(), random.random()))
            def my_autopct_phyl(pct):
                #only write the percentage if it belongs to the list biggest_percentage
                #return ('%1.1f%%' % pct) if pct in biggest_percentage else ''
                pct_to_print=''
                for i in range(len(biggest_percentage_phyl)):
                    if pct>=perc_limit:
                        pct_to_print='%1.1f%%' % pct
                return pct_to_print
            #shuffle the index to have not the ones in id_biggest_percentage_phyl next to each other
            total_id=len(labels)
            id_to_put=[]
            for i in range(0,total_id):
                if i not in id_biggest_percentage_phyl:
                    id_to_put.append(i)
            total_biggest=len(id_biggest_percentage_phyl)
            separation=total_id//total_biggest
            new_id=[id_biggest_percentage_phyl[0]]
            k=1
            for i in id_to_put:
                if len(new_id)%separation==0 and k<total_biggest:
                    new_id.append(id_biggest_percentage_phyl[k])
                    k+=1
                new_id.append(i)
            labels_masked=[labels_masked[i] for i in new_id]
            nb_seq_per_org_prop=[nb_seq_per_org_prop[i] for i in new_id]
            colors=[colors[i] for i in new_id]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('equal')
            patches, texts, pcts =ax.pie(nb_seq_per_org_prop, labels=labels_masked, colors=colors, autopct=my_autopct_phyl)
            #have white background for pcts
            for pct in pcts:
                pct.set_bbox({'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'white'})
            plt.title(f'{np.sum(nb_seq_per_org)} sequences in {phylum_max}')
            plt.savefig(output.split('.png')[0]+"-"+phylum_max+".png")
            plt.show()
    ############################################################################################


    ###################################### save the dictionary #################################
    directory = os.path.join(os.path.dirname(__file__), "DictionaryColors")
    dateToday = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = os.path.join(directory, f"colors_{dateToday}.txt")
    with open(file_name, "w") as file:
        for key, value in dictionary_for_colors.items():
            file.write(f"{key}: {value}\n")
    print("The colors for each phylum are saved in the file: ", file_name)
    ################################################################################################

    return

################################################################################################

def execute(fasta_file,file_taxonomy,removes='unclassified, environmental samples',seed=0,sub_plots=0,nb_biggest_organism=0,dictionary_for_colors=None,output_directory=None,output_name=None):
    print("---------- WELCOME TO THE DISTRIBUTION CERCLE ----------")
    fasta_name=fasta_file.split('/')[-1]
    figure_name=fasta_name.split('.')[0]+".png"
    if output_directory!=None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if output_name==None:
            output_name=figure_name
        output=os.path.join(output_directory,output_name)
    else:
        output=os.path.join(os.path.dirname(__file__), "Results", figure_name)
    print("The output figure will be saved in: ", output)
    if sub_plots>0:
        output_2=output.split('.png')[0]
        print(f"And the {str(sub_plots)} sub plots will be save in {output_2}-<phyl>.png with <phyl> the phyl name.")

    print("---- preparation of the dictionary for the plot... -----")
    #take a list of removes and remove space at the begining and end of elements
    if removes=="None":
        remove=[None]
    else:
        remove=removes.split(',')
        remove=[el.strip() for el in remove]

        
    
    dictionary_organism=population_organism(fasta_file,file_taxonomy,remove)
    print("---- dictionary prepared, let's plot the cercle... -----")
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
    PLOT_different_phylum_in_cercle(dictionary_organism,seed,sub_plots,nb_biggest_organism,dictionary_for_colors,output)

            
    




