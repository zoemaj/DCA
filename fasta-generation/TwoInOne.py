
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

def create_dictionary_of_operon_id(line,OLN_index,ORF_index):
    
    dictionary_of_ids={}
    ox_found=False
    if line[OLN_index]!="None":
        ox_found=True
        OLNs=line[OLN_index].split(" ")
        for OLN in OLNs:
            if "/" in OLN:
                #check if there is a "/" because: "If two predicted genes have been merged to form a new gene, both OLNs are indicated, separated by a slash"
                two_OLN=OLN.split("/")
                id,operon_id=find_sequential_ordering_gene(two_OLN[0])
                if dictionary_of_ids.get(id)==None:
                    dictionary_of_ids[id]=[operon_id]
                else:
                    dictionary_of_ids[id].append(operon_id)
                
                id,operon_id=find_sequential_ordering_gene(two_OLN[1])
                if dictionary_of_ids.get(id)==None:
                    dictionary_of_ids[id]=[operon_id]
                else:
                    dictionary_of_ids[id].append(operon_id)
            else:
                id,operon_id=find_sequential_ordering_gene(OLN)
                if dictionary_of_ids.get(id)==None:
                    dictionary_of_ids[id]=[operon_id]
                else:
                    dictionary_of_ids[id].append(operon_id)

    if line[ORF_index]!="None":
        ox_found=True
        ORFs=line[ORF_index].split(" ")
        for ORF in ORFs:
            if "/" in ORF:
                two_ORF=ORF.split("/")
                id,operon_id=find_sequential_ordering_gene(two_ORF[0])
                if dictionary_of_ids.get(id)==None:
                    dictionary_of_ids[id]=[operon_id]
                else:
                    dictionary_of_ids[id].append(operon_id)
                id,operon_id=find_sequential_ordering_gene(two_ORF[1])
                if dictionary_of_ids.get(id)==None:
                    dictionary_of_ids[id]=[operon_id]
                else:
                    dictionary_of_ids[id].append(operon_id)
            else:
                id,operon_id=find_sequential_ordering_gene(ORF)
                if dictionary_of_ids.get(id)==None:
                    dictionary_of_ids[id]=[operon_id]
                else:
                    dictionary_of_ids[id].append(operon_id)
    return dictionary_of_ids,ox_found

def find_indices_header(headers, *args):
    indices = {}
    for arg in args:
        if arg in headers:
            indices[arg] = headers.index(arg)
        else:
            print(f"Header {arg} not found in the header of the csv file")
            return None
    return indices

def sort_sequences(fasta_file,file_name=None):
    ''' 
    This function takes a fasta file and returns a dictionary with the organism as key and the sequences as values
    input: fasta_file -> fasta file with the homologues sequences of different organisms
    output: organism_sequences -> dictionary with the organism as key and the sequences as values
    '''
    #################################################################################################
    ##################### extraction of the fasta file ##############################################
    #################################################################################################
    #print the numbers of sequences in fasta file
    print(f"Number of sequences: {len(list(SeqIO.parse(fasta_file, 'fasta')))}")
    organism_sequences=defaultdict(list) #defaultdict is a dictionary that has a default value for keys that were not added yet
    #################################################################################################

    #################################################################################################
    ############# if we have acces to the fasta file -> usefull for operons consideration ###########
    #################################################################################################
    if file_name!=None:
        with open(file_name, "r") as file:     
            lines = file.readlines()
            lines = [line.strip().split(",") for line in lines]
            #consider each line as a list of elements separated by a comma

            headers = lines[0]  #.strip() -> ensures that any leading or trailing whitespace characters (such as spaces, tabs, or newline characters) are removed 
            print(f"Read the tax file {file_name}...")
            print("The headers of the csv file are (for the tax information): ", headers)
            required_headers = ["OLN", "Organism_name","ORF", "Strain", "OX"]
            header_indices = find_indices_header(headers, *required_headers)
            OLN_index = header_indices["OLN"]
            ORF_index = header_indices["ORF"]
            organism_id = header_indices["Organism_name"]
            Strain_index = header_indices["Strain"]
            OX_index = header_indices["OX"]

            print("Removing the lines with 'unclassified' or 'Unclassified' in the organism name...")
            print("Number sequence before removing: ", len(lines))
            #remove each line that have "unclassified" or "Unclassified" in organism_name:
            lines = [line for line in lines if "unclassified" not in line[organism_id].lower()]
            print("Number sequence after removing: ", len(lines))
    #################################################################################################

    #################################################################################################
    ######################## extraction of the attribute of the sequences ###########################
    #################################################################################################
    for sequence in tqdm(SeqIO.parse(fasta_file, "fasta")):
        header_parts = sequence.description.split() #split the fasta header by spaces
        # example of headers: >lcl|Query_317909 tr|A0A494C039|A0A494C039_HUMAN Hypoxia up-regulated protein 1 OS=Homo sapiens OX=9606 GN=HYOU1 PE=1 SV=1
        #but also: >sp|P11021| BIP_HUMAN Endoplasmic reticulum chaperone BiP OS=Homo sapiens OX=9606 GN=HSPA5 PE=1 SV=2

        
        # try to Extract organism from the second term after splitting by '_' (in the example it will take HUMAN)
        #because some time the '_' is in the header_parts[0]
        try:
            organism_name = header_parts[1].split('_')[1]  # Extract organism from the second term after splitting by '_' (in the example it will take HUMAN)
            pass
        except:
            organism_name = header_parts[0].split('_')[1]
            pass
        try: 
            number_name=header_parts[1].split('|')[1]
            pass
        except:
            number_name=header_parts[0].split('|')[1]
            pass
        
        organism_name = organism_name.upper()
        ##############################################################################################
        #keep only the sequence of the letters after the header parts
        sequence_prot= sequence.seq
        #################### extraction of the OX number and the OS name ############################
        #extract the OX number:
        OX=None
        for el in header_parts:
            try:
                OX=el.split('OX=')[1]
                pass
            except:
                pass
        OS=None
        try:
            OS=sequence.description.split('OS=')[1]
            OS=OS.split('OX=')[0]
            pass
        except:
            pass
        if OS==None:
            R=input('No OS identification, it will not be possible to preprocess this file with taxonomy in the future. Do you still want to continue? (yes/no)')
            while  R!='yes' and R!='no':
                R=input('Please write yes or no:')
            if R=='no':
                return
        ##############################################################################################

        #############################################################################################
        ################# EXTRACTION of the ORL, ORF and Strain #####################################
        #############################################################################################

        if file_name!=None:
            ox_found=False
            #print("OX: ",OX)
            line_with_OX = next((line for line in lines[1:] if int(line[OX_index]) == int(OX)), None) #find the line with the same OX number, if not found return None
            #print("Line with OX: ",line_with_OX)
            if line_with_OX != None:
                dictionary_of_ids,ox_found=create_dictionary_of_operon_id(line_with_OX,OLN_index,ORF_index)
                id=line_with_OX[organism_id]
            #else:
            #    print(f"No line with the specified OX {OX} found. Sequence removed")
            if ox_found==True:
                organism_sequences[id].append([number_name,OX,OS,organism_name,sequence_prot,dictionary_of_ids])   
        #############################################################################################
        #############################################################################################     
        else:
            id=OX #should correspond to the organism_id
            organism_sequences[id].append([number_name,OX,OS,organism_name,sequence_prot])
    print(f"Number of different organism: {len(organism_sequences.keys())}")
    return organism_sequences

def find_sequential_ordering_gene(chain):
    id=""
    numbers=[]
    #transform the chain string as a list of charcaters
    chain=list(chain)
    last_digit=0
    find_digit=False
    for i in range(len(chain)-1,0,-1):
         #if the character is a digit but not a "_"
        if chain[i].isnumeric():
            numbers.append(chain[i])
            find_digit=True
        elif find_digit==True:
            break
        last_digit=i
    #reverse the list
    numbers=numbers[::-1]
    while numbers[0]=="0": #be sure to have no 0 at the beginning
        numbers=numbers[1:]
    #CONVERT the list into a string
    numbers="".join(numbers)
    #Convert into a integer
    numbers=int(numbers)
    #find the identifiant exemple: SAMEA2683035_02658 -> id: SAMEA2683035, number: 02658
    #other exemple: JW2594 -> id: JW, number: 2594
    
    id=chain[:last_digit] 
    
    if "_" in id:
        id=id[:id.index("_")]
    #convert the list into a string
    id="".join(id)
    #print(f"(id,numbers): ({id},{numbers})")


    return id,numbers

def TwoInOne(fastafile_one,fastafile_two):

    ''' 
    This function will combine two fasta files be combining the sequence of the same organism
    '''
    name_one=fastafile_one.split("/")[-1].split(".")[0]
    name_two=fastafile_two.split("/")[-1].split(".")[0]
    overwrite="yes"

    print(f"Do you want to combine same organism from proteins according to the operons? (yes/no)")
    while True:
        operons=input("Please enter yes or no: ")
        if operons=="yes" or operons=="no":
            break

    ###################################################################################################
    ############################ case of operons consideration ########################################
    ###################################################################################################
    if operons=="yes":
        how_many_keeped=0
        distanceMax=0
        print("Strategy by pair: select a certain number of pairs per organism (with lower operons distance) ")
        print("Strategy with distanceMax: keep only the pairs with an operon distance smaller than a distance max")
        strategy=input(f" Please select the strategy (pair/distanceMax) ")
        while strategy!="pair" and strategy!="distanceMax":
            strategy=input("Please enter pair or distanceMax: ")
        if strategy=="pair":
            how_many_keeped=input(f"Please enter the number (int) of maximal pairs that you want to keep per organism: ")
            while not how_many_keeped.isnumeric():
                how_many_keeped=input("Please enter a number: ")
            how_many_keeped=int(how_many_keeped)
        else:
            distanceMax=input(f"Please enter the maximal distance (int) that you want to keep per organism: ")
            while not distanceMax.isnumeric():
                distanceMax=input("Please enter a number: ")
            distanceMax=int(distanceMax)
        
        
        good_names=[]
        good_name=input(f"Please confirme that the name of your first proteins is {name_one} (yes/no): ")
        while good_name!="yes" and good_name!="no":
            good_name=input("Please enter yes or no: ")
        if good_name=="no":
            name_one=input("Please enter the name of the first protein: ") #to find its tax file
        good_names.append(name_one)
        good_name=input(f"Please confirme that the name of your second proteins is {name_two} (yes/no): ")
        while good_name!="yes" and good_name!="no":
            good_name=input("Please enter yes or no: ")
        if good_name=="no":
            name_two=input("Please enter the name of the second protein: ") #to find its tax file
        good_names.append(name_two)

        ############################### extractions of the tax file #####################################
        file_name_list=[]
        for R in good_names:
            file_name="/"
            if os.path.exists('uniprot-tax/'+R+"-tax.csv"):
                file_name='uniprot-tax/'+R+"-tax.csv"
            if file_name!="/":
                print("The file ",file_name," is in the folder uniprot-tax.")
                Found=input("Do you want to use this file? (yes/no) ")
                while Found!="yes" and Found!="no":
                    Found= input("Please write yes or no")
            else:
                print("No file with the protein name "+R+" in folder the uniprot-tax been found...")
            if Found=="no" or file_name=="/":
                file_name=input("Please write the path for the csv file that you are looking for the protein "+R)
            file_name_list.append(file_name)
        #################################################################################################

    ###################################################################################################
    ###################################################################################################

    if operons=="no":
        organism_one=sort_sequences(fastafile_one) #dictionary with keys organism_id=OX and values: [number_name,OX,OS,organism_name,sequence_prot]
        organism_two=sort_sequences(fastafile_two)
        
    else: #we will extract the first sequence of the two proteins to have them as pair at the beginning of the new file
        organism_one=sort_sequences(fastafile_one,file_name_list[0]) 
        for key in organism_one.keys():
            first_seq_one=organism_one[key][0]
            break
        organism_two=sort_sequences(fastafile_two,file_name_list[1]) #dictionary with keys organism_id and values: [number_name,OX,OS,organism_name,sequence_prot,dictionary_of_ids]
        for key in organism_two.keys():
            first_seq_two=organism_two[key][0]
            break
    ###################################################################################################
    ########################## PREPARATION OF THE COMBINATION  ########################################
    ################### (list of common organisms and the name of the new file) #######################
    ###################################################################################################
    #check is they are similar keys name between organism_one and organism_two:
    common_keys = filter(lambda x: x in organism_one, organism_two)
    common_keys=list(common_keys)
    #check that common_keys is not empty:
    if len(common_keys)==0:
        print("No common keys found")
        return
    
    #ask the name of the two proteins
    file_name_end= input("Please enter the name of the two proteins (without extension) separated by a dash: ")
    path= input("Please enter the directory path where you want to save the new fasta file: ")
    if operons=="no":
        file_info= path + '/' + file_name_end + "-infos.txt"
        file_name = path + '/' +file_name_end + ".fasta"
        path_figure=path+"/"+file_name_end
    if operons=="yes" and strategy=="pair":
        file_info= path + "/" +file_name_end +"-"+ str(how_many_keeped)+ "pairs-infos.txt"
        file_name = path + "/" +file_name_end +"-"+ str(how_many_keeped)+ "pairs.fasta"
        path_figure=path+"/"+file_name_end+"-"+ str(how_many_keeped)+"pairs"
    elif operons=="yes" and strategy=="distanceMax":
        file_info= path + "/"+file_name_end + "-"+str(distanceMax)+"distance-infos.txt"
        file_name = path + "/" +file_name_end + "-"+str(distanceMax)+"distance.fasta"
        path_figure=path+"/"+file_name_end+"-"+str(distanceMax)+"distance"
    
    #if file name already exist ask if we want to overwrite it
    if os.path.exists(file_name):
        print(f"The file {file_name} already exist")
        overwrite = input("Do you want to overwrite the file? (yes/no) ")
        while overwrite!="yes" and overwrite!="no":
            overwrite = input("Please enter yes or no ")
        if overwrite=="yes":
            print(f"Overwriting the file {file_name}")
        else:
            print("The file will not be overwritten")
            new_path_R=input("Do you want to save the new fasta file in a new directory? (yes/no) ")
            while new_path_R!="yes" and new_path_R!="no":
                new_path_R = input("Please enter yes or no ")
            if new_path_R=="yes":
                new_path= input("Please enter the directory path where you want to save the new fasta file (without extension): ")
                file_info= new_path+"/"+file_name_end + "-infos.txt"
                file_name = new_path+"/"+file_name_end + ".fasta"
                path_figure=new_path+"/"+file_name_end
            else:
                return

    #create a file_info containing the numbers of different organisms per protein
    #check before if it already exits. If yes ask if we want to overwrite it
    if os.path.exists(file_info):
        print(f"The file {file_info} already exist")
        overwrite = input("Do you want to overwrite the file? (yes/no) ")
        while overwrite!="yes" and overwrite!="no":
            overwrite = input("Please enter yes or no ")
        if overwrite=="yes":
            print(f"Overwriting the file {file_info}")
            with open(file_info, "w") as file:
                file.write(f"Number of different organism in the first fasta file: {len(organism_one.keys())}\n")
                file.write(f"Number of different organism in the second fasta file: {len(organism_two.keys())}\n")
                file.write(f"Number of different organism in the new fasta file: {len(common_keys)}\n")

            pass
        else:
            print("The file will not be overwritten")
            return
    else:
        with open(file_info, "w") as file:
            file.write(f"Number of different organism in the first fasta file: {len(organism_one.keys())}\n")
            file.write(f"Number of different organism in the second fasta file: {len(organism_two.keys())}\n")
            file.write(f"Number of different organism in the new fasta file: {len(common_keys)}\n")
    ###################################################################################################


            
    ###################################################################################################
    ############################## COMBINATION OF THE SEQUENCES #######################################
    ###################################################################################################
    n_pairs_per_organism={}
    with open(file_name, "w") as output_handle: #write the sequence of the two files combined only for the common keys
        #write the two first sequences
        if operons=="yes":
            output_handle.write(f">sp {first_seq_one[0]}-{first_seq_two[0]}_{first_seq_one[1]} OS={first_seq_one[2]}-{first_seq_two[2]} OX={first_seq_one[1]}-{first_seq_two[1]}\n")
            output_handle.write(str(first_seq_one[4]) + str(first_seq_two[4])+ "\n")
        for key in tqdm(common_keys): #dico: key:organism_id -> lists of [number_name,OX,OS,organism_name,sequence_prot,dictionary_of_ids]
            #take the same number of sequences for the two proteins
            n=min(len(organism_one[key]),len(organism_two[key]))
            
            ###########################################################################################
            ############################# CASE OF OPERONS CONSIDERATION ###############################
            ###########################################################################################
            if operons=="yes":
                if distanceMax==0:
                    random_number=1.e9
                else:
                    random_number=distanceMax
                
                #min_n_distance is a dictionary of maximum n list with [distance,position i, position j,strand], the key are 1,2,3,...,n
                min_n_distance={}
                for k in range(n): #start with big values
                    min_n_distance[k]=[random_number,random_number,random_number,"None"] #where are stock the minimal distance
                    #the second argument corresponds to the ith element of organism_one[key][i]
                    #the third argument corresponds to the jth element of organism_two[key][j]
                
                #lists with the different strand for the two proteins:
                operons_id_one=[]
                operons_id_two=[]
                #find the organisms that have the closest distanced, extract all the dictionary_of_ids for the two proteins with key=organism_id
                for i in range(len(organism_one[key])):
                    #for example: {'b': [2614], 'JW': [2594]} -> we want 'b' and 'JW'
                    for key_id in organism_one[key][i][5].keys():
                        operons_id_one.append(key_id)
                for i in range(len(organism_two[key])):
                   for key_id in organism_two[key][i][5].keys():
                        operons_id_two.append(key_id)
                    
                #find the common strand between organism_one and organism_two (and organism name == key)
                common_strand = filter(lambda x: x in operons_id_one, operons_id_two)
                common_strand=list(common_strand)

                for strand in common_strand:    #find the minimals distance possible
                    for i in range(len(organism_one[key])):
                        if organism_one[key][i][5].get(strand)!=None: #if the strand is the key
                            list_one=organism_one[key][i][5][strand] #list of operon_id for this strand and this organism
                            for j in range(len(organism_two[key])):
                                min_n_distance_ij=[]
                                if organism_two[key][j][5].get(strand)!=None: #if the strand is the key
                                    list_two=organism_two[key][j][5][strand] #list of operon_id for this strand and this organism
                                    #extract the minimal distance possible between
                                    for operon_position in list_one:
                                        for operon_position_two in list_two:
                                            min_n_distance_ij.append(abs(operon_position-operon_position_two))
                                    #sort the list of distances with the smallest first
                                    min_n_distance_ij=sorted(min_n_distance_ij)
                                    
                                    #keep the min distance that are smallest than in min_n_distance (for loop on the keys)
                                    for k in range(n):
                                        # check if there are values in min_n_distance_ij that are smaller than min_n_distance[k]
                                        if min_n_distance_ij[0]<min_n_distance[k][0]:
                                            #decalage des valeurs de min_n_distance 
                                            for l in range(n-1,k,-1): #from n-1 to k+1
                                                min_n_distance[l]=min_n_distance[l-1]
                                            #remplace min_n_distance[k][0] by min_n_distance_ij[0]
                                            min_n_distance[k]=[min_n_distance_ij[0],i,j,strand]
                                            #remove the min_n_distance_ij[0] from the list
                                            min_n_distance_ij.remove(min_n_distance_ij[0])  #to look for the next smallest values
                                            if len(min_n_distance_ij)==0:
                                                break
                
                for el in range(len(min_n_distance)):
                    
                    if int(min_n_distance[el][0])==int(random_number): #this means than all the others t0o
                        #remove it
                        #and delete the next ones
                        for l in range(el,n):
                            del min_n_distance[l]
                        break



                    
                #check if min_n_distane is not empty:
                if len(min_n_distance)>0:
                    #print(f"Organism name: {key} with {n} pairs of sequences, the minimal distance are: {min_n_distance}")
                    
                    
                    seqs=[]
                    #extract the i and the j from min_n_distance 
                    if how_many_keeped>0:
                        max_k=min(len(min_n_distance),how_many_keeped)
                    else:
                        max_k=len(min_n_distance)
                    #print("max_k=",max_k)
                    for k in range(max_k): #take only the best pairings
                        i=min_n_distance[k][1]
                        j=min_n_distance[k][2]
                        seq=str(organism_one[key][i][4]) + str(organism_two[key][j][4])+ "\n"
                        if seq not in seqs: #to be sure to not have similar sequences
                            seqs.append(seq)
                            output_handle.write(f">sp {organism_one[key][i][0]}-{organism_two[key][j][0]}_{organism_one[key][i][3]} organism={key} OS={organism_one[key][i][2]}-{organism_two[key][j][2]} OX={organism_one[key][i][1]}-{organism_two[key][j][1]}\n")
                            #take the seq of the protein, second argument of the list
                            output_handle.write(seq)
                        else:
                            #remove the pair (i,j):
                            #min_n_distance[0],...,min_n_distance[k-1],min_n_distance[k],min_n_distance[k+1],...,min_n_distance[n] becomes min_n_distance[0],...,min_n_distance[k-1],min_n_distance[k+1],...,min_n_distance[n]
                            
                            del min_n_distance[k]
                            #print("keys of min_n_distance after removing k: ",min_n_distance.keys())
                            #if how_many_keeped>0:
                            #    max_k=min(len(min_n_distance),how_many_keeped) #to take into account the fact that we remove one element
                            #else:
                            #    max_k=len(min_n_distance)
                            #k-=1 #to take into account the fact that we remove one element
                    n_pairs_per_organism[key]=[n,min_n_distance]
                    #print("n_pairs_per_organism: ",n_pairs_per_organism)
                                          
            else:  
                for i in range(n): 
                    if organism_one[key][i][2]!=organism_two[key][i][2]:
                        print(f"Organism name (OS) are different for the same OX {key} number: {organism_one[key][i][2]} and {organism_two[key][i][2]}")
                        OS=organism_one[key][i][2]+"-"+organism_two[key][i][2]
                    else:
                        n_pairs_per_organism[key]=[n]
                        OS=organism_one[key][i][2]
                    output_handle.write(f">sp {organism_one[key][i][0]}-{organism_two[key][i][0]}_{organism_one[key][i][1]} OS={OS} OX={key}\n") #should be the same for the two proteins
                    #take the seq of the protein, second argument of the list
                    output_handle.write(str(organism_one[key][i][3]) + str(organism_two[key][i][3])+ "\n")
    
    #---------------–--------------------------------------------------#
    organisms=[key for key in n_pairs_per_organism.keys()]
    
    #n_pairs_per_organism=dictionary with as key, the organisms names, and as values the lists [n_tot,n_keeped,min_n_distance]
    n_tot=[n_pairs_per_organism[key][0] for key in organisms]
    id=np.argsort(n_tot) #sort the list of n_tot, id is the list of the index of the sorted list
    id=list(id)

    #keep only n_organisms with values >5
    #id=[i for i in id if n_organisms[i]>5]
    organisms=[organisms[i] for i in id] 
    n_tot=[n_tot[i] for i in id]

    if operons=="yes":
        min_n_distance=[n_pairs_per_organism[key][1] for key in organisms]    
        min_n_distance=[min_n_distance[i] for i in id] #min_n_distance is a list of n_pairs_per_organism[key][1] for key in organisms -> we have [distance1,distance2,...,distance_n]

    #save in a text file the number of pairs of sequences per organism
    if overwrite=="yes":
        with open(file_info, "a") as file:
            file.write("Number of pairs of sequences per organism: \n")
            for i in range(len(organisms)):
                #print("Organism: ",organisms[i])
                #print("min_n_distance: ",min_n_distance[i])
                file.write(f"{organisms[i]}: {n_tot[i]}\n")
                file.write("----------------------------------------- \n")
                if operons=="yes":
                    file.write("Minimal distances between the operons: \n")
                    keys_min_n_distance=min_n_distance[i].keys()
                    for k in keys_min_n_distance:
                        #print("min_n_distance[i][k][3]: ",min_n_distance[i][k][3])
                        
                        file.write(f"strand {str(min_n_distance[i][k][3])}: ")
                        file.write(f"{min_n_distance[i][k][0]}, ")
                    file.write("\n")
                file.write("----------------------------------------- \n")

    path_figure_1=path_figure+"_distribution_pairs_organism.png"
    plt.figure()
    #histogram with axis x the number of pairs of sequences and axis y the number of organisms with this number of pairs
    plt.hist(n_tot,bins=50)
    plt.xlabel("Number of pairs of sequences")
    plt.ylabel("Number of organisms")
    plt.xlim(0,max(n_tot)+1)
    plt.grid()
    plt.savefig(path_figure_1)
    plt.show()
    plt.close()

    if operons=="yes":
        #find the list min_n_distance[i] with the bigger len
        if how_many_keeped>0:
            #plot the distribution of the distance
            min_n_distance_k={}
            organisms_k={}
            for i in range(len(organisms)):
                keys_min_n_distance=min_n_distance[i].keys()
                #keep only the max_k first keys
                keys_min_n_distance=[k for k in keys_min_n_distance if k<how_many_keeped]
                for k in keys_min_n_distance:
                    #sometimes min_n_distance[i][n] is empty if n>n_tot[i]
                    #we need to remove it from the list
                    #min_n_distance_n=[min_n_distance[i][n][0] for i in range(len(organisms)) if len(min_n_distance[i])>=n]
                    try: #not sure if min_n_distance[i][k] exist
                        #min_n_distance_k.append(int(min_n_distance[i][k][0]))
                        #organisms_k.append(organisms[i])
                        min_n_distance_k[k]=min_n_distance_k.get(k,[])+[int(min_n_distance[i][k][0])] #list of the distance for the kth best pairing
                        organisms_k[k]=organisms_k.get(k,[])+[organisms[i]]
                    except:
                        pass
            for k in min_n_distance_k.keys():
                if min_n_distance_k[k]!=[]:
                    path_figure_k=path_figure+"_distance_"+str(k+1)+".png"
                    plt.figure()
                    plt.title(f"Distance distribution for the {k+1} best pairing")
                    plt.hist(min_n_distance_k[k],bins=50)
                    plt.xlabel("Distance")
                    plt.ylabel("Number of organisms")
                    plt.grid()
                    plt.savefig(path_figure_k)
                    plt.show()
                    plt.close()
        else:
            min_n_distance_k=[]
            organisms_k=[]
            for i in range(len(organisms)):
                keys_min_n_distance=min_n_distance[i].keys()
                for k in keys_min_n_distance:
                    #print("min_n_distance[i][k][0]: ",min_n_distance[i][k][0])
                    if int(min_n_distance[i][k][0])<int(distanceMax):
                        min_n_distance_k.append(int(min_n_distance[i][k][0]))
                        organisms_k.append(organisms[i])
            if min_n_distance_k!=[]:
                path_figure_k=path_figure+"_distance.png"
                plt.figure()
                plt.title(f"Distance distribution for the best pairing with distance smaller than {distanceMax}")
                plt.hist(min_n_distance_k,bins=50)
                plt.xlabel("Distance")
                plt.ylabel("Number of organisms")
                plt.grid()
                plt.savefig(path_figure_k)
                plt.show()
                plt.close()



    print(f"New fasta file saved in {file_name}")
    print(f"Info file saved in {file_info}")
    print(f"Figure saved in {path_figure}")
   

