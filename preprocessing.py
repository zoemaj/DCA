import numpy as np
import pandas as pd
from Bio import SeqIO
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def find_the_tax(file_name):
    """
    This function will find the different tax in the file and ask to the user which tax he wants to use.
    It will return a dictionary with the node and the taxonomy elements to keep
    input:  file_name       ->      name of the csv file containing the taxonomy of the sequences (usually in the uniprot-tax folder)
                                    string
    output: E_dic           ->      dictionary with the node as key and the taxonomy as value
                                    dictionary
            unique_class    ->      list of the different taxonomy elements to keep
                                    list
    """
    print('-------------------------------------------------------')
    print('-------------- ', file_name, ' ----------------')
    print('-------------------------------------------------------')
    with open(file_name, "r") as file:
                lines = file.readlines()
                #determine the number of different tax
                diff_tax=lines[0].split(",") #the first element correpond to the node #for example gives['superkingdom', 'phylum\n']
                last_name=diff_tax[-1].split("\n")[0]
                diff_tax_new=diff_tax[:-1]
                diff_tax_new.append(last_name)

                start_id=2 #remove the print of the node and organism_name
                if "OLN" in diff_tax_new:
                    start_id=3
                if "ORF" in diff_tax_new:
                    start_id=4
                if "Strain" in diff_tax_new:
                    start_id=5
                print("The different tax are: ", diff_tax_new[start_id:]) #remove the print of the node
                R2=input("Which tax do you want to use?")
                while R2 not in diff_tax_new:
                    R2=input("Please write a valid tax present in the list ", diff_tax_new, ": ")
                #create a dictionary with the node and the division or the kingdom
                E_dic = {}
                for line in lines[start_id:]:
                    list_tax = line.split(", ")
                    print("list_tax: ", list_tax)
                    node = list_tax[0]
                    print("OX: ", node)
                    tax = list_tax[diff_tax_new.index(R2)]
                    if " " in tax:
                        tax= tax.split(" ")[0]
                    if "\n" in tax:
                        tax= tax.split("\n")[0]
                    E_dic[node] = tax
                print("The unique elements of the tax ", R2, " and their proportions in the file from uniprot, are the following: ")
                #print firstly the unique element with lower proportion
                unique_class=sorted(set(E_dic.values()), key=lambda x: list(E_dic.values()).count(x))
                
                for tax in unique_class:
                    print(tax, ":", list(E_dic.values()).count(tax)/len(E_dic)*100, "%")
                #ask to the user a list of element that he wants to keep
                R3=input("Do you want to keep all the elements? (yes/no) ")
                while R3!='yes' and R3!='no':
                    R3=input("Please write yes or no ")
                if R3=='no':
                    R4=input("Please enter the elements you want to keep (separate by a comma and no space) ")
                    R4=R4.split(",")
                    while set(R4).issubset(unique_class)==False: 
                        for element in R4:
                            if element not in unique_class:
                                print("the type ", element, " is not in the ", R2, " list.")
                        R4=input("Please enter valid elements ")
                        
                        R4=R4.split(",")
                    unique_class=R4
    return E_dic, unique_class

def filter_data_with_sim(fasta_file,min_sim=0.0,max_sim=1.0):
    ''' compare the sequences with the first one and remove the sequences with a similarity less than min_sim and more than max_sim'''
    ''' will create a new fasta file with the filtered sequences '''
    print("------- Filtering the data by keeping only the sequences with a similarity between ", min_sim, " and ", max_sim, " --------")
    directory = os.path.dirname(fasta_file)
    print("The directory is ", directory)
    fasta_file_filtered = directory + "/filtered_min_sim_" + str(min_sim) + "_max_sim_" + str(max_sim) + "/" + os.path.basename(fasta_file)
    if not os.path.exists(os.path.dirname(fasta_file_filtered)):
        os.makedirs(os.path.dirname(fasta_file_filtered))
    print("The filtered fasta file will be saved in ", fasta_file_filtered)
    #read the fasta file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    #take the first sequence as the reference
    reference = sequences[0].seq
    with open(fasta_file_filtered, "w") as file:
        file.write(">"+sequences[0].description+"\n")
        file.write(str(sequences[0].seq)+"\n")
        for sequence in sequences[1:]:
            #compute the similarity
            similarity = sum([1 for i in range(len(reference)) if reference[i] == sequence.seq[i]])/len(reference)
            if similarity > min_sim and similarity < max_sim:
                #write the sequence in the new file
                file.write(">"+sequence.description+"\n")
                file.write(str(sequence.seq)+"\n")

    return fasta_file_filtered






def preprocessing(input_name, output_name='', threshold=1.0,min_sim=0.0,max_sim=1.0) :
    """
    input_name: name of the file containing the MSA in fasta format or csv
    output_name: name for the output file, default is the same path than the input file with the name preprocessing-(threshold)gaps/preprocessed-(threshold)gaps.csv
    threshold: the threshold for the percentage of gaps in a sequence, default is 1.0
    """
    if min_sim!=0.0 and max_sim!=1.0:
        input_name=filter_data_with_sim(input_name, min_sim, max_sim)
            
    if output_name == '':
        path_folder=input_name.split("/")[:-1] #take the path of the input file
        path_folder="/".join(path_folder)
        output_name=path_folder+"/preprocessing-"+str(threshold)+"gaps/preprocessed-"+str(threshold)+"gaps.csv"
        good_output_name=input(f"Do you want to save the output file will in  {output_name} ? (yes/no) ")
        while good_output_name!='yes' and good_output_name!='no':
            good_output_name=input("Please write yes or no ")
        if good_output_name=='no':
            output_name=input("Please write the path for the output file ")
    else:
        path_folder=output_name.split("/")[:-1] #take the path of the input file
        path_folder="/".join(path_folder)


    K=21
    #load the sequences
    if input_name.endswith('.fasta'):
        
        MSA = list(SeqIO.parse(input_name, "fasta"))
        #transform the characters in upper case
        for sequence in MSA:
            sequence.seq = sequence.seq.upper()
        print("------- Filtering the data by keeping only the sequences with less than ", threshold, " gaps --------")
        print("number sequences before filtering:", len(MSA))  
        #ask if the user want to know the taxonomy of the sequences for the training
        A=input("Do you want to know the taxonomy of the sequences for the training? (yes/no) ")
        #accept only yes or no
        while A!='yes' and A!='no':
            A=input("Please write yes or no ")
        if A=='no':
            infos_name=path_folder+"/INFOS_no_tax.txt"
            print(" You will need to take K=21 in the model parameters file")
            MSA = filter_data(MSA,'fasta',float(threshold),tax=False)
            print("number sequences after filtering:", len(MSA))
            print("------- encode the MSA into numbers --------")
            MSA=amino_acids_to_numbers(MSA, type_file='fasta',tax=False)
        if A=='yes':
            infos_name=path_folder+"/INFOS_with_tax.txt"
            number=input("How man different proteins type do you have in the MSA? For example if it is the homologous of two proteins together write 2.")
            number=int(number)
            list_file_prot=input("Please can you write on which protein are you working (BiP, DnaK,..)? separated by a comma and no space. ")
            list_file_prot=list_file_prot.split(",")
            print("The list of proteins you are working on is: ", list_file_prot)
            while len(list_file_prot)!=number:
                input("invalid number of proteins")
                input("Please write the ", number, " proteins you are working on separated by a comma. ")
            #look in the folder uniprot-tax if there is a file with the name of the protein with at the end -tax.csv:
            #(the protein can be written in lower case or upper case or both
            file_name_list=[]
            for R in list_file_prot:
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
            MSA = filter_data(MSA,'fasta',float(threshold),tax=True)
            big_E_dic={}
            big_unique_class=[]
            for file_name in file_name_list:
                #find the different tax in the file and ask to the user which tax he wants to use
                E_dic, unique_class=find_the_tax(file_name)
                #add the dictionary to the bip_E_dic and to the existing key if it already exist
                big_E_dic.update(E_dic)
                #add the unique_class to the existing list but only if it is not already in the list
                big_unique_class.extend([x for x in unique_class if x not in big_unique_class])
                #print("The big unique class is: ", big_unique_class)
            if number>1:
                print("The unique elements keeped for the files are: ", big_unique_class)
            newMSA = []
            for sequence in tqdm(MSA):
                #extrat the Description
                #take only the part that start with OX
                if "-" in sequence[0]: #case when two proteins
                    OX1=sequence[0].split("-")[0]
                    OX2=sequence[0].split("-")[1]
                    if OX1==OX2:
                        OX=OX1
                        OX=str(OX)
                        print("OX: ", OX)
                    else:
                        print("problem")
                        print("OX1: ", OX1)
                        print("OX2: ", OX2)
                        return
                else:
                    OX=sequence[0]
                
                #find the node in the dictionary
                if OX not in big_E_dic:
                    tax='Other'
                    print(f"{OX} not in big_E_dic")
                else:
                    name=str(big_E_dic.get(OX)).split()[0]
                    #print("name: ", name)
                    if name in big_unique_class:
                        tax=name
                    else:
                        tax='Other'
                newMSA.append(sequence[1] + ", "+ tax)
            MSA=newMSA
            #find the number of different tax (last term of each sequence)
            big_unique_tax = set([value.split(',')[1] for value in MSA])
            print("unique_tax: ", big_unique_tax)
            K=len(big_unique_tax)+K
            print("You will need to take K=", K, " in the model parameters file")
            #save an histogram of the different tax in the MSA
            plt.hist([value.split(',')[1] for value in MSA], bins=len(big_unique_tax))
            plt.title('Histogram of the different tax in the MSA')
            plt.xlabel('Tax')
            plt.ylabel('Number of sequences')
            #save it in the same path than output_name 
            #plt.savefig(output_name.split(".")[0]+"_distribution-tax.png")
            path_hist=output_name.split("/")[:-1]
            path_hist="/".join(path_hist)
            
            #path_img=os.join(path_hist, "distribution-tax.png")
            path_img=path_hist+"/distribution-tax.png"


            #if the folder doesn't exist create it
            if not os.path.exists(os.path.dirname(path_img)):
                #create it
                os.makedirs(os.path.dirname(path_img))
            

            plt.savefig(path_img)
            #save also a text file with two colomns: the tax and the number associated (22,23,...,K-1)
            with open(path_hist+"/distribution-tax.txt", "w") as file:
                for i, tax in enumerate(big_unique_tax):
                    id=i+21
                    file.write(tax + " : " + str(id) + "\n")
            print("------- encode the MSA into numbers --------")
            MSA=amino_acids_to_numbers(MSA, type_file='fasta', tax=True, Taxonomy=big_unique_tax)
        print("MSA shape: ", MSA.shape)

    elif input_name.endswith('.csv'):
        print("The input file is in csv format, we consider the data type")
        data_full = pd.read_csv(input_name)
        print("we treat the ", data_type, " data")
        #ask to the user if we take only the JD_sequence or the full_prot_sequence with JD or Full 
        data_type=input("Write JD if you want to take the JD_sequence or Full if you want to take the full_prot_sequence ")
        while data_type!='JD' and data_type!='Full':
            data_type=input("Please write JD or Full ")
        if data_type=='Full':
            print("Only JD can be used for now")#need to update the code to take the full_prot_sequence
        if data_type=='JD':
            tax=input("Do you want to know the taxonomy of the sequences for the training? (yes/no) ")
            MSA=data_full['JD_sequence'].tolist()

        print("------- Filtering the data by keeping only the sequences with less than ", threshold, " gaps --------")
        print("number sequences before filtering:", len(MSA))
        MSA = filter_data(MSA, 'csv', float(threshold))
        print("number sequences after filtering:", len(MSA))
        if tax=='no':
            infos_name=path_folder+"/INFOS_no_tax.txt"
            print("------- encode the MSA into numbers --------")
            print(" You will need to take K=21 in the model parameters file")
            MSA=amino_acids_to_numbers(MSA, type_file='csv',tax=False)
            print("MSA shape: ", MSA.shape)
        
        if tax=='yes':
            infos_name=path_folder+"/INFOS_with_tax.txt"
            print("------- encode the different class into numbers --------")
            print(" 8 different classes: \n "
                "For eukaryotes: \n "
                    "21: Viridiplantae \n "
                    "22: Fungi \n "
                    "23: Metazoa \n "
                    "24: Other \n "
                "For bacteria: \n "
                    "25: Alphaproteobacteria \n "
                    "26: Gammaproteobacteria \n "
                    "27: Firmicutes \n "
                    "28: Other \n ")
            print(" You will need to take K=29 in the model parameters file")
            #determine the different class
            class_data,index_to_keep=attribute_class(data_full, MSA, data_type)
            #encode the class_data in a pandas dataframe 
            class_data=pd.DataFrame(class_data)
            #encode the MSA into numbers in a pandas dataframe (the rows are the sequences, the columns are the
            #amino acid postions)
            print("------- encode the MSA into numbers --------")
            MSA=amino_acids_to_numbers(MSA, type_file="csv", tax=True)
            #keep only the sequences that are in the data_sequences file
            MSA=MSA.iloc[index_to_keep]

            # Resetting the index of MSA and class_data before concatenation
            MSA = MSA.reset_index(drop=True)
            class_data = class_data.reset_index(drop=True)
            #concatanate the class to the MSA
            MSA=pd.concat([MSA, class_data], axis=1)
            print("MSA shape (with the class): ", MSA.shape)
    #if the folder doesn't exist create it
    if not os.path.exists(os.path.dirname(output_name)):
        #create it
        os.makedirs(os.path.dirname(output_name))

    #save the (N,L,K) to use it in the model
    #two types of files: with tax and without tax
    
    if os.path.exists(infos_name):
        #look if there is a line writted f"preprocessing with gap threshold of {threshold} %" 
        #if it is the case, remove it and the line after also
        #adapt the file to the new informations
        lines=[]
        with open(infos_name, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line==f"preprocessing with gap threshold of {threshold*100} %\n":
                    print("The line is already in the file")
                    lines.pop(i) #remove the line
                    #if the next line is not ""
                    if lines[i]!="----------------------------------------------------------------\n":
                        lines.pop(i) #remove the next line
                        lines.pop(i)
                    break

        with open(infos_name, "w") as file:
            file.writelines(lines)

        

        
    #write the new informations without removing the previous ones
    with open(infos_name, "a") as file: #"a" for append
        file.write(f"preprocessing with gap threshold of {threshold*100} %\n")
        file.write(f"(N,L,K) = ({MSA.shape[0]},{MSA.shape[1]},{K})\n")
        file.write("----------------------------------------------------------------\n")
    print("(N,L,K) are saved in the file ", infos_name)
    print("Please don't remove this file since the informations are used in the couplings :)")

        
                    
    MSA.to_csv(output_name, header=None, index=None)

def filter_data(MSA,type_file,threshold=1.0,tax=False) :
    """
    remove inserts and sequences with more that 10% gaps from the MSA given in parameter
    """
    output = []
    if type_file=='fasta':
        for sequence in MSA :
            #remove inserts
            sequence.seq = ''.join(res for res in sequence.seq if not (res.islower() or res == '.'))
            
            #keep only sequences with less that 10% gaps
            #if sequence.seq.count('-') < 0.1 * len(sequence) :
            if sequence.seq.count('-') < threshold * len(sequence) :
                if tax==False:
                    output.append(sequence.seq)
                else:
                    OX=sequence.description.split('OX=')[1].split(' ')[0]
                    output.append((OX, sequence.seq))
    if type_file=='csv':
        for sequence in MSA :
            if sequence.count('-') < threshold * len(sequence) : 
                output.append(sequence)
    return output



def attribute_class(data_full, MSA, data_type):
    class_data=[]
    index_to_keep=[]
    for seq in MSA:
        class_seq=find_the_class(data_full, data_type, seq)
        #be sure to have an integer
        if class_seq is not None:
            #print("The sequence is in the data_sequences file")
            class_data.append(class_seq)
            index_to_keep.append(MSA.index(seq))
    
    return class_data,index_to_keep



    
        


def find_the_class(data_full, data_type, seq):
    """
    find the class of the sequence seq in the data_sequences file
    """
    
    if data_type=='JD':
        data_seq=data_full['JD_sequence'].tolist()
        #print the different keys for TheSeq['labelPhyloDom']
        #print("The different keys for TheSeq['labelPhyloDom'] are: ", data_full['labelPhyloDom'].unique())
        if seq in data_seq:
            index=data_seq.index(seq)
            TheSeq=data_full.iloc[index]
            #TheSeq=data_sequences[data_sequences['Sequence']==seq]
            if TheSeq['labelPhyloKing']=='Eukaryota':
                if TheSeq['labelPhyloDom']=='Viridiplantae':
                    return 21
                elif TheSeq['labelPhyloDom']=='Fungi':
                    return 22
                elif TheSeq['labelPhyloDom']=='Metazoa':
                    return 23
                elif TheSeq['labelPhyloDom']=='Other':
                    return 24
            elif TheSeq['labelPhyloKing']=='Bacteria':
                if TheSeq['labelPhyloDom']=='Alphaproteobacteria':
                    return 25
                if TheSeq['labelPhyloDom']=='Gammaproteobacteria':
                    return 26
                if TheSeq['labelPhyloDom']=='Firmicutes':
                    return 27
                elif TheSeq['labelPhyloDom']=='Other':
                    return 28
            return None
    #need to update the code to take the full_prot_sequence

def amino_acids_to_numbers(MSA, type_file, tax=False, Taxonomy=None) :
    """
    takes an MSA in the form of a list of strings and return a panda DataFrame where the rows
    are the sequences and the columns the amino acid positions and the amino acid letters
    are encoded into numbers from 0 to 21
    """
    #put the MSA in the form of a panda DataFrame where the rows are the sequences and
    #the columns the amino acid positions
    MSA = pd.DataFrame(separate_residues(MSA,type_file, tax))
    if tax==False or type_file=='csv':
        #create dictionnary to encode the amino acid letters into numbers from 0 to 20 and consider the unkown letter as gap "-":
        aaRemapInt={'-':0, 'X':0, 'Z':0, 'B':0, 'O':0, 'U':0, 'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8,
                'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}
    else:

        aaRemapInt={'-':0, 'X':0, 'Z':0, 'B':0, 'O':0, 'U':0, 'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8,
                'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}
        #add the tax possibilities to the dictionnary
        for i, tax in enumerate(Taxonomy):
            aaRemapInt[tax]=21+i
        
    #encode the MSA using the dictionnary
    MSA = MSA.replace(aaRemapInt)


    return MSA


def separate_residues(sequences, type_file, tax=False) :
    """
    takes a list of string sequences and returns an array where the row are the sequences
    a the columns are the amino acid positions
    """
    MSA = []
    if tax==False or type_file=='csv' :
        for sequence in sequences :
            MSA.append(list(sequence))
    else: #need to take only with terms before the comma
        for sequence in sequences :
            seq=list(sequence.split(',')[0])
            tax=sequence.split(',')[1]
            MSA.append(seq+[tax])

    return MSA

