import numpy as np
import pandas as pd
from Bio import SeqIO
import os

def preprocessing(input_name, threshold, output_name) :
    """
    load the sequences from the input file, remove inserts, remove sequences with more than 10% gaps, encode the MSA into numbers
    and write the preprocessed MSA in the output_file

    :param input_name: name of the input file containing the sequences in fasta format
    :type input_name: string for the csv file
    :param output_name: name of the output file which will contain the preprocessed MSA in csv format
    :type output_name: string
    :return: nothing
    :rtype: None
    """
    #load the sequences
    print("hey")
    if input_name.endswith('.fasta'):
        print("The input file is in fasta format, we don't consider the class type")
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
            print(" You will need to take K=21 in the model parameters file")
            MSA = filter_data(MSA,'fasta',float(threshold),tax=False)
            print("number sequences after filtering:", len(MSA))
            print("------- encode the MSA into numbers --------")
            MSA=amino_acids_to_numbers(MSA, type_file='fasta',tax=False)
        if A=='yes':
            MSA = filter_data(MSA,'fasta',float(threshold),tax=True)
            #ask to the user if the data are eukaryote or bacteria or both
            R=input("Are the data eukaryota, bacteria or both? ")
            while R!='eukaryota' and R!='bacteria' and R!='both':
                R=input("Please write eukaryota, bacteria or both ")
            if R=='bacteria':
                print("only eukaryote types can be used with taxonomy for now")
                return
            if R=='both':
                print("only eukaryote types can be used with taxonomy for now")
                return
            if R=='eukaryota':
                #ask if the user want to distinct the taxonomy by kingdom (Animalia, Fungi, Plantae, Protista, Other) or by division (chordata, arthropoda, ...)
                R2=input("write K if you want to distinct the taxonomy by kingdom (Animalia, Fungi, Plantae, Protista, Other) or write D if you want to disctinct the taxonomy by division (chordata, arthropoda, ...)? ")
                while R2!='K' and R2!='D':
                    input("Please write K or D ")
                #load the file euakaryota-species.txt and create a dictionary with the node and the division or the kingdom
                with open("uniprot-with-tax/eukaryota-species.txt", "r") as file:
                    lines = file.readlines()
                    #create a dictionary with the node and the division or the kingdom
                    E_dic = {}
                    for line in lines:
                        node, kingdom, division = line.split(", ")
                        if R2=='K':
                            E_dic[node] = kingdom
                        if R2=='D':
                            E_dic[node] = division
                newMSA = []
                for sequence in MSA:
                    #extrat the Description
                    #take only the part that start with OX
                    OX=sequence[0]
                    #find the node in the dictionary
                    if OX not in E_dic:
                        tax='Other'
                    else:
                        tax=str(E_dic.get(OX))
                    newMSA.append(sequence[1] + ", "+ tax)
                MSA=newMSA
                #find the number of different tax (last term of each sequence)
                unique_tax = set([value.split(',')[1] for value in MSA])
                #print("The different tax are: ", unique_tax)
                K=21
                print("You will need to take K=", len(unique_tax)+K, " in the model parameters file")
            print("------- encode the MSA into numbers --------")
            MSA=amino_acids_to_numbers(MSA, type_file='fasta', tax=True, Taxonomy=unique_tax)
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
            print("------- encode the MSA into numbers --------")
            print(" You will need to take K=21 in the model parameters file")
            MSA=amino_acids_to_numbers(MSA, type_file='csv',tax=False)
            print("MSA shape: ", MSA.shape)
        
        if tax=='yes':
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
    MSA.to_csv(output_name, header=None, index=None)

def filter_data(MSA,type_file,threshold=0.1,tax=False) :
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

