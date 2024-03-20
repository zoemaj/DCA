
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm

def sort_sequences(fasta_file):
    ''' 
    This function takes a fasta file and returns a dictionary with the organism as key and the sequences as values
    input: fasta_file -> fasta file with the homologues sequences of different organisms
    output: organism_sequences -> dictionary with the organism as key and the sequences as values
    '''
    #print the numbers of sequences in fasta file
    print(f"Number of sequences: {len(list(SeqIO.parse(fasta_file, 'fasta')))}")
    organism_sequences=defaultdict(list) #defaultdict is a dictionary that has a default value for keys that were not added yet

    for sequence in SeqIO.parse(fasta_file, "fasta"):



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
        #keep only the sequence of the letters after the header parts
        sequence_prot= sequence.seq
        #extract the OX number:
        OX=None
        for el in header_parts:
            try:
                OX=el.split('OX=')[1]
                pass
            except:
                pass
        if OX==None:
            R=input('No OX identification, it will not be possible to preprocess this file with taxonomy in the future. Do you still want to continue? (yes/no)')
            while  R!='yes' and R!='no':
                R=input('Please write yes or no:')
            if R=='no':
                return
            
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
            
        
        organism_sequences[OX].append([number_name,organism_name,OS,sequence_prot]) #add the sequence to the list of sequences for this organism

    print(f"Number of different organism: {len(organism_sequences.keys())}")
    return organism_sequences

def TwoInOne(fastafile_one,fastafile_two):
    ''' 
    This function will combine two fasta files be combining the sequence of the same organism
    '''
    organism_one=sort_sequences(fastafile_one)
    organism_two=sort_sequences(fastafile_two)
    #check is they are similar keys name between organism_one and organism_two:
    common_keys = filter(lambda x: x in organism_one, organism_two)
    common_keys=list(common_keys)
    #check that common_keys is not empty:
    if len(common_keys)==0:
        print("No common keys found")
        return
    
    #ask the name of the two proteins
    file_name= input("Please enter the name of the two proteins (without extension) separated by a dash: ")
    path= input("Please enter the path of the new fasta file: ")
    file_name = path + '/' +file_name + ".fasta"
    
    #if file name already exist ask if we want to overwrite it
    try:
        with open(file_name, "r") as file: #"r" is for read
            print(f"The file {file_name} already exist")

            overwrite = input("Do you want to overwrite the file? (yes/no) ")
            while overwrite!="yes" and overwrite!="no":
                overwrite = input("Please enter yes or no ")
            if overwrite=="yes":
                print(f"Overwriting the file {file_name}")
                pass
            else:
                print("The file will not be overwritten")
                return
    except:
        pass
    #write the new fasta file
    with open(file_name, "w") as output_handle: #write the sequence of the two files combined only for the common keys
        for key in tqdm(common_keys):
            #take the same number of sequences for the two proteins
            n=min(len(organism_one[key]),len(organism_two[key]))
            for i in range(n):
                if organism_one[key][i][2]!=organism_two[key][i][2]:
                    print(f"Organism name (OS) are different for the same OX {key} number: {organism_one[key][i][2]} and {organism_two[key][i][2]}")
                    OS=organism_one[key][i][2]+"-"+organism_two[key][i][2]
                else:
                    OS=organism_one[key][i][2]
                output_handle.write(f">sp {organism_one[key][i][0]}-{organism_two[key][i][0]}_{organism_one[key][i][1]} OS={OS} OX={key}\n") #should be the same for the two proteins
                #take the seq of the protein, second argument of the list
                output_handle.write(str(organism_one[key][i][3]) + str(organism_two[key][i][3])+ "\n")
    
    print(f"Number of sequences in the new fasta file: {len(list(SeqIO.parse(file_name, 'fasta')))}")
   

