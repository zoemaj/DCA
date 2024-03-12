
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
        # try to Extract organism from the second term after splitting by '_' (in the example it will take HUMAN)
        #because some time the '_' is in the header_parts[0]
        try:
            organism = header_parts[1].split('_')[1]  # Extract organism from the second term after splitting by '_' (in the example it will take HUMAN)
            pass
        except:
            organism = header_parts[0].split('_')[1]
            pass

        organism = organism.upper()
        #keep only the sequence of the letters after the header parts
        sequence_prot= sequence.seq
        organism_sequences[organism].append(sequence_prot) #add the sequence to the list of sequences for this organism

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
            #ask in terminal if we want to overwrite the file
            overwrite = input("Do you want to overwrite the file? (y/n) ")
            if overwrite.lower() == "y":
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
                output_handle.write(f">{key}\n")
                output_handle.write(str(organism_one[key][i]) + str(organism_two[key][i]) + "\n")
    
    print(f"Number of sequences in the new fasta file: {len(list(SeqIO.parse(file_name, 'fasta')))}")
    #check that the two first sequences are not the same
    with open(file_name, "r") as file: #"r" is for read
        sequences = list(SeqIO.parse(file_name, "fasta"))
        if sequences[5].seq == sequences[6].seq:
            print("The two first sequences are the same")
        else:
            print("oke")
    return

#TwoInOne("construction-fasta/bip-40to65_new.fasta","construction-fasta/hyou1_new.fasta")
