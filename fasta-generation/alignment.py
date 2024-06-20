from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

#the main function is at the end of the file and is called transformation
#the other functions are used in the main function
    #convertissor -> convert a stockolm file to fasta
    #gap_to_remove -> identify the indexes to remove in the fasta file
    #execute -> compare the first sequence of the fasta file with the sequence base and remove the necessary characters

##########################################################################
def convertissor(stockolm_file):
    '''
    This function takes a stockolm file as input and convert it to fasta format.
    It will ask if we want to use the same name and path for the fasta file or a new one.
    input:
        stockolm_file   ->      the stockolm file to convert
                                stockolm file
    output:
        file_name       ->      the name of the fasta file
                                fasta file
    '''
    print("Converting the stockolm file to fasta...")

    file_name=stockolm_file.split(".")[0]
    new_name = input(f"Do you want to use the same name and path for the fasta file? (y/n) ")
    if new_name.lower() == "n":
        file_name = input(f"Please enter the new name of the fasta file (without the extension): ")
    #check is the fasta file already exist
    try:
        with open(f"{file_name}.fasta", "r") as file: 
            print(f"The file {file_name}.fasta already exist")
            overwrite = input("Do you want to overwrite the file? (y/n) ")
            if overwrite.lower() == "y":
                print(f"Overwriting the file {file_name}.fasta")
                pass
            else:
                print("The file will not be overwritten")
                return file_name+".fasta"        
    except:
        pass
    stockfile=SeqIO.parse(stockolm_file, "stockholm")
    SeqIO.write(stockfile, f"{file_name}.fasta", "fasta")
    print(f"The fasta file {file_name}.fasta has been created")
    return file_name+".fasta"
##########################################################################

##########################################################################
def gap_to_remove(seq,fasta_file,seq_init=0,seq_end=0):
    ''' 
    This function will remove all the "-" in the first sequence of the fasta file and remove the same characters in all the sequences of the fasta file.
    It will also remove the part that is not in the sequence of reference.
    input:
            seq         ->    the first sequence of the fasta file
                            string
            fasta_file  ->    the fasta file to modify
                            fasta file
            seq_init    ->    the index of the first character to keep in the sequences
                            int
            seq_end     ->    the index of the last character to keep in the sequences  
                            int
    output:
            new_name_file -> the name of the new fasta file
                            fasta file
    '''
    gap_to_remove = []
    if seq_end == 0:
        seq_end = len(seq)
    for i, char in enumerate(seq):
        if char == "-" or i < seq_init or i > (seq_end-1):
            gap_to_remove.append(i)
    new_records = [] #list of sequences without the "-" characters
    #remove each character of index in gap_to_remove from the sequences in the fasta file. Name this new file ..._new.fasta
    with open(fasta_file, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta")) #list of sequences
        for record in tqdm(records):
            #we join the characters of the sequence that are not in gap_to_remove
            new_seq = "".join([record.seq[i] for i in range(len(record.seq)) if i not in gap_to_remove])
            new_record = record
            new_record.seq = Seq(new_seq)
            new_records.append(new_record)
    
    new_name_file = f"{fasta_file.rsplit('.', 1)[0]}_new.fasta" #new name of the fasta file
    with open(new_name_file, "w") as handle:
        SeqIO.write(new_records, handle, "fasta")
    print("New FASTA file created: ", new_name_file)
##########################################################################


##########################################################################
def execute(seq_base,fasta_file):
    ''' 
    This function takes a sequence seq_base as reference and a fasta file fasta_file as input.
    It compares seq_base with the first sequence in the fasta file and keep only the characters that are the same in the two sequences.
    If one character position is removed, this position will be removed in all the fasta file.
    input:
        seq_base        ->      the sequence to compare with the first sequence of the fasta file
                                string
        fasta_file      ->      the fasta file to modify
                                fasta file
    output:
        fasta_file_new  ->      fasta file with the sequence base as reference
                                fasta file
    '''
    
    print("Treatment of the fasta file...")
    with open(fasta_file, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        for record in records:
            record.seq = record.seq.upper() #put all the sequences in uppercase
        with open(fasta_file, "w") as handle:
            SeqIO.write(records, handle, "fasta") #write the sequences in the fasta file in uppercase

    with open(fasta_file, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        print("Number of sequences in the fasta file: ", len(records))
        seq = records[0].seq #take the first sequence of the fasta file that we will compare with seq_base

    ##################### comparison between seq_base and seq[0] #####################
    seq_without_gap=seq.replace("-","") #look the first sequence of the fasta file without the gap to see if the characters correspond to seq_base
    print("seq_without_gap=",seq_without_gap)
    #compare amino by amino acid and put a gap in seq if the amino acid is different from seq_base
    print("Looking for the id start and id end....")
    if seq_base!=seq_without_gap: 
        print("The sequences are different!")
        if len(seq_base)<len(seq_without_gap):
            Cut=input("It seems that the sequence base is shorter than the sequence in the fasta file. Did you cut it? (y/n) ")
            print("first seq of the fasta file without gap =",seq)
            print("seq_base=",seq_base)
            while Cut!="y" and Cut!="n":
                Cut=input("Please enter (y/n)")
            if Cut=="y":
                seq_init,seq_end=input("Please enter the start and end of the sequence base (separated by a space). Assume the first letter of the original sequence beginning with index 1. ").split()
                while int(seq_init)>int(seq_end):
                    seq_init,seq_end=input("seq_init must be < seq_end" ).split()
                while int(seq_init)<1 :
                    print(seq_base)
                    seq_init,seq_end=input(f"Please enter numbers >1 ").split()
                seq_init=int(seq_init)-1 #index start at 0
                seq_end=int(seq_end)-1
                if seq_base!=seq_without_gap[seq_init:seq_end+1]:
                    print("The sequences are still different")
                    print(seq_base)
                    print(seq_without_gap[seq_init:seq_end+1])
                else:
                    print("Now the sequences are the same and length is: ", len(seq_base))
                    #find the index in seq corresponding to the first element == to seq_without_gap[seq_init]
                    #count the number of same caracters than seq_without_gap[seq_init] before seq_without_gap[seq_init]-1
                    count_start=0
                    count_end=0
                    nb_c=0
                    i=0
                    print("id start...")
                    while nb_c<=seq_init: #stop when we find the good numbers of characters
                        count_start+=1
                        i+=1
                        #check if we have a gap
                        if seq[i]=='-':
                            continue
                        else: #we have a character
                            nb_c+=1 
                    nb_c=0
                    i=len(seq)-1
                    print("id end...")
                    while nb_c<=len(seq_without_gap)-seq_end: #stop when we find the good numbers of characters
                        count_end+=1
                        i-=1
                        #check if we have a gap
                        if seq[i]=='-':
                            continue
                        else:
                            nb_c+=1
                    seq_init=count_start
                    seq_end=len(seq)-count_end+1
                    print("first sequence remained:",seq[seq_init:seq_end])
                    gap_to_remove(seq,fasta_file, seq_init,seq_end)             
    else:
        print("The sequences are the same and length is: ", len(seq_base))
        gap_to_remove(seq,fasta_file)
    ################################################################################  

###########################################################################################################################################
#********************************************** MAIN FUNCTION *****************************************************************************
########################################################################################################################################### 
def transformation(seq_base,file):
    ''' 
    This functions will combine a file already aligned (with for example hmmalign from hmm.org) with a sequence seq_base of length L
    This should remove the unnecessary gaps in the sequences of the fasta file and create a new fasta file with sequences of length L
    input:
        seq_base        ->      the sequence to compare with the first sequence of the fasta file
                                txt file, shape (L,) with L the length of the sequence
                                For example take BIP_HUMAN from uniprot to adjust the homologuous sequences of the fasta file to it
        file            ->      the file to compare with the sequence of base
                                fasta or stockolm file, shape (N,L') with N the number of sequences and L' the length of the sequences

    output:
        fasta_file      ->      if the original file was in stockolm format, the function will convert it to fasta
                                fasta file, shape (N,L') with N the number of sequences and L' the length of the sequences
        fasta_file_new  ->      fasta file with the sequence base as reference, shape (N,L)
    '''
    print("-------- Welcome to the alignment function :) --------")
    #first check that the file is in fasta format
    if file.split(".")[1] == "sto":
        file_new=convertissor(file)
    else:
        file_new = file
    #extract the sequence from the seq_base file
    with open(seq_base, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        seq_ref = records[0].seq
    execute(seq_ref,file_new)
    print("------------- The alignment is done :) --------------")
    print("See you soon!")

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

 

