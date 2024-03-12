from Bio import SeqIO
from Bio.Seq import Seq
import tqdm

def convertissor(stockolm_file):
    print("Converting the stockolm file to fasta...")
    #we will remove the extension of the file
    file_name=stockolm_file.split(".")[0]
    #ask if we want to use the same path and name or a new one:
    new_name = input(f"Do you want to use the same name and path for the fasta file? (y/n) ")
    if new_name.lower() == "n":
        file_name = input(f"Please enter the new name of the fasta file (without the extension): ")

    #check is the fasta file already exist
    try:
        with open(f"{file_name}.fasta", "r") as file: #"r" is for read
            print(f"The file {file_name}.fasta already exist")
            #ask in terminal if we want to overwrite the file
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

def gap_to_remove(seq,fasta_file,seq_init=0,seq_end=0):
    gap_to_remove=[]
    if seq_end==0:
        seq_end=len(seq)
    #print("seq=",seq)
    with open("seq_id.txt","w") as file:
        for i in range(len(seq)):
            if int(i)<int(seq_init) or int(i)>int(seq_end):
                gap_to_remove.append(i)
                #wrrite the index of the gap to remove in a file
                file.write(str(i)+"\n")
            elif seq[i] == "-":
                gap_to_remove.append(i)
                #wrrite the index of the gap to remove in a file
                file.write(str(i)+"\n")

    #remove each character of index in gap_to_remove from the sequences in the fasta file. Name this new file ..._new.fasta
    with open(fasta_file, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta")) #list of sequences
        #delete each character of index in gap_to_remove from the sequences in the fasta file bip.fasta
        for record in tqdm.tqdm(records):
        #for record in records:
            record.seq = "".join([record.seq[i] for i in range(len(record.seq)) if i not in gap_to_remove])
            #define record.seq as a sequence Seq
            record.seq = Seq(str(record.seq))
            
        path=fasta_file.split("/")[:-1]
        path="/".join(path)
        new_name_file=path+"/"+fasta_file.split("/")[-1].split(".")[0]+"_new.fasta"
        with open(new_name_file, "w") as handle:
            SeqIO.write(records, handle, "fasta")
        print("new fasta file created in the same folder as the original fasta file: ", new_name_file)

def execute(seq_base,fasta_file):
    ''' 
    This function takes a sequence seq_base and a fasta file fasta_file as input.
    It compares seq_base with the first sequence in the fasta file. If the sequences are different, it prints "The sequences are different".
    If the sequences are the same, it prints "The sequences are the same and length is: " and the length of the sequence.
    It then removes all the "-" in the first sequence of the fasta file and remove the same characters in all the sequences of the fasta file.
    It creates a new fasta file bip_new.fasta with the sequences without the "-" characters.
    '''
    print("in execute")
    #put in upper case all sequence in the fasta file bip
    with open(fasta_file, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        for record in records:
            record.seq = record.seq.upper()
        with open(fasta_file, "w") as handle:
            SeqIO.write(records, handle, "fasta")

    #take seq (should correspond to the one given by seq_base) as the first sequence in the fasta file construction-fasta/bip.fasta
    with open(fasta_file, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        seq = records[0].seq

    #compare seq_base with seq: remove all the "-" in seq. 
    seq_without_gap=seq.replace("-","")
    #compare amino by amino acid and put a gap in seq if the amino acid is different from seq_base
    #print(seq_without_gap)
    if seq_base!=seq_without_gap:
        print("The sequences are different")
        if len(seq_base)<len(seq_without_gap):
            Cut=input("It seems that the sequence base is shorter than the sequence in the fasta file. Did you cut it? (y/n) ")
            while Cut!="y" and Cut!="n":
                Cut=input("Please enter (y/n)")
            if Cut=="y":
                seq_init,seq_end=input("Please enter the start and end of the sequence base (separated by a space). Assume the first letter of the original sequence beginning with index 1. ").split()
                while int(seq_init)>int(seq_end):
                    seq_init,seq_end=input("seq_init must be < seq_end" ).split()
                while int(seq_init)<1 or int(seq_end)>len(seq_base)+1:
                    seq_init,seq_end=input(f"Please enter numbers between 1 and {len(seq_base)} ").split()
                seq_init=int(seq_init)-1 #index start at 0
                seq_end=int(seq_end)-1
                if seq_base!=seq_without_gap[seq_init:seq_end]:
                    print("The sequences are still different")
                else:
                    print("Now the sequences are the same and length is: ", len(seq_base))
                    #find the index in seq corresponding to the first element == to seq_without_gap[seq_init-1]
                    #count the number of same caracters than seq_without_gap[seq_init-1] before seq_without_gap[seq_init-1]
                    count_start=0
                    count_end=0
                    for i in range(0,seq_init,1):
                        if seq_without_gap[i]==seq_without_gap[seq_init]:
                            count_start+=1

                    #in the opposite direction for count_end, from the end to seq_end
                    #will go from len(seq_without_gap[seq_end:])+1 to len(seq_without_gap) included
                    for i in range(seq_end+1,len(seq_without_gap),1):
                        if seq_without_gap[i]==seq_without_gap[seq_end]:
                            count_end+=1
                    

        
                    for i in range(len(seq)): #stop when seq[i] correspond to seq_without_gap[seq_init-1] after count_start
                        if seq[i]==seq_without_gap[seq_init]:
                            if count_start==0:
                                seq_init=i
                                break
                            count_start-=1

                
                    
                    #find the index in seq corresponding to the count_end eme element before seq_without_gap[seq_end-1] that is == to seq_without_gap[seq_end-1]
                    # in the opposite direction 
                    for i in range(len(seq)-1,-1,-1):
                        if seq[i]==seq_without_gap[seq_end]:
                            if count_end==0:
                                seq_end=i
                                break
                            count_end-=1
                    
                    gap_to_remove(seq,fasta_file, seq_init,seq_end)
                    
                    
    else:
        print("The sequences are the same and length is: ", len(seq_base))
        gap_to_remove(seq,fasta_file)
        


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
 

