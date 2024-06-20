import pandas as pd
import csv
import gzip
import os



def extract_df(path_file):
    #extract the type of the file
    if path_file.endswith(".xls"):
        df = pd.read_excel(path_file)
        column_names=df.columns.tolist()
        #add the columns names as first line in df
        data = [column_names]
        for i in range(len(df)):
            data.append(df.iloc[i].tolist())
        df=pd.DataFrame(data, columns=column_names)

    elif path_file.endswith(".tsv"):
        # Initialize an empty list to hold the data
        data = []
        try: #the file can be compressed or not
            with open(path_file, 'rb') as file:
                # Check if the file is gzip compressed
                print("treatment of the compressed file")
                if file.read(2) == b'\x1f\x8b':
                    file.seek(0)  # Reset file pointer to beginning
                    with gzip.open(file, 'rt', encoding='utf-8') as tsv_file:
                        tsv_reader = csv.reader(tsv_file, delimiter="\t")
                        for line in tsv_reader:
                            data.append(line)
                            
                else: #the file is not compressed
                    file.seek(0)  # Reset file pointer to beginning
                    with open(path_file, 'r', encoding='utf-8') as tsv_file:
                        tsv_reader = csv.reader(tsv_file, delimiter="\t")
                        for line in tsv_reader:
                            data.append(line)
            #print(data[0]) #the first line is the name of the columns
            column_names=data[0]
            df=pd.DataFrame(data, columns=column_names)
            print("file treated")
        except Exception as e:
            print("An error occurred:", str(e))
    
    else:
        print("The format ",path_file.split(".")[-1]," is not supported. Please use a .xls or .tsv file. (Or update the code)")
        return
    
    return df

def extract_list(df):
    ''' 
    This function extract the different taxonomy identifiers from the dataframe df and save them in a csv file.
    The user can choose the columns to extract from the dataframe.
    If the df contain the columns 'Gene Names (ordered locus)' and 'Gene Names (ORF)', they will be extracted.
    Your two first columns will be "OX" and "Organism_name" (if the column 'Taxonomic lineage' is in the dataframe)

    input:
        df      ->      dataframe obtained from the uniprot file
    output:
        csv file with the different taxonomy identifiers
    '''
    name_sample=input("Please enter the name of your sample (BiP, DnaK, ...) ")
    #node correspond to the first columns
    Node=df["Organism (ID)"][1:]
    #keep only the values
    Node=Node.tolist()

    
    entry_name=df["Entry Name"][1:]
    
    entry_name=entry_name.tolist()
    try:
        Organism_lineage=df["Taxonomic lineage"][1:]
        Organism_name=[]
        for organism in Organism_lineage:
            Organism_name.append(organism.split(",")[-1])
        #be sure to have a space before the name
        for i in range(len(Organism_name)):
            if Organism_name[i][0]!=" ":
                Organism_name[i]=str(" "+Organism_name[i])

            
    except:
        print("The column 'Taxonomic lineage' is not in the dataframe. The column 'Organism_name' will not be extracted")
        pass

    find_one_criteria=False
    try:
        OLN_tab=df["Gene Names (ordered locus)"][1:]
        OLN=[]
        for O in OLN_tab:
            if O!="" and O!="; " and O!=";":
                OLN.append(O)
            else:
                OLN.append(" None")
        #be sure to have a space before the name
        for i in range(len(OLN)):
            if OLN[i][0]!=" ":
                OLN[i]=str(" "+OLN[i])
        find_one_criteria=True
    except:
        print("The column 'Gene Names (ordered locus)' is not in the dataframe. The column 'OLN' will not be extracted")
        pass
    
    try:
        ORF_tab=df["Gene Names (ORF)"][1:]

        ORF=[]
        for O in ORF_tab:
            if O!="" and O!=" " and O!="; " and O!=";":
                ORF.append(O)
            else:
                ORF.append(" None")
        #be sure to have a space before the name
        

        for i in range(len(ORF)):
            if ORF[i][0]!=" ":
                ORF[i]=str(" "+ORF[i])

        find_one_criteria=True
    except:
        print("The column 'Gene Names (ORF)' is not in the dataframe. The column 'ORF' will not be extracted")
        pass
    #print(ORF)
    try:
        organism_strain=df["Organism"][1:]
        strains=[]
        for org in organism_strain:
            if "strain" in org:
                strain_info = org.split("strain ")[1].split(")")[0].strip()
                if "," in strain_info:
                    #remplace it by a dot
                    strain_info = strain_info.replace(",", ".")
                if strain_info.startswith('"') and strain_info.endswith('"'):
                    strain_info = strain_info[1:-1]
                strains.append(strain_info)
            else:
                strains.append("None")
        

        #be sure to have a space before the name
        for i in range(len(strains)):
            if strains[i][0]!=" ":
                strains[i]=str(" "+strains[i])

    except:
        print("The column 'Organism' is not in the dataframe. The column 'Strain' will not be extracted")
        pass

    if find_one_criteria==True:
        for i in range(len(ORF)):
            if ORF[i]==" None" and OLN[i]==" None":
                print(f"The organism with entry_name {entry_name[i]} has no information in the columns 'Gene Names (ordered locus)' and 'Gene Names (ORF)'")
    
    

    column_to_keep=['superkingdom','kingdom']

    tax=df["Taxonomic lineage"]
    columns_names=tax[1].split(",") #take the first line
    #keep only the terms in (..)
    for i in range(len(columns_names)):
        columns_names[i]=columns_names[i].split("(")[1][:-1] #remove the () and the space
    print("The different taxonomy identifiers are the following: ",columns_names)
    #check that column_to_keep is in columns_names, if not remove it from column_to_keep
    for column in column_to_keep:
        if column not in columns_names:
            column_to_keep.remove(column)
    print("The extracted columns will be the following: ",column_to_keep)
    R=input("Do you want to extract another column from the dataframe? (yes/no) ")
    while R!="yes" and R!="no":
        R=input("Please enter yes or no ")
    while R=="yes":
        column=input("Please enter the name of the column you want to extract ")
        if column not in columns_names:
            print("The column ",column," is not in the list of the columns. Please enter a valid column. ")
            print("The different taxonomy identifiers are the following: ",columns_names)
        else:
            column_to_keep.append(column)
            print("The extracted columns will be the following: ",column_to_keep)

        R=input("Do you want to extract another column from the dataframe? (yes/no) ")
        while R!="yes" and R!="no":
            R=input("Please enter yes or no ")
    
    dictionary_columns_to_keep={}
    list_columns_found={}
    dictionary_columns_to_keep["OX"]=Node 
    try:
        dictionary_columns_to_keep["Organism_name"]=Organism_name
    except:
        pass
    try:
        dictionary_columns_to_keep["OLN"]=OLN 
    except:
        pass
    try:
        dictionary_columns_to_keep["ORF"]=ORF
    except:
        pass
    try:
        dictionary_columns_to_keep["Strain"]=strains
    except:
        pass
    
    for column in column_to_keep: #initialize the dictionary
        dictionary_columns_to_keep[column]=[]
        list_columns_found[column]=False
    for i,line in enumerate(tax[1:]):#for each line except the first one corresponding the the header names
        line=line.split(",") #split the line
        for el in line:
            #look if there is a "()"
            if "(" in el:
                line_name=el.split("(")[1][:-1] #remove the term in ()
            if line_name in column_to_keep: #for example if it is a superkingdom or a kingdom
                #dictionary_columns_to_keep[line_name].append(el.split("(")[0]) # modification due to the following reason
                #the next operation is because we detect a problem with the column superkingdom for "Viruses" that doesn't have a space before....
                #this causes problem with the preprocessing.py. To be sure that we don't have any problem, we add a space before the values that don't start with a space
                name_to_add=el.split("(")[0]
                if name_to_add[0]!=" ":
                    name_to_add=" "+name_to_add
                dictionary_columns_to_keep[line_name].append(name_to_add)
                ############################
                list_columns_found[line_name]=True
        #check that we have all the columns
        for key, value in list_columns_found.items():
            if value==False:
                dictionary_columns_to_keep[key].append(" Other")
        #reset the list
        list_columns_found={key: False for key in list_columns_found.keys()}
                   
    #
    #for each key of dictionary except Node
    for key, value in dictionary_columns_to_keep.items():
        if key=="OX" or key=="Organism_name" or key=="OLN" or key=="ORF" or key=="Strain":
            continue
        else:
            #save the unique values of the column, and order them with the higher number of occurences first
            unique_N = set(value)
            unique_N = sorted(unique_N, key=lambda x: value.count(x), reverse=True)
            #write a text file with the different unique values for 'N' of the dictionary eukaryota. So we should not have any repeated value
            with open("uniprot-tax/"+name_sample+"-"+key+".txt", "w") as file:
                for line in unique_N:
                    file.write(line + ": " + str(value.count(line)) + "\n")
    #add the organism name after the Node     
    #save the dictionary as a dataframe
    df=pd.DataFrame(dictionary_columns_to_keep)
    #save the dataframe as a csv file
    name_output="uniprot-tax/"+name_sample+"-tax.csv"
    #look if the file exist
    if os.path.exists(name_output):
        R=input("The file "+name_output+" already exists. Do you want to overwrite it? (yes/no) ")
        while R!="yes" and R!="no":
            R=input("Please enter yes or no ")
        if R=="yes":
            df.to_csv(name_output,index=False)
            print("The file ",name_output," has been created")
        else:
            R2=input("Do you want to save the file with another name? (yes/no) ")
            while R2!="yes" and R2!="no":
                R2=input("Please enter yes or no ")
            if R2=="yes":
                name_output=input("Please enter the name of the file ")
                df.to_csv(name_output,index=False)
                print("The file ",name_output," has been created")
    else:
        df.to_csv(name_output,index=False)
        print("The file ",name_output," has been created")


def execute(path_file):
    df=extract_df(path_file)
    extract_list(df)
    return 





