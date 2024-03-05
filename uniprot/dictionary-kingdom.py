import pandas as pd
# Open the text file and read its contents
with open("uniprot/list-uniprot.txt", "r") as file:
    lines = file.readlines()

# Find the starting index of the table
start_index = 0
for idx, line in enumerate(lines):
    if line.startswith("Code"):
        start_index = idx+4
        break
end_index = 0
#in reverse order
for idx, line in enumerate(lines):
    if line.startswith("(2)"):
        end_index= idx-3

# Extract the table lines from start_index and end_index
# Extract also the table from end_index+6 to the end
#put together the two parts of the table
table_lines = lines[start_index:end_index] + lines[end_index+6:-5]

#save
with open("uniprot/table-kingdom.txt", "w") as file:
    for line in table_lines:
        file.write(line)

# Initialize an empty dictionary to store the mappings
node_to_c_dict = {}
#Iterate over the lines
i = 0
while i < len(table_lines)-3:
    # Extract the two lines
    line1 = table_lines[i].split()
    kingdom= line1[1]
    node = line1[2][:-1]
    N= line1[3][2:]
    #check that the following line contains the c value, this means has a C at the beginning
    if str(table_lines[i + 1].split()[0][0:2]) == "C=":
        line2 = table_lines[i + 1].split()
        C = line2[0][2:]
        
        # Check if the next line starts with a space (unicode value=32)
        if i + 2 < len(table_lines) and table_lines[i + 2].split()[0][0:2] == "S=":
            i += 3  # Skip the next line (and the next after it)
        else:
            i += 2  # Move to the next pair of lines
    else:
        C=""
        # Check if the next line starts with a space (unicode value=32)
        if i + 1 < len(table_lines) and table_lines[i + 1].split()[0][0:2] == "S=":
            i += 2  # Skip the next line (and the next after it)
        else:
            i += 1  # Move to the next pair of lines


    # Store the mapping
    #print(node, kingdom, N,C)
    node_to_c_dict[node] = [kingdom,N,C]

#save the dictionary as a dataframe
df = pd.DataFrame(list(node_to_c_dict.items()),columns = ['Node',['Kingdom','N','C']])
#save the dataframe as a csv file
df.to_csv("uniprot/dictionary-kingdom.csv",index=False)

kingdoms = list(node_to_c_dict.values())

''' A: archaea kingdom'''
''' B: bacteria kingdom
    -> Alphaproteobacteria
    -> Gammaproteobacteria
    -> Firmicutes
    -> Other
''' 
''' E: eukaryota domain
    -> animalia kingdom 
    -> fungi kingdom
    -> plantae kingdom
    -> protista kingdom
    -> other
    

'''
''' V: viruses kingdom'''
''' U: unclassified kingdom'''

#make a dictionary with only 'E' Kingdoms
node_to_c_dict_eukaryota = {}
for key, value in node_to_c_dict.items():
    if value[0] == 'E':
        node_to_c_dict_eukaryota[key] = value
#write a text file with the different unique values for 'N' of the dictionary eukaryota. So we should not have any repeated value
unique_N = set([value[1] for value in node_to_c_dict_eukaryota.values()])
with open("uniprot/listE-uniprot.txt", "w") as file:
    for line in unique_N:
        file.write(line + "\n")

#create a new file for eukaryota species by looking the kingdom and the node. 
with open("uniprot/eukaryota-species.txt", "w") as file:
    for key, value in node_to_c_dict_eukaryota.items():
        #write the node number followed by the division/embranchment found in taxonomy-uniprot.txt for the correct 'N'
        file.write(key + ", ")
        #open the file taxonomy-uniprot.txt and look for the line of the good name 'N'
        with open("uniprot/E-uniprot.txt", "r") as file2:
            lines = file2.readlines()
            for line in lines:
                if line.startswith(value[1]): #should write Abalistes, Animalia, Chordata 
                    #check if line.split()[2] exist, if not write Other
                    if len(line.split()) > 2:
                        file.write(line.split()[1] + " " + line.split()[2] + "\n")
                    elif len(line.split()) > 1:
                        file.write(line.split()[1] + ", Other\n")
                    else:
                        file.write("Other, Other\n")
                    break
            else:
                file.write("Other, Other\n")

