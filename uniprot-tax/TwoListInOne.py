import os

def combine(file1, file2,name):
    #create a new file with the elements of file1 and file2.
    #Don't add two times the same element (i.e the same OX=file1.split(", ")[0]=file2.split(", ")[0])
    with open(name, "w") as file:
        with open(file1, "r") as f:
            for line in f:
                file.write(line)
        with open(file2, "r") as f:
            for line in f:
                if line not in open(name).read():
                    file.write(line)
    print("The file ",name," has been created")
def execute(file1, file2):
    #create the file in the same folder than file1 and file2 with name: name1-name2
    path_file = os.path.dirname(file1) + "/"
    name=path_file + file1.split("/")[-1].split("-")[0] + "-" + file2.split("/")[-1].split("-")[0] + ".csv"
    print(name)
    #check if the file already exists
    if os.path.exists(name):
        R=input("The file "+name+" already exists. Do you want to overwrite it? (yes/no) ")
        while R!="yes" and R!="no":
            R=input("Please enter yes or no ")
        if R=="no":
            return
        if R=="yes":
            combine(file1, file2,name)
    else:
        combine(file1, file2,name)

    return
            

    
            
        