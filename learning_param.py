   
def execute(epochs, batchs, model_type, n_models, seed, output_name, activation="square",nb_hidden_neurons=32,optimizer="/",separation="/"):
    '''  save as a .csv file the different parameters 
    input: epochs -> int
    batchs -> int
    model_type -> string
    optimizer -> list in string of the name of the optimizer and the different parameters to initialize
    list_dataset -> list of 3 bool elements corresponding to [val, test]
    output: a file named as output_name.txt
    '''
    #check that output_name has an extension .text
    try:
        if output_name[-4:]!=".txt":
            output_name=output_name+".txt"
            pass
    except:
        pass
    # create the csv file
    file = open(output_name, "w")
    # write the parameters
    file.write("epochs: "+str(epochs)+"\n")
    file.write("batchs: "+str(batchs)+"\n")
    file.write("model_type: "+str(model_type)+"\n")
    file.write("n_models: "+str(n_models)+"\n")
    file.write("activation: "+str(activation)+"\n")
    file.write("nb_hidden_neurons: "+str(nb_hidden_neurons)+"\n")
    file.write("seed: "+str(seed)+"\n")

    #separation is loaded as "(int1,int1)" and we want to have int1:
    if separation=="/":
        separation=["0.7,0.7"]
    separation=separation[0].split(",")
    separation[0]=float(separation[0])
    separation[1]=float(separation[1])
    if separation[1] < separation[0]:
        print("Error: the second number of the separation is smaller than the first one")
        return
    if separation[0] == separation[1]:
        file.write("Validation: "+"False"+"\n")
        file.write("Test: "+"True"+"\n")

    elif separation[0]+separation[1]==1:
        file.write("Validation: "+"True"+"\n")
        file.write("Test:"+"False"+"\n")
    else:
        file.write("Validation: "+"True"+"\n")
        file.write("Test: "+"True"+"\n")
    file.write("separation: ("+str(separation[0])+","+str(separation[1])+")\n")


    if optimizer=="/":
        optimizer=["SGD,0.008,0.01,0,0"]
    #write optimizer name : [the parameters separated by comma]
    optimizer=optimizer[0].split(",")
    name_optimizer=optimizer[0]

    optimizer_recognized={}
    optimizer_recognized["Adam"]={}
    if optimizer[0]=="Adam":
        optimizer_recognized["Adam"]["name"]="Adam"
        optimizer_recognized["Adam"]["lr"]=optimizer[1] if len(optimizer)>1 else 0.001
        optimizer_recognized["Adam"]["beta1"]=optimizer[2] if len(optimizer)>2 else 0.9
        optimizer_recognized["Adam"]["beta2"]=optimizer[3] if len(optimizer)>3 else 0.999
        optimizer_recognized["Adam"]["epsilon"]=optimizer[4] if len(optimizer)>4 else 1e-8
        optimizer_recognized["Adam"]["weight_decay"]=optimizer[5] if len(optimizer)>5 else 0
        optimizer_recognized["Adam"]["amsgrad"]=optimizer[6] if len(optimizer)>6 else "False"


    optimizer_recognized["AdamW"]={}
    if optimizer[0]=="AdamW":
        optimizer_recognized["AdamW"]["name"]="AdamW"
        optimizer_recognized["AdamW"]["lr"]=optimizer[1] if len(optimizer)>1 else 0.001
        optimizer_recognized["AdamW"]["beta1"]=optimizer[2] if len(optimizer)>2 else 0.9
        optimizer_recognized["AdamW"]["beta2"]=optimizer[3] if len(optimizer)>3 else 0.999
        optimizer_recognized["AdamW"]["epsilon"]=optimizer[4] if len(optimizer)>4 else 1e-8
        optimizer_recognized["AdamW"]["weight_decay"]=optimizer[5] if len(optimizer)>5 else 0
        optimizer_recognized["AdamW"]["amsgrad"]=optimizer[6] if len(optimizer)>6 else "False"

    optimizer_recognized["SGD"]={}
    if optimizer[0]=="SGD":
        optimizer_recognized["SGD"]["name"]="SGD"
        optimizer_recognized["SGD"]["lr"]=optimizer[1] if len(optimizer)>1 else 0.001
        optimizer_recognized["SGD"]["momentum"]=optimizer[2] if len(optimizer)>2 else 0
        optimizer_recognized["SGD"]["dampening"]=optimizer[3] if len(optimizer)>3 else 0
        optimizer_recognized["SGD"]["weight_decay"]=optimizer[4] if len(optimizer)>4 else 0
        optimizer_recognized["SGD"]["nesterov"]=optimizer[5] if len(optimizer)>5 else "False"
    
    optimizer_recognized["Adagrad"]={}
    if optimizer[0]=="Adagrad":
        optimizer_recognized["Adagrad"]["name"]="Adagrad"
        optimizer_recognized["Adagrad"]["lr"]=optimizer[1] if len(optimizer)>1 else 0.01
        optimizer_recognized["Adagrad"]["lr_decay"]=optimizer[2] if len(optimizer)>2 else 0
        optimizer_recognized["Adagrad"]["weight_decay"]=optimizer[3] if len(optimizer)>3 else 0
        optimizer_recognized["Adagrad"]["initial_accumulator_value"]=optimizer[4] if len(optimizer)>4 else 0
        optimizer_recognized["Adagrad"]["eps"]=optimizer[5] if len(optimizer)>5 else 1e-10

    optimizer_recognized["Adadelta"]={}
    if optimizer[0]=="Adadelta":
        optimizer_recognized["Adadelta"]["name"]="Adadelta"
        optimizer_recognized["Adadelta"]["lr"]=optimizer[1] if len(optimizer)>1 else 1
        optimizer_recognized["Adadelta"]["rho"]=optimizer[2] if len(optimizer)>2 else 0.9
        optimizer_recognized["Adadelta"]["eps"]=optimizer[3] if len(optimizer)>3 else 1e-6
        optimizer_recognized["Adadelta"]["weight_decay"]=optimizer[4] if len(optimizer)>4 else 0

    if name_optimizer not in optimizer_recognized:
        print("the optimizer is not recognized. Correct the error or initialise a new type of optimizer inside main_learning_param.py and model.py")
        return
    #write in the file the dictionary elements:
    #for example if optimizer is "Adam" we write in the file: "Adam: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0, amsgrad=False"
    file.write("optimizer defined by : (")
    for key in optimizer_recognized[name_optimizer]:
        #if it is the last term doesn't write ", " at the end but "\n"
        if key==list(optimizer_recognized[name_optimizer].keys())[-1]:
            file.write(key+")\n")
        else:
            file.write(key+", ")
    file.write("optimizer : ")
    for key in optimizer_recognized[name_optimizer]:
        #if it is the last term doesn't write ", " at the end but "\n"
        if key==list(optimizer_recognized[name_optimizer].keys())[-1]:
            file.write(str(optimizer_recognized[name_optimizer][key])+"\n")
        else:
            file.write(str(optimizer_recognized[name_optimizer][key])+", ")


    # close the file
    file.close()
    
    
        


    