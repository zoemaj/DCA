$${\color{red} still \space \color{red} in \space \color{red} progress....}$$

**last update 25.04.24**


Zoé Majeux

EPFL, Laboratoire de biophysique statistique (LBS)

Prof. Paolo De Los Rios


# Neural Network DCA
-----------------------------------------------------------

Welcome to the exploration of DnaJ domain and SIS1 protein sequences by using Direct Coupling Analysis, pseudolikelihood maximization, and machine learning. If you just want to read the information about the functions, please go directly to **Content and Organization** :)

If you want to know everything, such as "what is the goal of this project" or "how the folders are structured", please continue.

If you want to test some functions with a short set of sequences please go directly to **TRY ME**

## Description
-----------------------------------------------------------
The goal of Neural Network DCA is to enable DCA-based protein contact prediction using non-linear models. The scripts contained in this repository allow to train different neural network architectures on an MSA to **predict the type of a residue given all other residues** in a sequence and then to **extract the knowledge learned by the network to do contact prediction**.

## What about chaperons?
-----------------------------------------------------------
Misfolded proteins can lead to aggregation, resulting in neuromuscular and neurodegenerative diseases, or lysosomal dysfunction. **Heat shock proteins 70 (HSP70)** play a crucial role as chaperones in various protein folding processes, involving **ATP hydrolysis facilitated by the J-domain binding to HSP70**. 

## Installation
-----------------------------------------------------------
To run these scripts, the user must have the following Python packages installed :

* numpy
* python
* pandas
* csv
* gzip
* biopython
* numba
* torch
* itertools
* matplotlib
* os
* tqdm
* psutils

## Orgnanization of the files
-----------------------------------------------------------
Please keep the same organization of folders and names.
Each protein has its own file with its name Z.

- **For each folder Z** you have these different folders:
* **preprocessing-X.Xgaps** where X.X is a float number indicating the percent of max gaps (ex preprocessing-0.1gaps for 10%)

    * preprocessed-X.Xgaps.csv
    * INFOS_no_tax.txt and INFOS_tax.txt where you can find for each preprocessing made with X.Xgaps the number of sequences N, the length L and the number K of different values possible
    * **weights-Y.Y** where Y.Y is a float number indicating the percentage of similarity (ex: weights-0.8 for 80%)

        * weights-Y.Y.txt
        * **model_MODEL.TYPE-Eepochs-Bbatchs/seedS** where MODEL.TYPE is "linear od non-linear", E is an int number for the epochs, B is an int number for the batchs and S an int number for the random choice (ex: model_linear-50epochs-32batchs/seed203). Note that is B is not precised it means that it is done with 32 batchs.
     
             * data_per_col.txt (representing which value (between 0 and K) is possible for each position 
             * model_0,...,model_n,model_average0-n,errors_0,...,errors_n,errors_positions_0,...,errors_positions_n,...
             * **average-models** couplings only done on the model_average
             * **average-couplings** couplings and ising done on each models before taken the average and then the frobenius norm
             * **average-couplings-and-frob** couplings, ising and frobenius norm done on each models before taken the average

- Additionally you have these different folders:

* **map**
    * Z.map where Z is the name of the protein
* **pdb**
    * Z.pdb where Z is the name of the protein
* **hmm**
    * F.hmm where F is the name of the Family of the protein Z
* **data** (Note: the folder data contain proteins used for my master of specialisation and the folder data-MP for the data used in my master project. Please feel free to create your own data folder.)
    * Z.fasta where Z is the name of the protein

## Content and Organization
-----------------------------------------------------------
There are two parts:

$${\color{purple} ------------------- \space \color{purple} PART \space \color{purple} I \space \color{purple} ------------------  }$$

**The construction of the fasta file and the structure 2D map for the contact prediction**. 
In this part you will:

   - align proteins from homologuous sequences data (uniprot, blast, hmm.org) :***alignment.py***
   - combine two proteins together : **TwoInOne.py**
   - construct a structure map file: **dcaTools/mapPDB**
   - define the taxonomy of a sequence: **write_list.py**
   - preprocess the sequences to remove the ones with too much gaps: **preprocessing.py**
     
$${\color{purple} ------------------- \space \color{purple} PART \space \color{purple} II \space \color{purple} ------------------  }$$

**The preparation for the model building and learning, and the couplings between the different positions of amino acids.**
In this part you will:

   - define the proprieties of your model(s), the batchs, number of models, optimizer, ... with **learning_param.py**
   - determine the weights of each sequences in order to have a distribution more "homogenous" (be carefull this is not the weights of the models but "how much a sequence will be considered". ***If a sequence is very semilar with others, its weight will be small to compensate its dominance***): **weights.py**
   - build and train the model(s): **model.py**
   - determine the couplings between the positions of amino acids **couplings.py**
$${\color{purple} ---------------------------------------------------------------------------------------------  }$$
(Additionally, we can run the contact map with **dcaTools/plotTopContacts**)

Each of them is accompanied by **a main file** that can be directly executed from command line: python3 main_NAME.py PARAMETER1 PARAMETER2 ...

### Part I: description of the files
-----------------------------------------------------------

***How find your fasta file?***

-You can use [uniprot](https://www.uniprot.org) to find some homologous sequences of your choice.

-If you want to extract from these sequences only the ones that are identical between Percentage_inf and Percentage_sup, you can use [Blast](https://blast.ncbi.nlm.nih.gov/Blast.cgi).
    


***How find your hmm file (indispensable for alignment)?***


-You can type the name of your family in "search by text" (Hsp70, GrpE, ...) in [Interpro](https://www.ebi.ac.uk/interpro/) and download the hmm file ***from PFAM source file*** in the section ***curration***


***Now that you have your fasta and hmm files, how align the sequences ?***


-Please read the protocole from [hmmer.org](http://hmmer.org)

-After downloading the folder hmmer-3.4 (from hmmer.org) you can use this command in the terminal:
    (you need to be in the folder hmmer-3.4)

 ```shell
    hmmalign path_file_hmm path_file_fasta > path_file_sto
 ```

    This will export you a stockolm file in the given path_file_sto


***End with a last transformation***

Even if hmmer.org is an amazing tool, it will extract the new sequences with a lot of gaps and some characters are in lowercase. To conclure, you can use ***alignment.py*** provided in this folder. This converts your stockolm file in fasta format and adjusts the sequences according to a reference sequence. ***Let's imagine you took BiP homologous in eukaryota: you will give to alignment.py your stockolm file and a a fasta file containing only the sequence BiP_HUMAN***. You will have a new alignment adjusted with BiP_HUMAN.

 ```shell
     python3  main_alignment.py path_seq_ref path_file_sto
 ```

(Note that if you already have your stockolm file converted into fasta file, you can still use this function.)

You can also use this file if you want to take only one part of your reference sequence (for exemple if you want to have only the J domain and not the entere protein), the program will notice that it is smaller than your first sequence and adapt your file. 

***Be carefull: You need to have the same sequence from your orginal fasta file at the beginning. For example you should have BIP_HUMAN as first sequence. This is important since the program compare your sequence of reference with you first sequence in the fasta file before to remove the useless gaps.***

***How align two family together?***

You first need to do the previous steps to have correct fasta files with sequences aligned in function of a reference sequence. Then you can use the following command:

 ```shell
    python3  main_TwoInOne.py file1 file2
 ```
***How define the taxonomy of a sequence with write_list.py***
folder needed: uniprot-tax/list

Before to use this module you need to download a tsv or xls of your homologous proteins file from uniprot. This file need to contain the attributes "Organism (ID)" and "Taxonomic lineage". You can save the file in the folder uniprot-tax/list.

This module will read your file and create a csv file (in the folder uniprot-tax) containing each organism ID with specific attributes of the taxonomic lineage determined by the user during inputs. This will also create summary txt files, for each attribute, giving the unique elements and number of occurences.

If your file contain the columns 'Gene Names (ordered locus)' and 'Gene Names (ORF)', they will be extracted. (usefull for pairing two types of bacteria proteins)

Arguments needed by the main :
* path_file : The path where to find the list from uniprot.
  
```shell 
   python3 main_write_list.py uniprot-tax/list/bip-taxonomy.xls
```

***How process you file with preprocessing.py***
 
The user will answer to some questions in the terminal (keep the taxonomy or not, which protein ?, which taxonomy (kingdom, division,...)? If you want to use the taxonomy, you absolutly need to create the appropriate file with write_list.py

You can specify the maximum percentage of gaps that you authorise. You can also specify the minimum/maximum of similarity between the sequences and the first one that you authorised. Every sequences with more than this percentage, or with too few or too many similarity, are deleted. Specify the similiraty percentage can be usefull if you don't want sequences too different with your reference sequence per exemple. 

folder needed: uniprot-tax

Arguments needed by the main :
* input_name : name of the file containing the MSA in fasta or csv format
* output_name : name that will be used to create the output file (**Default path/input_name/preprocessing-Tgaps/preprocessed_T.csv with T the threshold chosen**) 
* threshold : The threshold for the percentage of gaps in a sequence authorised. (**Default 1.0**)
* min_sim : The minimum similarity authorised between the sequences and the first one. (**Default=0.0**)
* max_sim : The maximum similarity authorised between the sequences and the first one. (**Default=1.0**)


```shell 
  python3 main_preprocessing.py DnaK/DnaK.fasta 
```

exemple using not the default values:

```shell
  python3 main_preprocessing.py DnaK/DnaK.fasta -output_name DnaK/preprocessing-1.0gaps/preprocessed-1.0gaps.csv -threshold 0.1 -min_sim 0.4 -max_sim 0.6
```

If you apply this function, you will notice that several files can be created:

-> If you don't use taxonomy: INFOS_no_tax.txt will be created, indicating every (N,L,K) for every type of threshold preprocessing that you used.

-> If you use taxonomy: INFOS_with_tax.txt will be created, indicating every (N,L,K) for every type of threshold preprocessing that you used. preprocessing-T/distribution-tax.txt and preprocessing-T/distribution-tax.png are created. The text file will give you which taxonomy correspond to which number (22,23,...) and the png file is a figure representing the distribution.

-> If you use min_sim or max_sim: The precedent files and the preprocesssing file will be in a new folder names filtered_min_sim_MIN_max_sim_MAX (with MIN and MAX the values that you have chosen)

### Part II: description of the files

***learning_param.py***

**Please note that you don't need to compile again different learning parameters files if you want to use one already existing in the folder Parameters_learning. If you want to use the condition considered as the most optimal take SDG_50_N_MODELSmodels_203_0.008lr with N_MODELS the numbers of models to train**

This module takes 

Arguments needed by the main :
* epochs : number of epochs to train, int
* batchs : batch_size used during the training, int
* modeL-type : type of the model: linear, non-linear or mix
* n_models : the numbers of models to train before to keep the average on them or apply the average on their couplings
* seed : the seed for the random number generator
* output_name : the name of the output files with the different parameters
* optimizer : a list with optimizer_name,optimizer_param1,...,optimizer_paramk if there are not enough parameters for this type of optimizer the last ones will be fill by default values
    [Adam, lr, beta1, beta2, epsilon, wd, amsgrad]
    [AdamW, lr, beta1, beta2, epsilon, wd, amsgrad]
    [SGD, lr, ,momentum, damping, wd, nesterov]
    [Adagrad, lr, lr_decay, wd, initial accumular value,epsilon]
    [Adadelta, lr, rho, epsilon, wd]
  **default= SGD, 0.008, 0.01, 0, 0, False**
* separations : the proportion that we keep for the training, the validation and the test perc_train,perc_val such that we take perc_train% of the data, (perc_train-perc_val)% of the remained data and finaly what is still there for the test data. **default=(0.7,0.7)**
* nb_hidden_neurons : the number of hidden_neurons (**Default=0, linear case**)

Example of usage :

```shell
python3 main_learning_param.py 50 32 linear 2 203 Parameters_learning/SDG_50_2models_203_0.008lr

```

***weights.py***

This module takes an MSA preprocessed using preprocessing.py and compute the weights of each sequence. The weights of a sequence is the inverse number of sequences in the MSA that have more than 80% similarity with it. Once computed, the weights will be written in an output file.


Arguments needed by the main :
* input_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* ouput_name : name for the output file containing the weights **Default=path(input_name)/weights-T/weights-T.txt with T the threshold that you have chosen**
* threshold : The percentage of simulutude accepted. **Default=0.8**


```shell
python3 main_weights.py DnaK/preprocessing-0.1gaps/DnaK_preprocessed-0.1gaps.csv 

```

Exemple with default value:

```shell
python3 main_weights.py DnaK/preprocessing-1.0gaps/preprocessed-1.0gaps -output_name DnaK/preprocessing-0.1gaps/DnaK_weights-0.8.txt -threshold 0.1

```


***model.py***

This module builds and trains a neural network on the MSA to be able to predict the value of a residue given all other residues of the sequence.

You can choose between 2 architectures for the neural network (by default: linear) :

* linear : this is simply a linear classifier (with softmax activation on the output layer and cross entropy loss) whose input and output are the residues and where the output residues are connected to every input residues exept from themselves (in order to avoid a trivial identity)
* non-linear : this model adds a hidden layer to the network, the architecture is designed such that the output residues are disconnected from the corresponding input residues. Two activation function are available for the hidden layer, a custom activation that squares the output of the hidden layer ("square") and a tanh activation ("tanh"). 

If your fasta file is actually homologous series of two proteins A and B in pair (you are want to visualise their binding) you have two options: 

* predict ai from A with the amino acids of A (without ai) and of B (and vice versa with B) with a linear or non-linear architecture -> normal situation, don't precise a length_prot1
* predict ai from A only with the amino acids of B (and vice versa with A) with a linear or non-linear architecture -> you need to precise the length of A with the argument length_prot1


After training, the learning curve will be plotted and the model will be saved as well as the error rate after each epoch and the final error rate per residue.

Arguments needed by the main :
* MSA_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* weights_name : name of the file containing the weights of the MSA (i.e. the output file of weights.py)
* model_parm : the file .txt format with the different learning parameters
* length_prot1 : If the fasta file is composed of pairs of proteins A and B, and you want to learn to find A with only B (and vice versa), you can specify the length of the first protein. **Default=0**
* path: path where to load the file  **Default:path(weights_name)/model_MODEL_TYPE-Eepochs-Bbatchs/seedS with MODEL_TYPE the type linear or non-linear, E the numbers of epochs and B the numbers of batchs (all defined in the model_param)**
* output_name : name that will be used to create the 3 output files (model_+output_name, errors_  +output_name+0-N_MODELS,error_postions + output_name+0-N_MODELS). **Default:model_average_0-N_MODELS WITH N_MODELS the number of models**
* errors_computations : If True, the errors will be computed. **Default=False**

Example of usage :

``` shell
python3 main_model.py DnaK/preprocessing-0.1gaps/preprocessed-0.1gaps.csv DnaK/preprocessing-0.1gaps/weights-0.8/weights-0.8.txt Parameters_learning/SDG_50_5models_203_0.008lr.txt
```

**What if you want to average several model(s) from a same folder?**
This case can happen if for example you have already made 5 models with seed203 and 5 others ones with seed24. In this case you can put every models in the same folder called for example "10models" and rename correctly the models going from model_0 to model_9. (Don't forget to add a txt file (with nano NAME_TXT_FILE) to indicate which model was made with which seed, parameters a.s.o.). Then don't worry a function will make an average of your models. You just need to use :

   **average_model.py**
   * model_name : The name of the models to average (without the index number _i)
   * n_models : The number of models to average
     
   ``` shell
   python3 main_average_model.py BiP/preprocessing-1.0gaps/weights-0.8/model_linear-50epochs/10models/model 10

   ```

***couplings.py***

This module extracts from the trained model (the output of model.py) the coupling coefficients (which describe the interaction between any two residues for each categories) and applies to them a series of operations to make them suitable for contact prediction. It applies the Ising gauge to the coupling coefficients, makes
the matrix of couplings symetrical by averaging with its transpose, takes the Frobenius norm of over all the categories of the residues and apply average product correction.

Arguments needed by the main :
* model_name : name of the file containing the saved model from model.py. For number_models=1 write the path of the model, for number_models>1 write the path of the model without the numbers. As instance if you have model_test_1, model_test_2 you juste need to write model_test
* length_prot1 : If the fasta file is composed of pairs of proteins A and B, and you want to learn to find A with only B (and vice versa), you can specify the length of the first protein. **Default=0**
* number_models : if =1, we have only one model, if >1 we can have an average on the couplings or an average on the couplings and frobenius. **Default=1**
* type average : if number_models=1 this is neglected. Otherwise it specifies the kind of average 
    ("average_couplings" or "average_couplings_frob"). **Default='average_couplings'**
* output_name : path of the output file that will contain the coupling coefficients. **Default= "path(model_name)/type_average/couplings"**
* figure : Boolean to decide if we want to plot the couplings or not (before and after ising). True or False. **Default: False**
* data_per_col : path for data_per_col.txt representing the number of possible a.a per column in the MSA (created during model.py) (Default: in the same place than the model(s))
* model_type : type of the model, can be "linear", "non-linear" or "mix". **Default: linear**
* L : length of the sequence, if we have not the INFOS file. **Default=0**
* K : value of K, if we have not the INFOS file **Default=0**

Example of usage :


```shell
python3 main_couplings.py DnaK/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/model -number_models 5
```

***
In addition to these 4 modules. The directory dcaTools contains [3 scripts](https://gitlab.com/ducciomalinverni/dcaTools.git) that enable to make and evaluate contact prediction using the output of couplings.py. Details on their usage can be found [here](https://link.springer.com/protocol/10.1007%2F978-1-4939-9608-7_16).

***extract the contact plot***
This module plot the predicted contact on the ones predicted by alpha fold
Argument needed by the main:
* pdbMap : Structural contact map extracted from PDB
* dcaFile : DCA prediction file
* Ntop : The number of top ranked DCA contacts to plot.
* contactThreshold : The threshold in Angstroms defining a structural contact
* output_name : Name for the output file -> this is not from dca-Tools but has been added for simplifications. (**Default: will same to the same path than the one of pdbMap**)
* ik : minimal sequence separation in the alignment to extract contacts (**default 4**)

```shell

python3 dcaTools/plotTopContacts PF00226/contact-map/PF00226.map PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/average-models-and-frob/couplings 150 8.5
```

***mapPDB***

Compute the distance map of a PDB structure for the purpose of evaluating the predictions made from the coupling coeffcicients.

***PlotTopContacts***

Plots a user-defined number of highest-ranking DCA predicted contacts, overlaid on PDB contact map. The output file format of couplings.py is adapted to make it suitable for this script

***PlotTPrates***

Plots the precision curves for DCA predicted contacts. The output file format of couplings.py is adapted to make it suitable for this script

# TRY ME
You can find a test into the folder TRY_ME to try the different modules before to run long simulation. Don't panick if the 2D structure is not good, it is normal since TRY_ME has not a lot of sequences

Example of usage :

Preprocessing:
```shell
python3 main_preprocessing.py TRY_ME/data/TRY_ME.fasta -threshold 0.1
```
Weights:
```shell
python3 main_weights.py TRY_ME/preprocessing-1.0gaps/preprocessed-1.0gaps.csv -threshold 0.3
```
Learning Parameters:
```shell
python3 main_learning_param.py linear 2 203 Parameters_learning/SDG_15_2models_203_0.008lr
```
Model:
``` shell
python3 main_model.py TRY_ME/preprocessing-1.0gaps/preprocessed-1.0gaps.csv TRY_ME/preprocessing-1.0gaps/weights-0.3/weights-0.3.txt Parameters_learning/SDG_15_2models_203_0.008lr.txt 
```
Couplings:
```shell

python3 main_couplings.py TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/model -number_models 2 -type_average average_couplings_frob
```





