[still in progress....]
# Neural Network DCA
-----------------------------------------------------------

Welcome to the exploration of DnaJ domain and SIS1 protein sequences by using Direct Coupling Analysis, pseudolikelihood maximization, and machine learning. If you just want to read the information about the functions, please go directly to **Content and Organization** :)

If you want to know everything, such as what is the goal of this project or how the folders are structured, please continue.

If you want to test some functions with a short set of sequences please go directly to **TRY ME**

## Description
-----------------------------------------------------------
The goal of Neural Network DCA is to enable DCA-based protein contact prediction using non-linear models. The scripts contained in this repository allow to train different neural network architectures on an MSA to predict the type of a residue given all other residues in a sequence and then to extract the knowledge learned by the network to do contact prediction.

## What about chaperons?
-----------------------------------------------------------
Misfolded proteins can lead to aggregation, resulting in neuromuscular and neurodegenerative diseases, or lysosomal dysfunction. Heat shock proteins 70 (HSP70)
play a crucial role as chaperones in various protein folding processes, involving ATP hydrolysis facilitated by the J-domain binding to HSP70. Recent research have explored DnaJ domain and SIS1 protein sequences using Direct Coupling Analysis, pseudolikelihood maximization, and machine learning. However, inappropriate results
necessitated a new approach, involving code modifications and optimization. These changes include a new couplings formulation, a variable number of amino acid values per position, a smaller batch size, hyperparameter tuning with different optimizers (Adam, AdamW, SGD, Adagrad, and AdaDelta), a comparison by taking the average across different models or couplings, or on the Frobenius norms. Additionally, a comparison was conducted by learning the taxonomy of the sequence or not. Furthermore, protein contact predictions were also performed for the Mitochondrial protein import protein MAS5 (gene YDJ1).

## Installation
-----------------------------------------------------------
To run these scripts, the user must have the following Python packages installed :

* numpy
* pandas
* csv
* gzip
* biopython
* numba
* torch
* itertools
* matplotlib
* os

## Orgnanization of the files
-----------------------------------------------------------
Please keep the same organization of folders and names.
Each protein has its own file with its name Z.

- **For each folder Z** you have these different folders:
* **preprocessing-X.Xgaps** where X.X is a float number indicating the percent of max gaps (ex preprocessing-0.1gaps for 10%)

    * Z-preprocessed-X.Xgaps.csv
    * **weights-Y.Y** where Y.Y is a float number indicating the percentage of similarity (ex: weights-0.8 for 80%)

        * weights-Y.Y.txt
        * **model_MODEL.TYPE-Eepochs** where MODEL.TYPE is "linear, non-linear or mix" and E is an int number for the epochs (ex: model_linear-50epochs)

            * **Adam-Bbatch** where B is a int number for the batch size (ex: Adam-64batchs)
                * Llr-B1beta1-B2beta2-Eepsilon-Wwd-Aamsgrad where L is the learning rate, B1 is beta1, B2 is beta2, E is epsilon, W is the weight decay and A is the amsgrad with T for true and F for false (ex: 0.001lr-0.9beta1-0.999beta2-1e-8epsilon-0wd-Famsgrad)
            
            * **AdamW-Bbatch** where B is a int number for the batch size (ex: Adam-64batchs)
                * Llr-B1beta1-B2beta2-Eepsilon-Wwd-Aamsgrad where L is the learning rate, B1 is beta1, B2 is beta2, E is epsilon, W is the weight decay and A is the amsgrad (ex: 0.001lr-0.9beta1-0.999beta2-1e-8epsilon-0.01wd-Famsgrad)

            * **SDG-Bbatch** where B is a int number for the batch size (ex: SDG-32batchs)
                * Llr-Mmom-Ddampening-Wwd-Nnesterov  where L is the learning rate, M is the momentum, D is the dampening, W is the weight decay and N is the nesterov with F for false and T for true (ex with default values: 0.001lr-0mom-0dampening-0wd-Fnesterov)
            
            * **Adagrad-Bbatch** where B is a int number for the batch size (ex: SDG-32batchs)
                * Llr-LDld-Wwd-Ainitaccum-Eeps where L is the learning rate,LD is the learning decay, W is the weight decay, A is the inital accumulator value and E is the epsilon with F for false and T for true (ex with default values:  0.01lr-0ld-0wd-0initaccum-1e-10eps)

            * **Adadelta-Bbatch** where B is a int number for the batch size (ex: Adadelta-32batch)
                * Llr-Rrho-Eepsilon-Wwd where L is the learning rate, R is rho, E is epsilon and W is the weight decay (ex with default values: 1.0lr-0.9rho-1e-6epsilon-0wd)
                    default val
            
* **map**
    * Z.map where Z is the name of the protein
* **pdb**
    * Z.pdb where Z is the name of the protein
* **hmm**
    * F.hmm where F is the name of the Family of the protein Z
* **data** (Note: the folder data contain proteins used for my master of specialisation and the folder data-MP for the data used in my master project. Please feel free to create your own data folder.)
    * Z.fasta where Z is the name of the protein

Note that inside **each folder of OPTIMIZER.Type-Bbatch** you can find the model numeroted, the model averaged for a selection of models numeroted, the png images and a folder for the average-couplings and average-couplings-frob
for exemple you can find
* **average-couplings** 
    * contact_0-4-300-8.5.png : a contact plot for 300 predictions on the models 0,1,2,3,4
    * couplings_0-4 : the couplings file for the average with the models 0,1,2,3,4
    * errors_positions_0.4.txt
    * errors_0-4.txt
* **average-couplings-and-frob**
    * contact_3-5-150-8.5.png : a contact plot for 150 predictions on the models 3,4,5
    * couplings_3-5 : the couplings file for the average with the models 3,4,5
    * errors_positions_3-5.txt
    * errors_3-5.txt
* **average-models**
    * contact_150-8.5.png : a contact plot for 150 predictions on the models 3,4,5
    * couplings_0-30 : the couplings file for the average with the 30 firstly models
    * errors_positions_0-30.txt
    * errors_0-30.txt
* contact_0-4-70-8.5.png : a contact plot for 70 predictions on the model average
* errors_positions_0-4.txt : the errors on the model average from the models 0,1,2,3,4
* errors_0-4.txt 
* model_0
* model_1
* model_2
* model_3
* model_4
* model_5
* model_average_0-4

## Content and Organization
-----------------------------------------------------------
There are two parts:


**PART I:**

The construction of the fasta file and the structure 2D map for the contact prediction. 
In this part you will:

   - align proteins from homologuous sequences data (uniprot, blast, hmm.org) :***alignment.py***
   - combine two proteins together : **TwoInOne.py**
   - construct a structure map file: **dcaTools/mapPDB**
   - define the taxonomy of a sequence: **write_list.py**
   - preprocess the sequences to remove the ones with too much gaps: **preprocessing.py**
     
**PART II:**

The preparation for the model building and learning, and the couplings between the different positions of amino acids.
In this part you will:

   - define the proprieties of your model(s), the batchs, number of models, optimizer, ... with **learning_param.py**
   - determine the weights of each sequences to make the distribution more "homogenous" (be carefull this is not the weights of the models but "how much a sequence will be considered". If a sequence is very semilar with others, its weight will be low to compensate its dominance): **weights.py**
   - build and train the model(s): **model.py**
   - determine the couplings between the positions of amino acids **couplings.py**

(Additionally, we can run the contact map with **dcaTools/plotTopContacts**)

Each of them is accompanied by **a main file** that can be directly executed from command line.

### Part I: description of the files
-----------------------------------------------------------

***How find your fasta file?***

-You can use [uniprot](https://www.uniprot.org) to find some homologous sequences of your choice.

-If you want to extract from these sequences only the ones that are identical between Percentage_inf and Percentage_sup, you can use [Blast](https://blast.ncbi.nlm.nih.gov/Blast.cgi).
    


***How find your hmm file (indispensable for alignment)?***


-You can type the name of your family in "search by text" (Hsp70, GrpE, ...) on [Interpro](https://www.ebi.ac.uk/interpro/) and download the hmm file ***from PFAM source file*** in the section ***curration***


***Now that you have your fasta and hmm files, how align the sequences ?***


-Please read the protocole from [hmmer.org](http://hmmer.org)

-After downloading the folder hmmer-3.4 (from hmmer.org) you can use this command in the terminal:
    (you need to be in the folder hmmer-3.4)

 ```shell
 $   hmmalign path_file_hmm path_file_fasta > path_file_sto
 ```

    This should export you a stockolm file in your folder


***End with a last transformation***

Even if hmmer.org is an amazing tool, it will extract the new sequences with a lot of gaps and some characters are in lowercase. You can finally use ***alignment.py*** provided in this folder. This will convert your stockolm file in fasta format and then will adjust the sequences according to a reference sequence. Let's imagine you took BiP homologous in eukaryota: you will give to alignment.py your stockolm file and a a fasta file containing only the sequence BiP_HUMAN. You will have a new alignment adjusted with BiP_HUMAN.

 ```shell
     $   python3  main_alignment.py path_seq_ref path_file_sto
 ```

(Note that if you already have your stockolm file converted into fasta file, you can still use this function.)

Be carefull: You need to have the same sequence from your orginal fasta file at the beginning. For example you should have BIP_HUMAN as first sequence. This is important since the program compare your sequence of reference with you first sequence in the fasta file before to remove the useless gaps.

***How align two family together?***

You first need to do the previous steps to have correct fasta files with sequences aligned in function of a reference sequence. Then you can use the following command:

 ```shell
     $   python3  main_TwoInOne.py file1 file2
 ```
***How define the taxonomy of a sequence with write_list.py***
folder needed: uniprot-tax/list

Before to use this module you need to download a tsv or xls of your homologous proteins file from uniprot. This file need to contain the attributes "Organism (ID)" and "Taxonomic lineage". You can save the file in the folder uniprot-tax/list.

This module will read your file and create a csv file (in the folder uniprot-tax) containing each organism ID with specific attributes of the taxonomic lineage determined by the user during inputs. This will also create summary txt files, for each attribute, giving the unique elements and number of occurences.

Arguments needed by the main :
* path_file : The path where to find the list from uniprot.
  
```shell 
$  python3 main_write_list.py uniprot-tax/list/bip-taxonomy.xls
```

***How process you file with preprocessing.py***
 
The user will answer to some questions in the terminal (keep the taxonomy or not, which protein ?, which taxonomy (kingdom, division,...)? If you want to use the taxonomy, you absolutly need to create the appropriate file with write_list.py

folder needed: uniprot-tax

Arguments needed by the main :
* input_name : name of the file containing the MSA in fasta or csv format
* output_name : name that will be used to create the output file
* threshold : The threshold for the percentage of gaps in a sequence. (Default 1.0)


```shell 
$  python3 main_preprocessing.py data/PF00226.fasta PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv 0.1
```

```shell
$  python3 main_preprocessing.py data-MP/hsp70-dnak-bacteria.fasta DnaK-with-tax-new/preprocessing-1.0gaps/preprocessed-1.0gaps
```

### Part II: description of the files

***learning_param.py***

This module takes 

Arguments needed by the main :
* epochs : number of epochs to train, int
* batchs : batch_size used during the training, int
* modeL-type : type of the model: linear, non-linear or mix
* n_models : the numbers of models to train before to keep the average on them or apply the average on their couplings
* separations : the proportion that we keep for the training, the validation and the test '(perc_train,perc_val)' such that we take perc_train% of the data, (perc_train-perc_val)% of the remained data and finaly what is still there for the test data
* nb_hidden_neurons : the number of hidden_neurons (0 for linear type)
* optimizer : a list with '[optimizer_name,optimizer_param1,...,optimizer_paramk]' if there are not enough parameters for this type of optimizer the last ones will be fill by default values
    [Adam, lr, beta1, beta2, epsilon, wd, amsgrad]
    [AdamW, lr, beta1, beta2, epsilon, wd, amsgrad]
    [SGD, lr, ,momentum, damping, wd, nesterov]
    [Adagrad, lr, lr_decay, wd, initial accumular value,epsilon]
    [Adadelta, lr, rho, epsilon, wd]
* seed : the seed for the random number generator
* output_name : the name of the output files with the different parameters

**Please note that you don't need to compile again different learning parameters files if you want to use one already existing in the folder Parameters_learning. If you want to use the condition considered as the most optimal take SDG_50_<n_models>models_203_0.008lr with <n_models> the numbers of models to train**




Example of usage :

```shell
$ python3 main_learning_param.py 50 32 linear 2 '(0.7,0.7)' 0 "[SGD,0.008, 0.01,0,0, False]" 203 Parameters_learning/SDG_50_2models_203_0.008lr
```



***preprocessing_new.py*** [update 05.03.24] -> will replace preprocessing.py
This module is similar to the previous one but allow the user to keep the information of the taxonomy not only for csv file but also fasta file. The user will answer to some questions in the terminal (keep the taxonomy or not, eukaryota or bacteria or others?, which taxonomy (kingdom, division,...)?

For now only the eukaryota species are possible with taxonomy. Need futur improvement.

file needed: uniprot/eukaryota-species.txt

Arguments needed by the main : (no more need to precise the data_type)
* input_name : name of the file containing the MSA in fasta or csv format
* threshold : The threshold for the percentage of gaps in a sequence. 
* output_name : name that will be used to create the output file

```shell
$  python3 main_preprocessing.py PF00226/data/PF00226.fasta 0.1 PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv
```

***weights.py***

This module takes an MSA preprocessed using preprocessing.py and compute the weights of each sequence. The weights of a sequence is the inverse number of sequences in the MSA that have more than 80% similarity with it. Once computed, the weights will be written in an output file.


Arguments needed by the main :
* input_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* threshold : The percentage of simulutude accepted
* ouput_name : name for the output file containing the weights

```shell
$ python3 main_weights.py PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv 0.8 PF00226/preprocessing-0.1gapps/weights-0.8/PF00226_weights-0.8.txt

```



***model.py***

This module builds and trains a neural network on the MSA to be able to predict the value of a residue given all other residues of the sequence.
The user can choose between 3 architectures for the neural network :
* linear : this is simply a linear classifier (with softmax activation on the output layer and cross entropy loss) whose input and output are the residues and where the output residues are connected to every input residues exept from themselves (in order to avoid a trivial identity)
* non-linear : this model adds a hidden layer to the network, the architecture is designed such that the output residues are disconnected from the corresponding input residues. Two activation function are available for the hidden layer, a custom activation that squares the output of the hidden layer ("square") and a tanh activation ("tanh")
* mix : this model is a combination of the first two, the input and output neurons are connected both linearly and via a hidden layer. Both the square and tanh activations are possible for the hidden layer

After training, the learning curve will be plotted and the model will be saved as well as the error rate after each epoch and the final error rate per residue.

The hyper parameters can be changed in the function "execute" of the file model.py

Arguments needed by the main :
* MSA_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* weights_name : name of the file containing the weights of the MSA (i.e. the output file of weights.py)
* model_parm : the file .txt format with the different learning parameters
* activation : activation function for the hidden layer if model_type is "non-linear" or "mix" (otherwise this parameter will be ignored), can be "square" or "tanh"
* path: path where to load the file 
* output_name : name that will be used to create the 3 output files (model_+output_name, errors_  +output_name,error_postions + output_name). if you want nothings juste write / for the output_name.

Example of usage :

``` shell
$ python3 main_model.py PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv PF00226/preprocessing-0.1gaps/weights-0.8/PF00226_weights-0.8.txt Parameters_learning/SDG_50_5models_203_0.008lr.txt square PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203 /
```

***couplings.py***

This module extracts from the trained model (the output of model.py) the coupling coefficients (which describe the interaction between any two residues for each categories) and applies to them a series of operations to make them suitable for contact prediction. It applies the Ising gauge to the coupling coefficients, makes
the matrix of couplings symetrical by averaging with its transpose, takes the Frobenius norm of over all the categories of the residues and apply average product correction.

Arguments needed by the main :
* model_name : name of the file containing the saved model from model.py. For number_models=1 write the path of the model, for number_models>1 write the path of the model without the numbers. As instance if you have model_test_1, model_test_2 you juste need to write model_test
* model_type : type of the model, can be "linear", "non-linear" or "mix"
* L : length of the sequences (second dimension of the preprocessed MSA)
* K : number of categories for the residues (21 if no considering class, 29 in general if considering class)
* data_per_col : number of possible a.a per column in the MSA (normally in the same place than the model(s))
* number_models : if =1, we have only one model, if >1 we can have an average on the couplings or an average on the couplings and frobenius
* type average : if number_models=1 this is neglected. Otherwise it specifies the kind of average 
    ("average_couplings" or "average_couplings_frob").
* output_name : path of the output file that will contain the coupling coefficients. 

Example of usage :

```shell

$ python3 main_couplings.py PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/model linear 63 21 PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/data_per_col.txt 5 'average_couplings_frob' PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/couplings
```
for data PF00226 -> L=63, K=21
for data P25294_SIS1_YEAST -> L=352, K=21
for data A6ZS16_YEAS7 -> L=409, K=21

***
In addition to these 4 modules. The directory dcaTools contains 3 scripts downloaded from https://gitlab.com/ducciomalinverni/dcaTools.git that enable to make and evaluate contact prediction using the output of couplings.py. Details on their usage can be found here https://link.springer.com/protocol/10.1007%2F978-1-4939-9608-7_16.

***extract the contact plot***
This module plot the predicted contact on the ones predicted by alpha fold
Argument needed by the main:
* pdbMap : Structural contact map extracted from PDB
* dcaFile : DCA prediction file
* Ntop : The number of top ranked DCA contacts to plot.
* contactThreshold : The threshold in Angstroms defining a structural contact
* output_name : Name for the output file -> this is not from dca-Tools but has been added for simplifications

```shell

$ python3 dcaTools/plotTopContacts PF00226/contact-map/PF00226.map PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/average-models-and-frob/couplings 150 8.5 PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/average-models-and-frob/contact-150-8.5.png
```

***mapPDB***

Compute the distance map of a PDB structure for the purpose of evaluating the predictions made from the coupling coeffcicients.

***PlotTopContacts***

Plots a user-defined number of highest-ranking DCA predicted contacts, overlaid on PDB contact map. The output file format of couplings.py is adapted to make it suitable for this script

***PlotTPrates***

Plots the precision curves for DCA predicted contacts. The output file format of couplings.py is adapted to make it suitable for this script

# TRY ME
You can find a test into the folder TRY_ME to try the different modules before to run long simulation. The sequences are the 30 first sequences of the j-domain PF00226.

Example of usage :

Preprocessing:
```shell
$ python3 main_preprocessing.py TRY_ME/data/TRY_ME.fasta 1.0 TRY_ME/preprocessing-1.0gaps/TRY_ME_preprocessed-1.0gaps.csv
```
Weights:
```shell
$ python3 main_weights.py TRY_ME/preprocessing-1.0gaps/TRY_ME_preprocessed-1.0gaps.csv 0.3 TRY_ME/preprocessing-1.0gaps/weights-0.3/TRY_ME_weights-0.3.txt
```
Learning Parameters:
```shell
$ python3 main_learning_param.py 15 32 linear 2 '(0.7,0.7)' 0 "[SGD,0.008, 0.01,0,0, False]" 203 Parameters_learning/SDG_15_2models_203_0.008lr
```
Model:
``` shell
$ python3 main_model.py TRY_ME/preprocessing-1.0gaps/TRY_ME_preprocessed-1.0gaps.csv TRY_ME/preprocessing-1.0gaps/weights-0.3/TRY_ME_weights-0.3.txt Parameters_learning/SDG_15_2models_203_0.008lr.txt square TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/SDG-32batch/0.001lr-0.01mom-0dampening-0wd-Fnesterov /
```
Couplings:
```shell

$ python3 main_couplings.py TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/SDG-32batch/0.001lr-0.01mom-0dampening-0wd-Fnesterov/model linear 63 21 TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/SDG-32batch/0.001lr-0.01mom-0dampening-0wd-Fnesterov/data_per_col.txt 2 "average_couplings_frob" TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/SDG-32batch/0.001lr-0.01mom-0dampening-0wd-Fnesterov/couplings
```





