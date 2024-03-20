[still in progress....]


last update 20.03.24: fasta-generation/TwoInOne.py, uniprot-tax/write-list.py, preprocessing.py


update 12.03.24: preprocessing.py, folder fasta-generation and folder uniprot-tax

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

**The construction of the fasta file and the structure 2D map for the contact prediction**. 
In this part you will:

   - align proteins from homologuous sequences data (uniprot, blast, hmm.org) :***alignment.py***
   - combine two proteins together : **TwoInOne.py**
   - construct a structure map file: **dcaTools/mapPDB**
   - define the taxonomy of a sequence: **write_list.py**
   - preprocess the sequences to remove the ones with too much gaps: **preprocessing.py**
     
**PART II:**

**The preparation for the model building and learning, and the couplings between the different positions of amino acids.**
In this part you will:

   - define the proprieties of your model(s), the batchs, number of models, optimizer, ... with **learning_param.py**
   - determine the weights of each sequences in order to have a distribution more "homogenous" (be carefull this is not the weights of the models but "how much a sequence will be considered". ***If a sequence is very semilar with others, its weight will be small to compensate its dominance***): **weights.py**
   - build and train the model(s): **model.py**
   - determine the couplings between the positions of amino acids **couplings.py**

(Additionally, we can run the contact map with **dcaTools/plotTopContacts**)

Each of them is accompanied by **a main file** that can be directly executed from command line: python3 main_<name>.py <parameter1> <parameter2> ...

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
 $   hmmalign path_file_hmm path_file_fasta > path_file_sto
 ```

    This will export you a stockolm file in the given path_file_sto


***End with a last transformation***

Even if hmmer.org is an amazing tool, it will extract the new sequences with a lot of gaps and some characters are in lowercase. To conclure, you can use ***alignment.py*** provided in this folder. This converts your stockolm file in fasta format and adjusts the sequences according to a reference sequence. ***Let's imagine you took BiP homologous in eukaryota: you will give to alignment.py your stockolm file and a a fasta file containing only the sequence BiP_HUMAN***. You will have a new alignment adjusted with BiP_HUMAN.

 ```shell
     $   python3  main_alignment.py path_seq_ref path_file_sto
 ```

(Note that if you already have your stockolm file converted into fasta file, you can still use this function.)

***Be carefull: You need to have the same sequence from your orginal fasta file at the beginning. For example you should have BIP_HUMAN as first sequence. This is important since the program compare your sequence of reference with you first sequence in the fasta file before to remove the useless gaps.***

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
* threshold : The threshold for the percentage of gaps in a sequence. (**Default 1.0**)


```shell 
$  python3 main_preprocessing.py data/PF00226.fasta PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv -threshold 0.1
```

exemple using the default values:

```shell
$  python3 main_preprocessing.py data-MP/hsp70-dnak-bacteria.fasta DnaK-with-tax/preprocessing-1.0gaps/preprocessed-1.0gaps
```

### Part II: description of the files

***learning_param.py***

**Please note that you don't need to compile again different learning parameters files if you want to use one already existing in the folder Parameters_learning. If you want to use the condition considered as the most optimal take SDG_50_<n_models>models_203_0.008lr with <n_models> the numbers of models to train**

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
$ python3 main_learning_param.py 50 32 linear 2 203 Parameters_learning/SDG_50_2models_203_0.008lr -optimizer SGD,0.005,0.01,0,0,False -separations 0.8,0.8  
```
or by using the default values:

```shell
$ python3 main_learning_param.py 50 32 linear 2 Parameters_learning/SDG_50_2models_203_0.008lr

```

***weights.py***

This module takes an MSA preprocessed using preprocessing.py and compute the weights of each sequence. The weights of a sequence is the inverse number of sequences in the MSA that have more than 80% similarity with it. Once computed, the weights will be written in an output file.


Arguments needed by the main :
* input_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* ouput_name : name for the output file containing the weights
* threshold : The percentage of simulutude accepted. **Default=0.8**


```shell
$ python3 main_weights.py PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv PF00226/preprocessing-0.1gapps/weights-0.8/PF00226_weights-0.8.txt -threshold 0.8

```

Exemple with default value:

```shell
$ python3 main_weights.py DnaK-with-tax/preprocessing-1.0gaps/preprocessed-1.0gaps DnaK-with-tax/preprocessing-1.0gaps/weights-0.8/weights-0.8.txt

```


***model.py***

This module builds and trains a neural network on the MSA to be able to predict the value of a residue given all other residues of the sequence.

The user can choose between 3 architectures for the neural network (by default: linear) :

* linear : this is simply a linear classifier (with softmax activation on the output layer and cross entropy loss) whose input and output are the residues and where the output residues are connected to every input residues exept from themselves (in order to avoid a trivial identity)
* non-linear : this model adds a hidden layer to the network, the architecture is designed such that the output residues are disconnected from the corresponding input residues. Two activation function are available for the hidden layer, a custom activation that squares the output of the hidden layer ("square") and a tanh activation ("tanh"). **need to be implemented, not actually working**
* mix : this model is a combination of the first two, the input and output neurons are connected both linearly and via a hidden layer. Both the square and tanh activations are possible for the hidden layer **need to be implemented, not actually working**

After training, the learning curve will be plotted and the model will be saved as well as the error rate after each epoch and the final error rate per residue.

The hyper parameters can be changed in the function "execute" of the file model.py

Arguments needed by the main :
* MSA_name : name of the file containing the preprocessed MSA (i.e. the output file of preprocessing.py)
* weights_name : name of the file containing the weights of the MSA (i.e. the output file of weights.py)
* model_parm : the file .txt format with the different learning parameters
* path: path where to load the file 
* output_name : name that will be used to create the 3 output files (model_+output_name, errors_  +output_name+0-<n_models>,error_postions + output_name+0-<n_models>). **Default:model_average_0-<n_models>**
* activation : activation function for the hidden layer if model_type is "non-linear" or "mix" (otherwise this parameter will be ignored), can be "square" or "tanh". **Default=square.** 

Example of usage :

``` shell
$ python3 main_model.py PF00226/preprocessing-0.1gaps/PF00226_preprocessed-0.1gaps.csv PF00226/preprocessing-0.1gaps/weights-0.8/PF00226_weights-0.8.txt Parameters_learning/SDG_50_5models_203_0.008lr.txt PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203 
```

**What if you want to average several model(s) from a same folder?**
This case can happen if for example you have already made 5 models with seed203 and 5 others ones with seed24. In this case you can put every models in the same folder called for example "10models" and rename correctly the models going from model_0 to model_9. (Don't forget to add a txt file (with nano <name_txt_file>) to indicate which model was made with which seed, parameters a.s.o.). Then don't worry a function will make an average of your models. You just need to use :

   **average_model.py**
   * model_name : The name of the models to average (without the index number _i)
   * n_models : The number of models to average
   ``` shell
   $ python3 main_average_model.py BiP/preprocessing-1.0gaps/weights-0.8/model_linear-50epochs/10models/model 10

   ```
   


***couplings.py***

This module extracts from the trained model (the output of model.py) the coupling coefficients (which describe the interaction between any two residues for each categories) and applies to them a series of operations to make them suitable for contact prediction. It applies the Ising gauge to the coupling coefficients, makes
the matrix of couplings symetrical by averaging with its transpose, takes the Frobenius norm of over all the categories of the residues and apply average product correction.

Arguments needed by the main :
* model_name : name of the file containing the saved model from model.py. For number_models=1 write the path of the model, for number_models>1 write the path of the model without the numbers. As instance if you have model_test_1, model_test_2 you juste need to write model_test
* L : length of the sequences (second dimension of the preprocessed MSA)
* K : number of categories for the residues (21 if no considering class, 29 in general if considering class)
* number_models : if =1, we have only one model, if >1 we can have an average on the couplings or an average on the couplings and frobenius. **Default=1**
* type average : if number_models=1 this is neglected. Otherwise it specifies the kind of average 
    ("average_couplings" or "average_couplings_frob"). **Default='average_couplings'**
* output_name : path of the output file that will contain the coupling coefficients. **Default= "path(model_name)/<type_average>/couplings"**
* figure : Boolean to decide if we want to plot the couplings or not (before and after ising). True or False. **Default: False**
* data_per_col : path for data_per_col.txt representing the number of possible a.a per column in the MSA (created during model.py) (Default: in the same place than the model(s))
* model_type : type of the model, can be "linear", "non-linear" or "mix". **Default: linear**

Example of usage :


```shell
$ python3 main_couplings.py PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/model 63 21 -number_models 5

```
for data PF00226 -> L=63, K=21
for data P25294_SIS1_YEAST -> L=352, K=21
for data A6ZS16_YEAS7 -> L=409, K=21
...

BE CAREFUL: need to take L+1 if we have chose with taxonomy

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

$ python3 dcaTools/plotTopContacts PF00226/contact-map/PF00226.map PF00226/preprocessing-0.1gaps/weights-0.8/model_linear-50epochs/seed203/average-models-and-frob/couplings 150 8.5
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
$ python3 main_preprocessing.py TRY_ME/data/TRY_ME.fasta TRY_ME/preprocessing-1.0gaps/TRY_ME_preprocessed-1.0gaps.csv
```
Weights:
```shell
$ python3 main_weights.py TRY_ME/preprocessing-1.0gaps/TRY_ME_preprocessed-1.0gaps.csv TRY_ME/preprocessing-1.0gaps/weights-0.3/TRY_ME_weights-0.3.txt -threshold 0.3
```
Learning Parameters:
```shell
$ python3 main_learning_param.py 15 32 linear 2 203 Parameters_learning/SDG_15_2models_203_0.008lr
```
Model:
``` shell
$ python3 main_model.py TRY_ME/preprocessing-1.0gaps/TRY_ME_preprocessed-1.0gaps.csv TRY_ME/preprocessing-1.0gaps/weights-0.3/TRY_ME_weights-0.3.txt Parameters_learning/SDG_15_2models_203_0.008lr.txt TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/SDG-32batch/0.001lr-0.01mom-0dampening-0wd-Fnesterov
```
Couplings:
```shell

$ python3 main_couplings.py TRY_ME/preprocessing-1.0gaps/weights-0.3/model_linear-15epochs/SDG-32batch/0.001lr-0.01mom-0dampening-0wd-Fnesterov/model 63 21 -number_models 2 -type_average "average_couplings_frob" 
```





