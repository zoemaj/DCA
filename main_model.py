import argparse
import model
from distutils.util import strtobool

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("MSA_name",help="File containing the preprocessed MSA obtained using preprocessing.py.", type=str)
    parser.add_argument("weights_name", help="File containing the weights of the MSA obtained using weights.py.", type=str)
    parser.add_argument("model_param", help="Type of model to be built and trained saved as a txt. with learning_param.py", type=str)
    parser.add_argument("-length_prot1", help="If the fasta file is composed of pairs of proteins A and B, and you want to learn to find A with only B (and vice versa), you can specify the length of the first protein. Default=0", default=0, type=int)
    parser.add_argument("-path",help="path for the output files. Default=path<weights_name>/model_<model_type>-<epochs>epochs-<batch>batch/seed(seed)", default="",type=str)
    parser.add_argument("-output_name", help="name for the output files, default=model_average_0-<n_models>", default="/", type=str)
    parser.add_argument("-errors_computation", help="If True, the errors will be computed. Default=False", default=False, action="store_true")
    args=parser.parse_args()
    
    model.execute(args.MSA_name, args.weights_name, args.model_param, args.length_prot1, args.path, args.output_name, args.errors_computation)

