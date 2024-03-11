import argparse
import model
from distutils.util import strtobool

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("MSA_name",help="File containing the preprocessed MSA obtained using preprocessing.py.", type=str)
    parser.add_argument("weights_name", help="File containing the weights of the MSA obtained using weights.py.", type=str)
    parser.add_argument("model_param", help="Type of model to be built and trained saved as a txt. with learning_param.py", type=str)
    parser.add_argument("path",help="path for the output files", type=str)
    parser.add_argument("-output_name", help="name for the output files, default=model_average_0-<n_models>", default="/", type=str)
    parser.add_argument("-activation", help="activation function of hidden layer if model_type is 'non-linear' or 'mix', can be 'square' or 'tanh', if model_type is 'linear' the parameter will be ignored. default='square'", default="square", type=str)
    args=parser.parse_args()
    
    model.execute(args.MSA_name, args.weights_name, args.model_param, args.path, args.output_name, args.activation)

