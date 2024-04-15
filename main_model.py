import argparse
import model
from distutils.util import strtobool

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("MSA_name",help="File containing the preprocessed MSA obtained using preprocessing.py.", type=str)
    parser.add_argument("weights_name", help="File containing the weights of the MSA obtained using weights.py.", type=str)
    parser.add_argument("model_param", help="Type of model to be built and trained saved as a txt. with learning_param.py", type=str)
    parser.add_argument("-path",help="path for the output files. Default=path<weights_name>/model_<model_type>-<epochs>epochs-<batch>batch/seed(seed)", default="",type=str)
    parser.add_argument("-output_name", help="name for the output files, default=model_average_0-<n_models>", default="/", type=str)
    parser.add_argument("-errors_computation", help="If True, the errors will be computed. Default=False", default=False, action="store_true")
    args=parser.parse_args()
    
    model.execute(args.MSA_name, args.weights_name, args.model_param, args.path, args.output_name,args.errors_computation)

