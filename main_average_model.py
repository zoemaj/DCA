import argparse
import average_model
import model 

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("model_name",help="The name of the models to average (without the index number _i).")
    parser.add_argument("n_models",help="The number of models to average.")
    args=parser.parse_args()
    
    average_model.compute_average(args.model_name,args.n_models)
