import argparse
import average_model

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model_name",help="The name of the model to average without the '_number'.")
    parser.add_argument("n_model", help="How many models we want to average (they should exist).")
    args=parser.parse_args()
    average_model.average_model(args.model_name, args.n_model)