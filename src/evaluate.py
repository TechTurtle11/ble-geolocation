import argparse
from pathlib import Path
from statistics import mode
import numpy as np

from localisation import run_convergence_localisation_on_file, run_localisation_on_file
import Models.models as models
import Utils.constants as const
from plotting import plot_evaluation_metric, produce_average_localisation_distance_plot


def rmse(predictions):
    """
    Returns the mean root mean square error for each algorithm
    """
    rmse_results = {}

    for algorithm,predictions in predictions.items():
        values = [np.linalg.norm(actual - prediction)**2 for actual, prediction in predictions]
        rmse_results[algorithm] = np.sqrt(np.mean(values))

    return rmse_results


def mae(predictions):
    """
    Returns the Mean Absolute error for each algorithm
    """
    mae_results = {}

    for algorithm,predictions in predictions.items():
        values = [np.linalg.norm(actual - prediction) for actual, prediction in predictions]
        mae_results[algorithm] = np.sqrt(np.mean(values))

    return mae_results

def mae_confidence_interval(predictions):
    """
    calculates 95% confidence interval
    """

    mae_conf_results = {}

    samples = len(predictions)
    for algorithm,predictions in predictions.items():
        dist = [np.linalg.norm(actual - prediction) for actual, prediction in predictions]
        mae = np.mean(dist)
        std = np.std(dist)
        confidence_interval  = 1.960 * std / np.sqrt(len(dist)) 
        mae_conf_results[algorithm] = np.round([mae,confidence_interval],2)

    return mae_conf_results

def rmse_confidence_interval(predictions):
    """
    calculates 95% confidence interval
    """

    mae_conf_results = {}

    samples = len(predictions)
    for algorithm,predictions in predictions.items():
        dist = [np.linalg.norm(actual - prediction)**2 for actual, prediction in predictions]
        rmse = np.sqrt(np.mean(dist))
        std = np.std(dist)
        confidence_interval  = 1.960 * std / np.sqrt(len(dist)) 
        mae_conf_results[algorithm] = np.round([rmse,confidence_interval],2)

    return mae_conf_results



def initialise_localisation_model(model:const.Model,training_data_filepath,prior=None):


    if model is const.Model.GAUSSIAN:
        return models.GaussianProcessModel(training_data_filepath,prior=prior,cell_size=1)
    elif model is const.Model.GAUSSIANKNN:
        return models.GaussianKNNModel(training_data_filepath,prior=prior,cell_size=1)
    elif model is const.Model.GAUSSIANMINMAX:
        return models.GaussianMinMaxModel(training_data_filepath,prior=prior,cell_size=1)
    elif model is const.Model.KNN:
        return models.KNN(training_data_filepath,)
    elif model is const.Model.WKNN:
        return models.WKNN(training_data_filepath,)
    elif model is const.Model.PROPOGATION:
        return models.PropagationModel(training_data_filepath,const.PROPAGATION_CONSTANT)
    elif model is const.Model.PROXIMITY:
        return models.ProximityModel(training_data_filepath)
    else:
        raise ValueError("Not a supported model")


def get_localisation_predictions(models, training_data_filepath, evaluation_data_filepath,filtering,prior):

    models = {model.value:initialise_localisation_model(model,training_data_filepath,prior) for model in models}

    if prior is const.Prior.LOCAL:
        predictions = {name: run_convergence_localisation_on_file(evaluation_data_filepath,model,filtering) for name,model in models.items()}
    elif prior is const.Prior.UNIFORM:
        predictions = {name: run_localisation_on_file(evaluation_data_filepath,model,filtering) for name,model in models.items()}

    predictions = dict(sorted(predictions.items(), key = lambda x:x[0]))

    return predictions



def run_filter_comparison(models,training_data_filepath, evaluation_data_filepath,prior):

    filtered = get_localisation_predictions(models,training_data_filepath,evaluation_data_filepath,True,prior)
    non_filtered = get_localisation_predictions(models,training_data_filepath,evaluation_data_filepath,False,prior)

    return {"Filtered": filtered,"Non-Filtered":non_filtered}



def predict_all_models(training_data_filepath,evaluation_data_filepath,prior,filtering):

    models = [model for model in const.Model]
    return get_localisation_predictions(models,training_data_filepath,evaluation_data_filepath,prior,filtering)


def filter_all_models_plot(training_data_filepath,evaluation_data_filepath):
    filtered_predictions = predict_all_models(training_data_filepath,evaluation_data_filepath,True,const.Prior.UNIFORM)
    non_filtered_predictions = predict_all_models(training_data_filepath,evaluation_data_filepath,False,const.Prior.UNIFORM)

    filtered_results = mae_confidence_interval(filtered_predictions)
    non_filtered_results = mae_confidence_interval(non_filtered_predictions)

    print("algorithm      :  filtered     :  non_filtered")
    for i,algorithm in enumerate(filtered_results.keys()):
        print(f"{algorithm:15}: {filtered_results[algorithm][0]:5} ± {filtered_results[algorithm][1]:5} : {non_filtered_results[algorithm][0]:5} ± {non_filtered_results[algorithm][1]:5}")


def prior_all_models_plot(training_data_filepath,evaluation_data_filepath):
    uniform_predictions = predict_all_models(training_data_filepath,evaluation_data_filepath,True,const.Prior.UNIFORM)
    local_predictions = predict_all_models(training_data_filepath,evaluation_data_filepath,True,const.Prior.LOCAL)

    uniform_results = mae_confidence_interval(uniform_predictions)
    local_results = mae_confidence_interval(local_predictions)

    print("algorithm      :  uniform     :  local")
    for i,algorithm in enumerate(uniform_results.keys()):
        print(f"{algorithm:15}: {uniform_results[algorithm][0]:5} ± {uniform_results[algorithm][1]:5} : {local_results[algorithm][0]:5} ± {local_results[algorithm][1]:5}")


def all_models_plot(training_data_filepath,evaluation_data_filepath):


    predictions = predict_all_models(training_data_filepath,evaluation_data_filepath,False,const.Prior.UNIFORM)

    mae = mae_confidence_interval(predictions)
    rmse = rmse_confidence_interval(predictions)

    print("algorithm      :  mae         :  rmse ")
    for i,algorithm in enumerate(predictions.keys()):
        print(f"{algorithm:15}: {mae[algorithm][0]:5} ± {mae[algorithm][1]:5} : {rmse[algorithm][0]:5} ± {rmse[algorithm][1]:5}")

    plot_evaluation_metric(mae,"mae")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Evaluation Wanted")
    parser.add_argument("training_file", help="The file with the training data in it.")
    parser.add_argument("evaluation_file", help="The file with the evaluation data in it.")
    args = parser.parse_args()

    modes = ["eval","all","prior","filter"]
    if args.mode not in modes:
        print("Mode should be in " + ",".join(modes))

    else:
        training_filepath = Path(args.training_file)
        evaluation_filepath = Path(args.evaluation_file)
        if args.mode == "eval":
            all_models_plot(training_filepath,evaluation_filepath)
        if args.mode == "all":
            all_models_plot(training_filepath,evaluation_filepath)
            prior_all_models_plot(training_filepath,evaluation_filepath)
            filter_all_models_plot(training_filepath,evaluation_filepath)
        elif args.mode == "prior":
            prior_all_models_plot(training_filepath,evaluation_filepath)
        elif args.mode == "filter":
            filter_all_models_plot(training_filepath,evaluation_filepath)

if __name__ == "__main__":
    main()