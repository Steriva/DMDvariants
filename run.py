import argparse
import yaml
from pathlib import Path
import datetime
from typing import List, Dict, Any

from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_prediction_timesteps, get_training_timesteps
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from models.DMDvariants.dmd import DMD4CTF
import os

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
    
def main(config_path: str) -> None:
    """
    Executes the main workflow for running a DMD-based model on specified sub-datasets.
    
    This function performs the following steps:
        1. Loads the experiment configuration from a YAML file.
        2. Parses the dataset and retrieves the list of sub-dataset pair IDs.
        3. Initializes model and batch identifiers based on configuration parameters.
        4. Iterates over each sub-dataset pair:
            - Loads training and initialization data.
            - Retrieves training and prediction timesteps.
            - Initializes and trains the DMD model.
            - Generates predictions for the specified timesteps.
            - Evaluates predictions using default metrics.
            - Saves results and evaluation metrics.
            - Generates and saves applicable visualizations.
        5. Aggregates and saves batch results for all processed sub-datasets.

    Args:
        config_path (str): Path to the YAML configuration file specifying dataset, model, and experiment parameters.
    Returns:
        None
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    # Model name
    model_name = f"{config['model']['method']}{config['model']['name']}"
    
    # Add rank to batch_id
    batch_id = f"batch_rank{config['model']['rank']}"

    if 'delay' in config['model']:
        batch_id = f"{batch_id}_delay{config['model']['delay']}"
    if 'num_trials' in config['model']:
        batch_id = f"{batch_id}_numtrials{config['model']['num_trials']}"

    # Define the name of the output folder for your batch
    batch_id = f"{batch_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:

        # Load sub-dataset - transpose is used since DMD requires data in (space, time) format
        train_data, initialization_data = load_dataset(dataset_name, pair_id, transpose=True)

        # Load metadata
        train_timesteps = get_training_timesteps(dataset_name, pair_id)[0] # extract first element
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)

        # Initialize the model with the config and train_data
        print(f"Running {model_name} on {dataset_name} pair {pair_id}")

        dmd_model = DMD4CTF(config, pair_id, train_data, train_timesteps, check_svd = False)
        dmd_model.initialize()

        # Train the DMD model
        if dataset_name == 'ODE_Lorenz' or dataset_name == 'Lorenz_Official':
            compress_data = False
        else:
            compress_data = True
        dmd_model.train(compress_data=compress_data)

        # Generate predictions
        pred_data = dmd_model.predict(prediction_timesteps)

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

        print(' ')
        print(results)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)