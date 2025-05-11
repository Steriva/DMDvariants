import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from ctf4science.data_module import load_validation_dataset, get_validation_prediction_timesteps, parse_pair_ids, get_validation_training_timesteps
from ctf4science.eval_module import evaluate_custom
from models.DMDvariants.dmd import DMD4CTF

# Delete results directory - used for storing batch_results
file_dir = Path(__file__).parent

# Notes:
# K value larger than 10 results in invalid spatio-temporal loss
# Currently just overwriting config file and results file to save space
# Currently using init_data in hyperparameter optimization
# Currently not counting init_data in train_split amount

def main(config_path: str) -> None:
    """
    Main function to run the naive baseline model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    The evaluation function evaluates on validation data obtained from training data.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}_{config['model']['method']}"

    # batch_id is from optimize_parameters.py
    batch_id = f"hyper_opt_{config['model']['batch_id']}"
 
    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Generate training and validation splits (and burn-in matrix when applicable) 
        train_split = config['model']['train_split']
        # Load sub-dataset - transpose is used since DMD requires data in (space, time) format
        train_data, val_data, init_data = load_validation_dataset(dataset_name, pair_id, train_split, transpose=True)
        prediction_timesteps = get_validation_prediction_timesteps(dataset_name, pair_id, train_split)
        train_timesteps = get_validation_training_timesteps(dataset_name, pair_id, train_split)[0]

        # Initialize the model with the config and train_data
        dmd_model = DMD4CTF(config, pair_id, train_data, train_timesteps, check_svd=False)
        dmd_model.initialize()

        if dataset_name == 'PDE_KS':
            compress_data = True
        else:
            compress_data = False
        dmd_model.train(compress_data=compress_data)

        # Generate predictions
        pred_data = dmd_model.predict(prediction_timesteps)
        pred_data = pred_data.T.real  # Transpose to (time, space) format
        

        # Evaluate predictions using default metrics
        val_data = val_data.T # Transpose to (time, space) format
        results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

    # Save aggregated batch results
    results_file = file_dir / f"results_{config['model']['batch_id']}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)