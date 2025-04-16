import argparse
import yaml
from pathlib import Path
import datetime
from typing import List, Dict, Any

from ctf4science.data_module import load_dataset
from ctf4science.eval_module import evaluate, save_results
from models.HankelDMD.hank_dmd import HankelDMD
import os

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plot_KS_results(train_data, test_data, pred_data, levels=100, cmap=cm.jet):
    x = np.linspace(0, 32 * np.pi, train_data.shape[0])
    assert train_data.shape[0] == test_data.shape[0] == pred_data.shape[0], "Space dimensions do not match"

    ttrain = np.arange(0, train_data.shape[1])
    ttest = np.arange(0, test_data.shape[1])
    tpred = np.arange(0, pred_data.shape[1])

    xgrid_train, tgrid_train = np.meshgrid(x, ttrain)
    xgrid_test, tgrid_test   = np.meshgrid(x, ttest)
    xgrid_pred, tgrid_pred   = np.meshgrid(x, tpred)

    fig, axs = plt.subplots(1, 3, figsize=(14,8))

    levels = np.linspace( min([np.min(train_data), np.min(test_data)]),
                          max([np.max(train_data), np.max(test_data)]),
                          levels)
    
    cont_train = axs[0].contourf(xgrid_train, tgrid_train, train_data.T, levels=levels, cmap=cmap)
    cont_test  = axs[1].contourf(xgrid_test, tgrid_test, test_data.T, levels=levels, cmap=cmap)
    cont_pred  = axs[2].contourf(xgrid_pred, tgrid_pred, pred_data.T, levels=levels, cmap=cmap)

    cbar_train = fig.colorbar(cont_train, ax=axs[0], orientation='horizontal', shrink=0.8)
    cbar_test  = fig.colorbar(cont_test, ax=axs[1], orientation='horizontal', shrink=0.8)
    cbar_pred  = fig.colorbar(cont_pred, ax=axs[2], orientation='horizontal', shrink=0.8)

    return fig

def plot_Lorenz_results(train_data, test_data, pred_data):
    fig, axs = plt.subplots(1, 3, figsize=(20,6))

    ttrain = np.arange(0, train_data.shape[1])
    ttest = np.arange(0, test_data.shape[1])
    tpred = np.arange(0, pred_data.shape[1])

    for ii in range(len(axs)):
        axs[ii].plot(ttrain, train_data[ii], 'b',   label='Train', color='blue')
        axs[ii].plot(ttest, test_data[ii],   'g--', label='Test', color='orange')
        axs[ii].plot(ttest, pred_data[ii],   'r-.', label='Pred', color='green')

    return fig

def parse_pair_ids(pair_id_config: Any) -> List[int]:
    """
    Parse the pair_id configuration to determine which sub-datasets to run.

    Args:
        pair_id_config (Any): The pair_id value from the config file (e.g., int, list, str, or None).

    Returns:
        List[int]: A list of pair_ids to process.

    Raises:
        ValueError: If the pair_id configuration is invalid.
    """
    if pair_id_config is None or pair_id_config == 'all':
        return list(range(1, 10))  # Assuming 9 sub-datasets; adjust as needed
    elif isinstance(pair_id_config, int):
        return [pair_id_config]
    elif isinstance(pair_id_config, str) and '-' in pair_id_config:
        start, end = map(int, pair_id_config.split('-'))
        return list(range(start, end + 1))
    elif isinstance(pair_id_config, list):
        return pair_id_config
    else:
        raise ValueError(f"Invalid pair_id configuration: {pair_id_config}")

def main(config_path: str) -> None:
    """
    Main function to run the naive baseline model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config['dataset']['name']
    model_name = f"{config['model']['name']}_rank{config['model']['rank']}"
    pair_ids = parse_pair_ids(config['dataset'].get('pair_id'))

    # Generate a unique batch_id for this run, you can add any descriptions you want
    #   e.g. f"batch_{learning_rate}_"
    batch_id = f"batch_"+model_name
    
    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'sub_datasets': []
    }

    # Load dataset configuration
    with open(os.path.join('data', dataset_name, f'{dataset_name}.yaml'), 'r') as f:
        _config = yaml.safe_load(f)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        train_data, test_data = load_dataset(dataset_name, pair_id)

        # Initialize the model with the config and train_data
        dmd_model = HankelDMD(config, train_data)

        # Select if reconstruction or prediction
        pair = next(p for p in _config['pairs'] if p['id'] == pair_id)

        # Generate predictions - TO BE CHECKED!!
        if sum([meth == 'reconstruction' for meth in pair['metrics']]) > 0:
            # Use the reconstruction method
            pred_data = dmd_model.reconstruct(test_data)
        else:
            # Use the prediction method
            pred_data = dmd_model.predict(test_data)

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, test_data, pred_data)

        # Save results for this sub-dataset
        save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Make Figures
        if dataset_name == 'PDE_KS':
            fig = plot_KS_results(train_data, test_data, pred_data)
        elif dataset_name == 'ODE_Lorenz':
            fig = plot_Lorenz_results(train_data, test_data, pred_data)

        results_dir = Path('results') / dataset_name / model_name
        fig.savefig(results_dir / f'predictions_{pair_id}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    
        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        results_for_yaml = {key: float(value) for key, value in results.items()}
        batch_results['sub_datasets'].append({
            'pair_id': pair_id,
            'metrics': results
        })

    # Save aggregated batch results
    batch_dir = Path('results') / dataset_name / model_name / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    with open(batch_dir / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)