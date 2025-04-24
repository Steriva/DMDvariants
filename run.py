import argparse
import yaml
from pathlib import Path
import datetime
from typing import List, Dict, Any

from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_prediction_timesteps
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from dmd import HankelDMD, ClassicDMD, HighOrderDMD, BaggingOptimisedDMD
import os

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class PlotKS:
    def __init__(self, aspect = 0.1, nx = 1024):
        self.aspect = aspect
        self.x = np.linspace(0, 32 * np.pi, nx)

    def plot_contour(self, ax, snap, cmap = cm.jet, levels = 100, show_ticks = True):
        t = np.arange(0, snap.shape[1])
        xgrid, tgrid = np.meshgrid(self.x, t)

        cont = ax.contourf(xgrid, tgrid, snap.T, levels=levels, cmap=cmap)

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_aspect(self.aspect)
        return cont

    def compare_prediction(self, test_data, pred_data, figsize=(10,8), levels=100, cmap=cm.jet,
                       cbar_options=None, show_ticks=True, show_titles=True):

        default_cbar_options = {
            'show': True,
            'orientation': 'horizontal',
            'shrink': 0.8
        }

        # Safely merge with user input
        if cbar_options is None:
            cbar_options = default_cbar_options
        else:
            cbar_options = {**default_cbar_options, **cbar_options}

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        if isinstance(levels, int):
            levels = np.linspace(test_data.min(), test_data.max(), levels)

        cont_test = self.plot_contour(axs[0], test_data, cmap=cmap, levels=levels, show_ticks=show_ticks)
        self.plot_contour(axs[1], pred_data, cmap=cmap, levels=levels, show_ticks=show_ticks)

        if cbar_options.get('show', True):
            cbar = fig.colorbar(
                cont_test,
                ax=axs,
                orientation=cbar_options.get('orientation', 'horizontal'),
                shrink=cbar_options.get('shrink', 0.8)
            )

            ticks = cbar_options.get('ticks')
            if ticks is not None:
                if isinstance(ticks, (np.ndarray, list)):
                    cbar.set_ticks(ticks)
                elif isinstance(ticks, int):
                    cbar.set_ticks(np.linspace(test_data.min(), test_data.max(), ticks))

            label = cbar_options.get('label')
            if label:
                cbar.set_label(label)

        if show_titles:
            axs[0].set_title('Test Data')
            axs[1].set_title('Predicted Data')

        return fig
    
def main(config_path: str) -> None:
    """ TO MODIFY
    Main function to run the ... model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    # Add rank to model name
    model_name = f"{config['model']['method']}{config['model']['name']}"
    
    # Add rank to batch_id
    batch_id = f"batch_rank{config['model']['rank']}"

    if config['model']['method'] == 'hankel' or config['model']['method'] == 'highorder':
        batch_id = f"{batch_id}_delay{config['model']['delay']}"
    elif config['model']['method'] == 'bagopt':
        if 'delay' in config['model']:
            batch_id = f"{batch_id}_delay{config['model']['delay']}"
        
        if 'num_trials' in config['model']:
            batch_id = f"{batch_id}_numtrials{config['model']['num_trials']}"
        else:
            batch_id = f"{batch_id}_numtrials0"

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

    # To be deleted in the future (embedded in visualization_module)
    if dataset_name == 'PDE_KS':
        plot_KS = PlotKS(aspect=0.1)

    # Process each sub-dataset
    for pair_id in pair_ids:

        # Load sub-dataset
        train_data, initialization_data = load_dataset(dataset_name, pair_id)

        # Load metadata (to provide forecast length)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)

        # Train model
        print(f"Running {model_name} on {dataset_name} pair {pair_id}")

        # Initialize the model with the config and train_data
        if config['model']['method'] == 'classic': 
            dmd_model = ClassicDMD(config, pair_id, train_data)
        elif config['model']['method'] == 'hankel': 
            dmd_model = HankelDMD(config, pair_id, train_data)
        elif config['model']['method'] == 'highorder':
            dmd_model = HighOrderDMD(config, pair_id, train_data)
        elif config['model']['method'] == 'bagopt':
            dmd_model = BaggingOptimisedDMD(config, pair_id, train_data)
        else:
            raise ValueError(f"Unknown model method: {config['model']['method']}")
        
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

        # Save the contours (to be later moved to visualization_module)
        if dataset_name == 'PDE_KS':
            plot_KS = PlotKS(aspect=0.1)
            from ctf4science.data_module import _load_test_data
            test_data = _load_test_data(dataset_name, pair_id)

            fig = plot_KS.compare_prediction(test_data, pred_data,
                                            cbar_options={'ticks': 5})
            fig.savefig(results_directory / f"visualizations/contour.png", dpi=200)
            print(f"Saved contour plot to {results_directory / f'visualizations/contour.png'}")
            plt.close(fig)
        print(' ')

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)