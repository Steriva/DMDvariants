import argparse
import yaml
from pathlib import Path
import datetime
from typing import List, Dict, Any

from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_config
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
    
class PlotLorenz:
    def __init__(self):
        pass

    def plot_Lorenz_trajectory(self, axs, snap, colors=['r', 'b', 'g'], linestyle='-'):
        assert len(axs) == snap.shape[0], "Number of axes must match number of trajectories"

        t = np.arange(snap.shape[1])

        for i, ax in enumerate(axs):
            ax.plot(t, snap[i], color=colors[i], linestyle=linestyle)

    def compare_prediction(self, test_data, pred_data, figsize=(18, 6)):

        fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True)

        self.plot_Lorenz_trajectory(axs, test_data, colors=['r', 'b', 'g'], linestyle='-')
        self.plot_Lorenz_trajectory(axs, pred_data, colors=['y', 'm', 'c'], linestyle='--')

        for ii, ax in enumerate(axs):
            ax.set_xlabel("Time step")
            ax.legend(['True', 'Predicted'])
            ax.grid(True)
            ax.set_title("Trajectory {}".format(ii + 1))
        plt.tight_layout()

        return fig
    
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

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['method']}{config['model']['name']}_rank{config['model']['rank']}"

    if config['model']['method'] == 'hankel' or config['model']['method'] == 'highorder':
        model_name = f"{model_name}_delay{config['model']['delay']}"
    elif config['model']['method'] == 'bagopt':
        model_name = f"{model_name}_numtrials{config['model']['num_trials']}"

    # Generate a unique batch_id for this run, you can add any descriptions you want
    #   e.g. f"batch_{learning_rate}_"
    batch_id = f"batch_"+model_name
    
    # Define the name of the output folder for your batch
    batch_id = f"{batch_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'sub_datasets': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        train_data, test_data, initialization_data = load_dataset(dataset_name, pair_id)

        print(f"Running {model_name} on {dataset_name} pair {pair_id}")
        
        # Initialize the model with the config and train_data
        if config['model']['method'] == 'classic': 
            dmd_model = ClassicDMD(config, train_data[0]) # train_data[0] is the data matrix
        elif config['model']['method'] == 'hankel': 
            dmd_model = HankelDMD(config, train_data[0]) # train_data[0] is the data matrix
        elif config['model']['method'] == 'highorder':
            dmd_model = HighOrderDMD(config, train_data[0]) # train_data[0] is the data matrix
        elif config['model']['method'] == 'bagopt':
            dmd_model = BaggingOptimisedDMD(config, train_data[0]) # train_data[0] is the data matrix
        else:
            raise ValueError(f"Unknown model method: {config['model']['method']}")

        # Select if reconstruction or prediction
        _config_dataset = get_config(dataset_name)
        pair = next((p for p in _config_dataset['pairs'] if p['id'] == pair_id), None)
        
        # Generate predictions
        if sum([meth == 'reconstruction' for meth in pair['metrics']]) > 0:
            # Use the reconstruction method
            pred_data = dmd_model.reconstruct(test_data)
        else:
            # Use the prediction method
            pred_data = dmd_model.predict(test_data)
        
        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, test_data, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        results_for_yaml = {key: float(value) for key, value in results.items()}
        batch_results['sub_datasets'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type)

        # Save the contours/trajectories
        if dataset_name == 'PDE_KS':
            plot_KS = PlotKS(aspect=0.1)
            fig = plot_KS.compare_prediction(test_data, pred_data,
                                            cbar_options={'ticks': 5})
            fig.savefig(results_directory / f"contour_{pair_id}.png", dpi=200)
        elif dataset_name == 'ODE_Lorenz':
            plot_Lorenz = PlotLorenz()
            fig = plot_Lorenz.compare_prediction(test_data, pred_data)
            fig.savefig(results_directory / f"trajectory_{pair_id}.png", dpi=200)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)