import os
import yaml

# method = 'classic'
# method = 'hankel'
# method = 'highorder'
method = 'bagopt'


for dataset in ['ODE_Lorenz', 'PDE_KS']:

    files = os.listdir(f'tuning_config/{method}/')
    files = sorted([f for f in files if f.startswith(f"config_{dataset}_") and f.endswith(".yaml")])

    # Optimize hyper-parameters for each config file
    for file in files:
        os.system(f'python optimize_parameters.py --config-path tuning_config/{method}/{file}')

    tune_folders = sorted(os.listdir(f'../../results/tune_results/DMD_{method}/{dataset}/'))
    for ii, fold in enumerate(tune_folders):

        # Get the best config file from the tuning results
        last_run = os.listdir(f'../../results/tune_results/DMD_{method}/{dataset}/{fold}/')[-1]
        os.makedirs(f'config/{method}/', exist_ok=True)
        yaml_file = sorted(os.listdir(f'../../results/tune_results/DMD_{method}/{dataset}/{fold}/{last_run}/'))[0]
        os.system(f'cp ../../results/tune_results/DMD_{method}/{dataset}/{fold}/{last_run}/{yaml_file} config/{method}/.')

        # Change the model name in the config file for consistency
        with open(f'config/{method}/{yaml_file}', 'r') as f:
            config = yaml.safe_load(f)

        config['model']['name'] = 'DMD'
        
        with open(f'config/{method}/{yaml_file}', 'w') as f:
            yaml.dump(config, f)

        # Run the model with the optimized config
        os.system(f'python run.py config/{method}/{yaml_file}')