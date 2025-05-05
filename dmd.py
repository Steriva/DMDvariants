import numpy as np
from typing import List, Optional, Dict
from pydmd import DMD, HODMD, BOPDMD, ParametricDMD
from ezyrb import POD, RBF
from pydmd.preprocessing import hankel_preprocessing
import warnings
from ctf4science.data_module import get_config

class DMD4CTF:
    """
    TO WRITE
    """

    def __init__(self, config: Dict, pair_id: int, train_data: List[np.ndarray], train_time: np.ndarray, check_svd: Optional[bool] = True):

        self.train_data = np.array(train_data) # Shape (num_samples, num_spatial_features, num_timesteps)

        self.method = config['model']['method']
        self.train_time = train_time
        self.dt = train_time[1] - train_time[0] # Assuming uniform time steps

        # Load dataset metadata
        _config_dataset = get_config(config['dataset']['name'])
        dataset_metadata = _config_dataset['metadata']

        self.spatial_dimension = dataset_metadata['spatial_dimension']
        assert self.train_data.shape[1] == self.spatial_dimension, "Training data has shape {} but expected shape is {}".format(self.train_data.shape[1], self.spatial_dimension)
        
        self.pair_id = pair_id
        self.matrix_start_index = dataset_metadata['matrix_start_index'][f'X{pair_id}test.mat']
        self.test_shape = dataset_metadata['matrix_shapes'][f'X{pair_id}test.mat']

        self.rank = config['model']['rank']
        if check_svd:
            self._check_svd_residual_energy(self.rank)

        # Set the DMD options based on the method
        self.delay = config['model']['delay'] if 'delay' in config['model'] else None
        self.num_trials = config['model']['num_trials'] if 'num_trials' in config['model'] else 0
        self.eig_constraints = {*config['model']['eig_constraints']} if 'eig_constraints' in config['model'] else None

        # Set the parametric options based on the pair_id and size of the training data
        if pair_id == 8:
            self.parametric = {
                'mode': config['model']['parametric'] if 'parametric' in config['model'] else 'monolithic',
                'train_params': np.array([1,2,4]) if len(self.train_data) > 2 else np.array([1,4]),
                'test_params': np.array([3]) if len(self.train_data) > 2 else np.array([2])
            }
        elif pair_id == 9:
            self.parametric = {
                'mode': config['model']['parametric'] if 'parametric' in config['model'] else 'monolithic',
                'train_params': np.array([1,2,3]) if len(self.train_data) > 2 else np.array([1,2]),
                'test_params': np.array([4]) if len(self.train_data) > 2 else np.array([3])
            }
        else:
            self.parametric = None

    def _check_svd_residual_energy(self, rank: int):
        """
        Compute the SVD of the training data.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for SVD")

        # Concatenate the training data
        concatenated_data = np.concatenate(self.train_data, axis=1)
        assert concatenated_data.shape[0] == self.spatial_dimension, "Concatenated data must have the same number of features as training data"

        # Compute the SVD
        _, sing_vals, _ = np.linalg.svd(concatenated_data, full_matrices=False)

        # Compute the residual energy
        residual_energy = np.sum(sing_vals[:rank]**2) / np.sum(sing_vals**2)

        if residual_energy <= 0.9:
            print("Warning: The residual energy of the SVD is {} <= 0.9 for rank {}. This may indicate that the rank is too low.".format(residual_energy, rank))
        elif np.isclose(residual_energy, 1.0):
            print("Warning: The residual energy of the SVD is {} for rank {}. This may indicate that the rank is too high.".format(residual_energy, rank))

    def initialize(self):
        """
        Set the DMD model based on the method specified in the config.
        """
        if self.method == 'classic':
            self.dmd = ClassicDMD(svd_rank = self.rank, parametric=self.parametric)
        elif self.method == 'hankel':
            self.dmd = HankelDMD(svd_rank = self.rank, delay=self.delay, parametric=self.parametric)
        elif self.method == 'highorder':
            self.dmd = HighOrderDMD(svd_rank = self.rank, delay=self.delay, parametric=self.parametric)
        elif self.method == 'bagopt':
            self.dmd = BaggingOptimisedDMD(svd_rank= self.rank, delay=self.delay, num_trials=self.num_trials, eig_constraints=self.eig_constraints, parametric=self.parametric)
        else:
            raise ValueError("Invalid DMD method specified in the config: {}".format(self.method))

    def train(self):
        """
        Train the DMD model.
        """
        self.dmd.train(self.train_data, self.train_time)
        
    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        Predict the future data using the trained DMD model.
        """
        return self.dmd.predict(prediction_timesteps)

class ClassicDMD():
    def __init__(self, svd_rank: int, parametric: Optional[dict] = None):
        """
        Initialize the DMD model.
        """
        
        self.parametric = parametric

        if parametric is None:
            self.dmd = DMD(svd_rank=svd_rank)
        else:
            if parametric['mode'] == 'monolithic':
                self._dmd = DMD(svd_rank=-1)
            elif parametric['mode'] == 'partitioned':
                self._dmd = [DMD(svd_rank=-1) for _ in range(len(parametric['train_params']))]

            self.pod = POD(rank = svd_rank, method='randomized_svd')
            self.interpolator = RBF()

            self.dmd = ParametricDMD(self._dmd, self.pod, self.interpolator)

    def train(self, train_data: np.ndarray, train_time: np.ndarray):
        """
        Train the DMD model.
        """

        warnings.filterwarnings("ignore")
        self.spatial_dimension = train_data.shape[1]
        self.dt = train_time[1] - train_time[0] # Assuming uniform time steps

        if self.parametric is not None:
            assert len(train_data) == len(self.parametric['train_params']), "Number of training data must match the number of training parameters"
            self.dmd.fit(train_data, self.parametric['train_params'])
        else:

            self.dmd.fit(train_data[0])
        
        self.dmd.original_time['t0'] = train_time[0]
        self.dmd.original_time['dt'] = self.dt
        self.dmd.original_time['tend'] = train_time[-1]

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        
        """
        
        # Set the DMD time for prediction
        self.dmd.dmd_time['t0'] = prediction_timesteps[0]
        self.dmd.dmd_time['dt'] = self.dt
        self.dmd.dmd_time['tend'] = prediction_timesteps[-1]

        if self.parametric is not None:
            self.dmd.parameters = self.parametric['test_params']
        
        pred_data = self.dmd.reconstructed_data

        if self.parametric is not None: # assuming only one test parameter
            pred_data = pred_data[0]

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"

        if pred_data.shape[1] > prediction_timesteps.shape[0]:
            pred_data = pred_data[:, :prediction_timesteps.shape[0]]
            
        return pred_data

class HankelDMD():
    """
    Initializes the DMD model with Hankel preprocessing.
    """
    def __init__(self, svd_rank: int, delay: int, parametric: Optional[dict] = None):
        """
        
        """
        
        self.parametric = parametric
        self.delay = delay

        if parametric is None:
            self.dmd = hankel_preprocessing(DMD(svd_rank=svd_rank), d=delay)
        else:
            if parametric['mode'] == 'monolithic':
                self._dmd = hankel_preprocessing(DMD(svd_rank=-1), d=delay)
            elif parametric['mode'] == 'partitioned':
                self._dmd = [hankel_preprocessing(DMD(svd_rank=-1), d=delay) for _ in range(len(parametric['train_params']))]

            self.pod = POD(rank = svd_rank, method='randomized_svd')
            self.interpolator = RBF()

            self.dmd = ParametricDMD(self._dmd, self.pod, self.interpolator)

    def train(self, train_data: np.ndarray, train_time: np.ndarray):
        """
        Train the DMD model.
        """

        warnings.filterwarnings("ignore")
        self.spatial_dimension = train_data.shape[1]
        self.dt = train_time[1] - train_time[0] # Assuming uniform time steps

        if self.parametric is not None:
            assert len(train_data) == len(self.parametric['train_params']), "Number of training data must match the number of training parameters"
            self.dmd.fit(train_data, self.parametric['train_params'])
        else:

            self.dmd.fit(train_data[0])
            
        self.dmd.original_time['t0'] = train_time[0]
        self.dmd.original_time['dt'] = self.dt
        self.dmd.original_time['tend'] = train_time[-1]

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """

        """
        
        # Set the DMD time for prediction
        self.dmd.dmd_time['t0'] = prediction_timesteps[0]
        self.dmd.dmd_time['dt'] = self.dt
        self.dmd.dmd_time['tend'] = prediction_timesteps[-1]

        if self.parametric is not None:
            self.dmd.parameters = self.parametric['test_params']

        pred_data = self.dmd.reconstructed_data
        
        if self.parametric is not None: # assuming only one test parameter
            pred_data = pred_data[0]

        pred_data = pred_data[:, :-self.delay+1]

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"
        
        if pred_data.shape[1] > prediction_timesteps.shape[0]:
            pred_data = pred_data[:, :prediction_timesteps.shape[0]]
            
        return pred_data

class HighOrderDMD():
    """
    
    """
    def __init__(self, svd_rank: int, delay: int, parametric: Optional[dict] = None):
        """
        
        """
        self.parametric = parametric

        if parametric is None:    
           self.dmd = HODMD(svd_rank=svd_rank, d=delay)
        else:
            if parametric['mode'] == 'monolithic':
                self._dmd = HODMD(svd_rank=-1, d=delay)
            elif parametric['mode'] == 'partitioned':
                self._dmd = [HODMD(svd_rank=-1, d=delay) for _ in range(len(parametric['train_params']))]

            self.pod = POD(rank = svd_rank, method='randomized_svd')
            self.interpolator = RBF()

            self.dmd = ParametricDMD(self._dmd, self.pod, self.interpolator)

    def train(self, train_data: np.ndarray, train_time: np.ndarray):
        """
        Train the DMD model.
        """

        warnings.filterwarnings("ignore")
        self.spatial_dimension = train_data.shape[1]
        self.dt = train_time[1] - train_time[0] # Assuming uniform time steps

        if self.parametric is not None:
            assert len(train_data) == len(self.parametric['train_params']), "Number of training data must match the number of training parameters"
            self.dmd.fit(train_data, self.parametric['train_params'])
        else:
            self.dmd.fit(train_data[0])
            
        self.dmd.original_time['t0'] = train_time[0]
        self.dmd.original_time['dt'] = self.dt
        self.dmd.original_time['tend'] = train_time[-1]

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        
        """
        # Set the DMD time for prediction
        self.dmd.dmd_time['t0'] = prediction_timesteps[0]
        self.dmd.dmd_time['dt'] = self.dt
        self.dmd.dmd_time['tend'] = prediction_timesteps[-1]+1e-12
        
        if self.parametric is not None:
            self.dmd.parameters = self.parametric['test_params']
        
        pred_data = self.dmd.reconstructed_data

        if self.parametric is not None: # assuming only one test parameter
            pred_data = pred_data[0]

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"
        
        if pred_data.shape[1] > prediction_timesteps.shape[0]:
            pred_data = pred_data[:, :prediction_timesteps.shape[0]]
            
        return pred_data

class BaggingOptimisedDMD():
    """
    
    """
    def __init__(self, svd_rank: int,
                 delay: Optional[int] = None, num_trials: Optional[int] = 0, eig_constraints: Optional[set] = None,
                 parametric: Optional[dict] = None):
        """
        
        """

        self.parametric = parametric

        if parametric is None:
            _dmd = BOPDMD(svd_rank=svd_rank, 
                        num_trials=num_trials, 
                        eig_constraints=eig_constraints)

            if delay is not None:
                self.dmd = hankel_preprocessing(_dmd, d=delay)
                self.delay = delay
            else:
                self.dmd = _dmd
                self.delay = None
        else:
            
            self.rom = POD(rank=svd_rank, method='randomized_svd')
            self.interpolator = RBF()

            # if delay is not None:
            #     self._dmd = hankel_preprocessing(BOPDMD(svd_rank=-1,  num_trials=num_trials, eig_constraints=eig_constraints, varpro_opts_dict={'verbose': False}), 
            #                                      d=delay) if parametric['mode'] == 'monolithic' else [hankel_preprocessing(BOPDMD(svd_rank=-1, num_trials=num_trials, eig_constraints=eig_constraints, varpro_opts_dict={'verbose': False}), d=delay) for _ in range(len(parametric['train_params']))]
            #     self.delay = delay
            # else:
            #     self._dmd = BOPDMD(svd_rank=-1,  num_trials=num_trials, eig_constraints=eig_constraints, varpro_opts_dict={'verbose': False}) if parametric['mode'] == 'monolithic' else [BOPDMD(svd_rank=-1,  num_trials=num_trials, eig_constraints=eig_constraints, varpro_opts_dict={'verbose': False}) for _ in range(len(parametric['train_params']))]
            #     self.delay = None
            base_dmd = lambda: BOPDMD(
                svd_rank=-1,
                num_trials=num_trials,
                eig_constraints=eig_constraints,
                varpro_opts_dict={'verbose': True}
            )

            if parametric['mode'] == 'monolithic':
                if delay is not None:
                    self._dmd = hankel_preprocessing(base_dmd(), d=delay)
                else:
                    self._dmd = base_dmd()
            else:
                n_models = len(parametric['train_params'])
                if delay is not None:
                    self._dmd = [hankel_preprocessing(base_dmd(), d=delay) for _ in range(n_models)]
                else:
                    self._dmd = [base_dmd() for _ in range(n_models)]

            self.delay = delay if delay is not None else None


            self.dmd = ParametricDMD(self._dmd, self.rom, self.interpolator) # , dmd_fit_args={'t': parametric['train_time']})
            
        self.num_trials = num_trials
        self.eig_constraints = eig_constraints

    def train(self, train_data: np.ndarray, train_time: np.ndarray):
        """
        Train the DMD model.
        """

        warnings.filterwarnings("ignore")
        self.spatial_dimension = train_data.shape[1]

        if self.delay is not None:
            train_time = train_time[:-self.delay+1]

        if self.parametric is not None:
            dmd_fit_kwargs = {'t': train_time}
            self.dmd._dmd_fit_kwargs = dmd_fit_kwargs

            self.dmd.fit(train_data, self.parametric['train_params'])
        else:

            self.dmd.fit(train_data[0], t=train_time)

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        
        """
        if self.parametric is not None:
            self.dmd.parameters = self.parametric['test_params']
            pred_data = self.forecast_parametric(prediction_timesteps)[0]

        else:
            pred_data = self.dmd.forecast(prediction_timesteps)

            if self.num_trials > 0:
                pred_data = pred_data[0]

            if self.delay is not None:
                pred_data = pred_data[:self.spatial_dimension]

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"

        return pred_data

    def forecast_parametric(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """

        """
        if self.dmd.is_partitioned:
            forecasted_modal_coefficients = list()
            for _dmd in self._dmd:
                if self.num_trials > 0:
                    forecasted_modal_coefficients.append(_dmd.forecast(prediction_timesteps)[0])
                else:
                    forecasted_modal_coefficients.append(_dmd.forecast(prediction_timesteps))
            forecasted_modal_coefficients =  np.vstack(forecasted_modal_coefficients)
        else:
            forecasted_modal_coefficients = self._dmd.forecast(prediction_timesteps)

            if self.num_trials > 0:
                forecasted_modal_coefficients = forecasted_modal_coefficients[0]    
        
        if self.delay is not None:
            _extracted_modal_idx = np.hstack([
                np.arange(self.dmd._spatial_pod.rank) + self.delay*mu*self.dmd._spatial_pod.rank 
                for mu in range(self.parametric['train_params'].shape[0])
            ])
            forecasted_modal_coefficients = forecasted_modal_coefficients[_extracted_modal_idx]

        interpolated_modal_coefficients = (
            self.dmd._interpolate_missing_modal_coefficients(
                forecasted_modal_coefficients
            )
        )

        return np.apply_along_axis(
            self.dmd._spatial_pod.expand, 1, interpolated_modal_coefficients
        )