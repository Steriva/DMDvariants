import numpy as np
from typing import List, Optional, Dict
from pydmd import DMD, HODMD, BOPDMD
from pydmd.preprocessing import hankel_preprocessing
import warnings
from ctf4science.data_module import get_config

class DMD4CTF:
    """
    TO WRITE
    """

    def __init__(self, config: Dict, pair_id: int, train_data: List[np.ndarray], dt : Optional[float] = None):
        """
        
        """
        self.train_data = train_data[0]

        # Load dataset metadata
        _config_dataset = get_config(config['dataset']['name'])
        dataset_metadata = _config_dataset['metadata']

        self.spatial_dimension = dataset_metadata['spatial_dimension']

        self.pair_id = pair_id
        self.matrix_start_index = dataset_metadata['matrix_start_index'][f'X{pair_id}test.mat']
        self.test_shape = dataset_metadata['matrix_shapes'][f'X{pair_id}test.mat']

        if dt is not None:
            self.dt = dt
        else:
            self.dt = dataset_metadata['delta_t']
        self.spatial_dimension = self.train_data.shape[0]

    def _set_original_time(self):
        """
        Set the original time for the DMD model.
        """
        self.dmd.original_time['t0'] = 0
        self.dmd.original_time['dt'] = self.dt
        self.dmd.original_time['tend'] = (self.train_data.shape[1]) * self.dt

    def _set_dmd_time(self, prediction_timesteps):
        """
        Set the DMD time for the DMD model.
        """
        self.dmd.dmd_time['t0'] = (self.matrix_start_index+1) * self.dt
        self.dmd.dmd_time['dt'] = self.dt
        self.dmd.dmd_time['tend'] = (self.matrix_start_index + prediction_timesteps.shape[0]) * self.dt

        assert self.dmd.dmd_timesteps.shape[0] == prediction_timesteps.shape[0], "DMD time steps must match prediction time steps: {} vs {}".format(self.dmd.dmd_timesteps.shape[0], prediction_timesteps.shape[0])

class ClassicDMD(DMD4CTF):
    def __init__(self, config, pair_id, train_data, dt = None):
        super().__init__(config, pair_id, train_data, dt)

        # Train the DMD model
        warnings.filterwarnings("ignore")
        self.rank = config['model']['rank']
        self.dmd = DMD(svd_rank=self.rank)
        self.dmd.fit(self.train_data)
        self._set_original_time()

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self._set_dmd_time(prediction_timesteps)

        pred_data = self.dmd.reconstructed_data.real

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"

        return pred_data

class HankelDMD(DMD4CTF):
    """
    TO WRITE
    """
    def __init__(self, config: Dict, pair_id: int, train_data: List[np.ndarray]):
        """
        
        """
        super().__init__(config, pair_id, train_data)

        # Train the DMD model
        warnings.filterwarnings("ignore")
        self.rank = config['model']['rank']
        self.delay = config['model']['delay']
        self.dmd = hankel_preprocessing(DMD(svd_rank=self.rank), d=self.delay)
        self.dmd.fit(self.train_data)
        self._set_original_time()

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self._set_dmd_time(prediction_timesteps)

        pred_data = self.dmd.reconstructed_data[:, :-self.delay+1].real

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"
        assert pred_data.shape[1] == prediction_timesteps.shape[0], "Predicted data must have the same number of time steps as training data"

        return pred_data

class HighOrderDMD(DMD4CTF):
    """
    
    """
    def __init__(self, config: Dict, pair_id: int, train_data: List[np.ndarray]):
        """
        
        """
        super().__init__(config, pair_id, train_data)

        # Train the DMD model
        warnings.filterwarnings("ignore")
        self.rank = config['model']['rank']
        self.delay = config['model']['delay']
        self.dmd = HODMD(svd_rank=self.rank, d=self.delay)
        self.dmd.fit(self.train_data)
        self._set_original_time()

    def _set_dmd_time(self, prediction_timesteps):
        """
        Set the DMD time for the DMD model.
        """
        self.dmd.dmd_time['t0'] = (self.matrix_start_index+1) * self.dt
        self.dmd.dmd_time['dt'] = self.dt
        self.dmd.dmd_time['tend'] = (self.matrix_start_index + prediction_timesteps.shape[0]+1e-12) * self.dt

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self._set_dmd_time(prediction_timesteps)

        pred_data = self.dmd.reconstructed_data

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"
        assert pred_data.shape[1] == prediction_timesteps.shape[0], "Predicted data must have the same number of time steps as training data: {} vs {}".format(pred_data.shape[1], prediction_timesteps.shape[0])

        return pred_data

class BaggingOptimisedDMD(DMD4CTF):
    """
    
    """
    def __init__(self, config: Dict, pair_id: int, train_data: List[np.ndarray]):
        """
        
        """
        super().__init__(config, pair_id, train_data)

        # Train the DMD model
        warnings.filterwarnings("ignore")
        self.rank = config['model']['rank']
        self.delay = config['model']['delay'] if 'delay' in config['model'] else None
        self.num_trials = config['model']['num_trials'] if 'num_trials' in config['model'] else 0
        self.eig_constraints = {*config['model']['eig_constraints']} if 'eig_constraints' in config['model'] else None

        train_time = np.arange(self.train_data.shape[1]) * self.dt

        _dmd = BOPDMD(svd_rank=self.rank,
                            num_trials=self.num_trials,
                            eig_constraints=self.eig_constraints)

        if self.delay is not None:
            self.dmd = hankel_preprocessing(_dmd, d=self.delay)
            delay_t = train_time[:-self.delay+1]
            self.dmd.fit(self.train_data, t=delay_t)
        else:
            self.dmd = _dmd
            self.dmd.fit(self.train_data, t=train_time)

    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        pred_data = self.dmd.forecast(prediction_timesteps)

        if self.num_trials > 0:
            pred_data = pred_data[0]

        if self.delay is not None:
            pred_data = pred_data[:self.spatial_dimension]

        assert pred_data.shape[0] == self.spatial_dimension, "Predicted data must have the same number of features as training data"

        return pred_data