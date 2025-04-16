import numpy as np
from typing import Optional, Dict
from pydmd import DMD, HODMD, BOPDMD
from pydmd.preprocessing import zero_mean_preprocessing, hankel_preprocessing
import warnings

class ClassicDMD:
    """
    
    """

    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None):
        """
        
        """
        self.rank = config['model']['rank']
        self.zero_mean_preprocessing = config['model']['zero_mean_preprocessing']
        self.train_data = train_data

        # Train the DMD model
        warnings.filterwarnings("ignore")
        if self.zero_mean_preprocessing:
            self.dmd = zero_mean_preprocessing(DMD(svd_rank=self.rank))
        else:
            self.dmd = DMD(svd_rank=self.rank)
        self.dmd.fit(train_data)

        self.dmd.original_time['t0'] = 0
        self.dmd.original_time['dt'] = 1
        self.dmd.original_time['tend'] = train_data.shape[1]-1

    def reconstruct(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.dmd.dmd_time['t0'] = 0
        self.dmd.dmd_time['dt'] = 1
        self.dmd.dmd_time['tend'] = test_data.shape[1]-1
            
        pred_data = self.dmd.reconstructed_data

        return pred_data

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.dmd.dmd_time['t0'] = 0
        self.dmd.dmd_time['dt'] = 1
        self.dmd.dmd_time['tend'] = test_data.shape[1] + self.dmd.original_time['tend']
            
        pred_data = self.dmd.reconstructed_data

        return pred_data[:, self.train_data.shape[1]:]
    
class HankelDMD:
    """
    
    """

    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None):
        """
        
        """
        self.rank = config['model']['rank']
        self.delay = config['model']['delay']
        self.train_data = train_data

        # Train the DMD model
        # print(f"Training DMD model with rank {self.rank}")
        warnings.filterwarnings("ignore")
        self.dmd = hankel_preprocessing(DMD(svd_rank=self.rank), d=self.delay)
        self.dmd.fit(train_data)

        self.dmd.original_time['t0'] = 0
        self.dmd.original_time['dt'] = 1
        self.dmd.original_time['tend'] = train_data.shape[1]-1

    def reconstruct(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.dmd.dmd_time['t0'] = 0
        self.dmd.dmd_time['dt'] = 1
        self.dmd.dmd_time['tend'] = test_data.shape[1]-1
            
        assert test_data.shape[0] == self.train_data.shape[0], "Test data must have the same number of features as training data"
        assert test_data.shape[1] == self.train_data.shape[1], "Test data must have the same number of time steps as training data"

        pred_data = self.dmd.reconstructed_data[:, :-self.delay+1]

        return pred_data

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.dmd.dmd_time['t0'] = 0
        self.dmd.dmd_time['dt'] = 1
        self.dmd.dmd_time['tend'] = test_data.shape[1] + self.dmd.original_time['tend']
            
        pred_data = self.dmd.reconstructed_data[:, :-self.delay+1]

        return pred_data[:, self.train_data.shape[1]:]
    
class HighOrderDMD:
    """
    
    """
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None):
        """
        
        """
        self.rank = config['model']['rank']
        self.delay = config['model']['delay']
        self.train_data = train_data


        # Train the DMD model
        # print(f"Training DMD model with rank {self.rank}")
        warnings.filterwarnings("ignore")
        self.dmd = HODMD(svd_rank=self.rank, d=self.delay)
        self.dmd.fit(train_data)

        self.dmd.original_time['t0'] = 0
        self.dmd.original_time['dt'] = 1
        self.dmd.original_time['tend'] = train_data.shape[1]-1

    def reconstruct(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.dmd.dmd_time['t0'] = 0
        self.dmd.dmd_time['dt'] = 1
        self.dmd.dmd_time['tend'] = test_data.shape[1]-1
            
        assert test_data.shape[0] == self.train_data.shape[0], "Test data must have the same number of features as training data"
        assert test_data.shape[1] == self.train_data.shape[1], "Test data must have the same number of time steps as training data"

        pred_data = self.dmd.reconstructed_data

        return pred_data

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.dmd.dmd_time['t0'] = 0
        self.dmd.dmd_time['dt'] = 1
        self.dmd.dmd_time['tend'] = test_data.shape[1] + self.dmd.original_time['tend']
            
        pred_data = self.dmd.reconstructed_data

        return pred_data[:, self.train_data.shape[1]:]
    
class BaggingOptimisedDMD:
    """
    
    """
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None):
        """
        
        """
        self.rank = config['model']['rank']
        self.num_trials = config['model']['num_trials']
        self.train_data = train_data
        self.train_time = np.arange(train_data.shape[1])
        self.eig_constraints = {*config['model']['eig_constraints']} if 'eig_constraints' in config['model'] else None

        # Train the DMD model
        # print(f"Training DMD model with rank {self.rank}")
        warnings.filterwarnings("ignore")
        self.dmd = BOPDMD(svd_rank=self.rank, 
                          num_trials=self.num_trials,
                          eig_constraints=self.eig_constraints)
        
        self.dmd.fit(train_data, self.train_time)

    def reconstruct(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.test_time = np.arange(test_data.shape[1])
            
        assert test_data.shape[0] == self.train_data.shape[0], "Test data must have the same number of features as training data"
        assert test_data.shape[1] == self.train_data.shape[1], "Test data must have the same number of time steps as training data"

        pred_data = self.dmd.forecast(self.test_time)

        if self.num_trials > 0:
            pred_data = pred_data[0]

        return pred_data

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.train_data is None:
            raise ValueError("Training data is required for DMD")
        
        self.test_time = np.arange(self.train_data.shape[1] + test_data.shape[1])

        pred_data = self.dmd.forecast(self.test_time)

        if self.num_trials > 0:
            pred_data = pred_data[0]

        return pred_data[:, self.train_data.shape[1]:]