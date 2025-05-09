import numpy as np
from typing import List, Optional, Dict
from pydmd import DMD, HODMD, BOPDMD, ParametricDMD
from ezyrb import POD, RBF
from pydmd.preprocessing import hankel_preprocessing
import warnings
from ctf4science.data_module import get_config

class DMD4CTF:
    """
    A wrapper class for performing various types of Dynamic Mode Decomposition (DMD) on spatiotemporal data for the CTF dataset.

    This class initializes and manages DMD variants (classic, Hankel, high-order, bagging-optimized), handling preprocessing, parameter setup, and model training/prediction.

    Parameters:
    -----------
    config : Dict
        Configuration dictionary containing model and dataset parameters. Must include:
        - model.method: One of ['classic', 'hankel', 'highorder', 'bagopt']
        - model.rank: SVD rank for dimensionality reduction
        - model.delay (optional): Delay parameter for time-embedded methods
        - model.num_trials (optional): Number of trials for bagging
        - model.eig_constraints (optional): Constraints on eigenvalues
        - model.parametric (optional): Parametric mode ('monolithic' or other)
    pair_id : int
        Identifier for the specific data pair used to define train/test splits.
    train_data : List[np.ndarray]
        List of training snapshots. Each element is a 2D array with shape 
        (spatial_features, timesteps).
    train_time : np.ndarray
        1D array of time points corresponding to the training data.
    check_svd : bool, optional
        Whether to check the SVD residual energy to assess the rank adequacy (default: True).

    Attributes:
    -----------
    dmd : object
        The instantiated DMD model (ClassicDMD, HankelDMD, etc.), set after `initialize()`.
    train_data : np.ndarray
        The stacked training data array of shape (num_samples, spatial_features, timesteps).
    train_time : np.ndarray
        Time vector used for training.
    dt : float
        Time step computed from `train_time`.
    spatial_dimension : int
        Number of spatial features expected in the dataset.
    pair_id : int
        Dataset pair identifier used to select metadata and parametric splits.
    matrix_start_index : int
        Index indicating the start of the test matrix within the data file.
    test_shape : tuple
        Shape of the test data matrix used for reshaping predictions.
    rank : int
        Truncation rank for SVD.
    parametric : dict or None
        Dictionary specifying parametric settings based on `pair_id`, or None if not applicable.
    method : str
        DMD method specified in the configuration.

    Methods:
    --------
    initialize():
        Initialize the DMD model instance according to the specified method.

    train():
        Fit the DMD model to the training data.

    predict(prediction_timesteps: np.ndarray) -> np.ndarray:
        Predict future states using the trained model over the provided time vector.
    
    _check_svd_residual_energy(rank: int):
        Internal method to compute and warn about the energy retained by the chosen SVD rank.


    References
    ----------
    - P. J. Schmid (2010). "Dynamic Mode Decomposition of numerical and experimental data." *Journal of Fluid Mechanics*, 656, 5. Cambridge University Press. DOI: https://doi.org/10.1017/S0022112010001217
    - J. Nathan Kutz, Steven L. Brunton, Bingni W. Brunton, and Joshua L. Proctor (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM - Society for Industrial and Applied Mathematics. DOI: https://doi.org/10.1137/1.9781611974508
    - S. M. Ichinaga, F. Andreuzzi, N. Demo, M. Tezzele, K. Lapo, G. Rozza, S. L. Brunton, and J. N. Kutz (2024). "PyDMD: A Python package for robust dynamic mode decomposition." Preprint available at: https://arxiv.org/abs/2402.07463
    
    """

    def __init__(self, config: Dict, pair_id: int, train_data: List[np.ndarray], train_time: np.ndarray, check_svd: Optional[bool] = True):
        """
        Initialize the DMD4CTF class with configuration, data, and metadata.

        Parameters
        ----------
        config : Dict
            Dictionary containing the model and dataset configuration.
        pair_id : int
            Identifier used to select the dataset pair and associated metadata.
        train_data : List[np.ndarray]
            List of training data arrays, each of shape (spatial_features, timesteps).
        train_time : np.ndarray
            Array of time points corresponding to the training data.
        check_svd : bool, optional
            If True, computes the SVD residual energy for the chosen rank to validate adequacy (default: True).
        """
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
        Compute the residual energy of the truncated SVD to evaluate the chosen rank.

        Parameters
        ----------
        rank : int
            Rank used to truncate the SVD for dimensionality reduction.
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
        Initialize the appropriate DMD model based on the method specified in the configuration.

        Supported methods:
        - 'classic': Classic DMD
        - 'hankel': Hankel DMD (time-delayed embedding)
        - 'highorder': High-Order DMD
        - 'bagopt': Bagging-Optimized DMD

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
        Train the initialized DMD model using the provided training data and time vector.
        """
        self.dmd.train(self.train_data, self.train_time)
        
    def predict(self, prediction_timesteps: np.ndarray) -> np.ndarray:
        """
        Use the trained DMD model to predict future states at the specified time steps.

        Parameters
        ----------
        prediction_timesteps : np.ndarray
            1D array of time points at which the DMD model should generate predictions.

        Returns
        -------
        np.ndarray
            Predicted state matrix of shape (spatial_features, num_timesteps).
        
        """
        return self.dmd.predict(prediction_timesteps)

class ClassicDMD():
    """
    Implementation of the classic (and optionally parametric) Dynamic Mode Decomposition (DMD).

    This class supports both standard and parametric DMD using a POD projection and RBF interpolation 
    for parameter space generalization. It can operate in 'monolithic' or 'partitioned' mode 
    depending on the configuration.

    Parameters
    ----------
    svd_rank : int
        Truncation rank used for SVD in DMD (or POD if parametric).
    parametric : dict, optional
        Dictionary defining parametric behavior. Expected keys:
        - 'mode': either 'monolithic' or 'partitioned'
        - 'train_params': list or array of training parameter values
        - 'test_params': list or array of test parameter values

    Attributes
    ----------
    dmd : object
        Instance of the core DMD or ParametricDMD model.
    pod : POD, optional
        POD object used for dimensionality reduction in the parametric case.
    interpolator : RBF, optional
        Interpolator used in parameter space (only for parametric DMD).
    spatial_dimension : int
        Number of spatial features in the training data.
    dt : float
        Time step inferred from the training time vector.
    """
    def __init__(self, svd_rank: int, parametric: Optional[dict] = None):
        """
        Initialize the DMD model with or without parametric capability.

        Parameters
        ----------
        svd_rank : int
            Rank for SVD truncation.
        parametric : dict, optional
            Parametric configuration dictionary. If None, standard DMD is used.
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
        Fit the DMD model to the provided training data and time vector.

        Parameters
        ----------
        train_data : np.ndarray
            Training data of shape (num_samples, spatial_features, num_timesteps).
            For parametric DMD, num_samples should match the number of training parameters.
        train_time : np.ndarray
            1D array representing the time points of the training data.

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
        Predict future states using the trained DMD model.

        Parameters
        ----------
        prediction_timesteps : np.ndarray
            1D array of time points over which to generate predictions.

        Returns
        -------
        np.ndarray
            Reconstructed data matrix of shape (spatial_features, len(prediction_timesteps)).

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
    Implementation of Dynamic Mode Decomposition (DMD) with Hankel embedding for delay-coordinate preprocessing.

    Supports both standard and parametric modes. In the parametric setting, a POD projection and RBF interpolation 
    are used for dimensionality reduction and parameter generalization, respectively.

    Parameters
    ----------
    svd_rank : int
        Truncation rank used for SVD in DMD or POD.
    delay : int
        Number of delays used to construct the Hankel matrix.
    parametric : dict, optional
        Dictionary defining parametric behavior. Expected keys:
        - 'mode': either 'monolithic' or 'partitioned'
        - 'train_params': list or array of training parameter values
        - 'test_params': list or array of test parameter values

    Attributes
    ----------
    dmd : object
        Core Hankel-embedded DMD model (standard or parametric).
    pod : POD, optional
        POD instance used in parametric setting for dimensionality reduction.
    interpolator : RBF, optional
        Interpolator used in parameter space.
    spatial_dimension : int
        Number of spatial features in the training data.
    dt : float
        Time step inferred from the training time vector.
    delay : int
        Delay used in Hankel embedding.
    """
    def __init__(self, svd_rank: int, delay: int, parametric: Optional[dict] = None):
        """
        Initialize the Hankel-embedded DMD model.

        Parameters
        ----------
        svd_rank : int
            Rank for SVD truncation.
        delay : int
            Number of time delays for the Hankel embedding.
        parametric : dict, optional
            Dictionary specifying parametric DMD configuration. If None, standard Hankel-DMD is used.
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
        Fit the Hankel-DMD model to the training data and corresponding time vector.

        Parameters
        ----------
        train_data : np.ndarray
            Training data of shape (num_samples, spatial_features, num_timesteps).
            For parametric DMD, num_samples should match the number of training parameters.
        train_time : np.ndarray
            1D array of time points corresponding to the training data.


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
        Predict future states using the trained Hankel-DMD model.

        Parameters
        ----------
        prediction_timesteps : np.ndarray
            1D array of time points over which to generate predictions.

        Returns
        -------
        np.ndarray
            Reconstructed data matrix of shape (spatial_features, len(prediction_timesteps)).

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
    Implementation of High-Order Dynamic Mode Decomposition (HODMD) with optional parametric support.

    This class supports both standard HODMD and a parametric variant with POD-based reduction and 
    RBF interpolation across parameters.

    Parameters
    ----------
    svd_rank : int
        Truncation rank for the SVD step in HODMD or POD.
    delay : int
        Number of delays used for high-order embedding.
    parametric : dict, optional
        Dictionary defining parametric behavior. Expected keys:
        - 'mode': either 'monolithic' or 'partitioned'
        - 'train_params': array of parameter values for training data
        - 'test_params': array of parameter values for prediction

    Attributes
    ----------
    dmd : object
        The underlying HODMD model (or ParametricDMD wrapper).
    pod : POD, optional
        POD instance used in parametric setting for dimensionality reduction.
    interpolator : RBF, optional
        Interpolator in the parameter space.
    spatial_dimension : int
        Number of spatial features in the input data.
    dt : float
        Time step computed from the training time vector.
    """
    def __init__(self, svd_rank: int, delay: int, parametric: Optional[dict] = None):
        """
        Initialize the HODMD model with optional parametric support.

        Parameters
        ----------
        svd_rank : int
            Truncation rank for the SVD step.
        delay : int
            Number of delays for high-order embedding.
        parametric : dict, optional
            Parametric model configuration. If None, standard HODMD is used.
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
        Fit the HODMD model to the training data.

        Parameters
        ----------
        train_data : np.ndarray
            Array of shape (num_samples, spatial_features, num_timesteps).
            In parametric mode, num_samples should match the number of training parameters.
        train_time : np.ndarray
            1D array of time points corresponding to the training data.

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
        Predict future states using the trained HODMD model.

        Parameters
        ----------
        prediction_timesteps : np.ndarray
            1D array of time points over which predictions should be made.

        Returns
        -------
        np.ndarray
            Reconstructed data array of shape (spatial_features, len(prediction_timesteps)).

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
    Bagging Optimised DMD (BOP-DMD) implementation with optional delay embedding and parametric extension.

    Supports both non-parametric and parametric modes using POD for dimensionality reduction and
    RBF interpolation in parameter space. Includes optional ensemble-based optimization with multiple trials.

    Parameters
    ----------
    svd_rank : int
        Truncation rank for the SVD step or POD reduction.
    delay : int, optional
        Number of delays for Hankel preprocessing. If None, no delay embedding is applied.
    num_trials : int, optional
        Number of trials used for bagging optimization. Default is 0 (no bagging).
    eig_constraints : set, optional
        Set of constraints on the eigenvalues (e.g., 'stable') passed to the optimizer.
    parametric : dict, optional
        Dictionary defining the parametric configuration. Expected keys:
        - 'mode': 'monolithic' or 'partitioned'
        - 'train_params': array of training parameter values
        - 'test_params': array of parameter values for prediction
    tol : float, optional
        Tolerance for the variable projection optimization. Default is 0.475.

    Attributes
    ----------
    dmd : object
        The internal DMD model (standard or wrapped with ParametricDMD).
    delay : int or None
        Delay used for Hankel embedding (if any).
    num_trials : int
        Number of optimization trials.
    eig_constraints : set
        Constraints on eigenvalues.
    spatial_dimension : int
        Number of spatial features in the training data.
    """
    def __init__(self, svd_rank: int,
                 delay: Optional[int] = None, num_trials: Optional[int] = 0, eig_constraints: Optional[set] = None,
                 parametric: Optional[dict] = None, tol=0.475):
        """
        Initialize the BaggingOptimisedDMD model with optional delay embedding and parametric support.

        Parameters
        ----------
        svd_rank : int
            SVD rank for truncation or POD reduction.
        delay : int, optional
            Number of delays for Hankel embedding.
        num_trials : int, optional
            Number of bagging trials for the optimizer.
        eig_constraints : set, optional
            Eigenvalue constraints for optimization (e.g., {'stable'}).
        parametric : dict, optional
            Dictionary configuring the parametric setup.
        tol : float, optional
            Optimization tolerance for varpro (default 0.475).
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

            base_dmd = lambda: BOPDMD(
                svd_rank=-1,
                num_trials=num_trials,
                eig_constraints=eig_constraints,
                varpro_opts_dict={'verbose': False, 'tol': tol}
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
        Train the BaggingOptimisedDMD model on the provided data and time vector.

        Parameters
        ----------
        train_data : np.ndarray
            Training data of shape (num_samples, spatial_features, time_steps).
            For parametric mode, num_samples should match the number of training parameters.
        train_time : np.ndarray
            1D array of time points corresponding to the training snapshots.

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
        Generate predictions for the given time steps using the trained model.

        Parameters
        ----------
        prediction_timesteps : np.ndarray
            1D array of future time points for which to forecast the system state.

        Returns
        -------
        np.ndarray
            Predicted data array of shape (spatial_features, len(prediction_timesteps)).

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
        Internal method to forecast modal coefficients in parametric settings (this extends the code of pydmd to bopdmd).

        Parameters
        ----------
        prediction_timesteps : np.ndarray
            1D array of time points for which to forecast the system state.

        Returns
        -------
        np.ndarray
            Interpolated predictions in the original spatial domain, reconstructed from modal coefficients.

        Notes
        -----
        - Handles both monolithic and partitioned parametric models.
        - Applies interpolation over the forecasted modal coefficients.
        - Extracts appropriate coefficients in case of delay embedding.
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