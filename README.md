# DMD Variants for Koopman Operator Learning

This directory ...

## Methods

- **Classic DMD**
- **Hankel Preprocessing**
- **High Order DMD**
- **Bagging Optimised DMD**

## Usage (check?)

Run a baseline using the `run.py` script followed by the path to a configuration file. For example:

```bash
python run.py config/config_average.yaml
```

This executes the average baseline on the specified dataset and pair ID, saving predictions and evaluation results in `results/<dataset_name>/CTF_NaiveBaselines/<run_name>/`.

## Configuration Files

Configuration files are located in the `config/` directory:

- `<conf_name>.yaml`:
-
Each file specifies the dataset, pair ID, and baseline method, plus method-specific parameters.

### Examples

- **Hankel Preprocessing**:
  ```bash
  python run.py config/config_average.yaml????
  ```

## Requirements

The baselines rely on packages already in the main `requirements.txt`:
- pyyaml
- pydmd
- ezyrb

No additional dependencies are required (see `requirements.txt`).

## Notes
