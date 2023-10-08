# Step Counting with Attention-based LSTM

Implementation of [Step Counting with Attention-based LSTM](https://arxiv.org/abs/2211.13114)

## Requirments
* torch
* numpy
* sklearn
* tqdm

Instructions for Writing the Code:

Ensure you have the file `WeAllWalk.pt` in your current directory, containing acceleration data from sighted walkers extracted from the [WeAllWalk dataset](https://dl.acm.org/doi/10.1145/3161711), comprising three components: acceleration signals, lengths of the signals, and number of steps in the signals.

Run the Python script named `main.py`. Within this script:
   a. `WeAllWalk.pt` is loaded
   b. The functions available in `utils.py` are used to pre-process the loaded data.
   c. The model in the class provided in `models.py` is trained and evaluated using five-fold cross-validation.
   d. The evaluation results are outputted by implementing the evaluation metrics from `metrics.py`.
