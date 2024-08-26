# Tunable Privacy Risk Evaluation of Generative Adversarial Networks

This is an example, on ChestMNIST dataset, of the attack presented in https://doi.org/10.3233/shti240634


## Requirements

- python
- numpy
- pandas
- matplotlib
- sklearn

## Repository structure

- `signal_values`: contains the discriminator outputs of a GAN trained on ChestMNIST dataset, on the training dataset `train_signals`, on the population dataset `population_signals`, and on the test dataset `test_signals`.
- `evaluator.py`: contains helper functions for the privacy risk evaluation
- `example.ipynb`: contains an example of the evaluation using the example signals on the ChestMNIST dataset.