# Data description

## Sources

The sources are saved in `sources.pkl`. These are of size 346x346 and constituted of:
- a synchrotron source,
- a thermal source,
- two iron emission line sources; they are called Gaussian emission line sources in the rest of the project.

## Spectra

Three family of spectra are provided in `spectra_sync.npz`, `spectra_thermal.npz` and `spectra_gauss.npz`. They are divided as follows:
- two or three anchor points,
- approximately 60% in a training set,
- approximately 20% in a validation set,
- approximately 20% in a test set.

## IAE models

IAE models for the three types of spectra are provided in `model_sync.pkl`, `model_thermal.pkl` and `model_gauss.pkl`.
They were generated with the same anchor points, training sets and validation sets as above.
These files store:
 - the weights of the multi-layer forward neural networks constituting the IAE,
 - various parameters of the IAE (e.g. activation functions, regularization hyperparamaters).
 - the anchor points.
