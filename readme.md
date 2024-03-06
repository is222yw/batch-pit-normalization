# Batch Probability Integral Transform Normalization

Or, _"BatchPitNormalization"_, is a layer for neural networks that, similar to ordinary Batch Normalization, will correct covariate shift.
Beyond that, it will modify the distribution of the data flowing through to be, e.g., uniform or normal (or to be of almost any other family).

This layer does not require the data to be normalized in any way before (actually, doing so would be wasteful, but not hurtful).
BatchPitNormalization estimates a Gaussian kernel density per each feature based on the observed data for each feature.
Then it uses the CDF of the density to transform each feature such that it has a uniform distribution.
This may then be further transformed into another distribution (we have built-in support for normal and plan others, too).
