# Benchmarking Kerastuner Algorithms
Small project testing the functionalities of [Keras Tuner](https://keras-team.github.io/keras-tuner/) and benchmarking the three available optimization algorithms.

## Problem

Given a learning algorithm and a problem to solve (in the simplest case these are classification or regression tasks) we know that exists a set of hyperparamenters which allow the algorithm to converge to the optimal solution both in terms of loss reduction and generalizability to previously unseen data.

Given the magnitude of all the possible combinations of hyperparameters and their associated values (i.e. the size of the hyperparameters space) the process of finding the best set is often left to optimization algorithms which aims to find promising candidates in an efficient manner. 

Then, understanding which algorithm is able to achieve the best performance in the shortest ammount of time (or employing the least ammount of computational resources) becomes of pivotal impartance.

Here we aim to compare a set of [three optimization algorithms](https://keras-team.github.io/keras-tuner/documentation/tuners/) provided by the library Keras Tuner, namely:

* Random Search
* Bayesian Optimization using Gaussian Processes
* Hypberband

The three algorithms are tasked to find the best hyperparameters of a Multilayer Perceptron (MLP) which is used for classifiying digits from [4 variations](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits) of the MNIST dataset.

We will compare the three algorithms on 3 main metrics:

1. Score achieved by the MLP using the proposed configuration.
2. Time required to complete the optimization process.
3. Number of hyperparameters configurations explored.

## Data 

## Methodology

## Results 

<p align="center">   
  <img width="500" height="500" src="https://github.com/vb690/kerastuner_benchmark/blob/master/results/boxplot_results.png">
</p> 
