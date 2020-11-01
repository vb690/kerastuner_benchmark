# Benchmarking Kerastuner Algorithms
Small project testing the functionalities of [Keras Tuner](https://keras-team.github.io/keras-tuner/) and benchmarking the three available optimization algorithms.

## Problem

Given a learning algorithm and a problem to solve (in the simplest case these are classification or regression tasks) we know that exists a set of hyperparamenters which allow the algorithm to converge to the optimal solution both in terms of loss reduction and generalizability to previously unseen data.

Given the magnitude of all the possible combinations of hyperparameters and their associated values (i.e. the size of the hyperparameters space) the process of finding the best set is often left to optimization algorithms which aims to find promising candidates in an efficient manner. 

Therefore, understanding which algorithm is able to achieve the best performance in the shortest ammount of time (or employing the least ammount of computational resources) becomes of pivotal impartance.

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

**Vanilla MNIST**
<p align="center">   
  <img width="500" height="250" src="https://github.com/vb690/kerastuner_benchmark/blob/master/data/figures/mn_v.png">
</p>  
  
**Back MNIST**
<p align="center">   
  <img width="395" height="100" src="https://github.com/vb690/kerastuner_benchmark/blob/master/data/figures/mn_b.png">
</p>  
  
**Rotated MNIST**
<p align="center">   
  <img width="395" height="100" src="https://github.com/vb690/kerastuner_benchmark/blob/master/data/figures/mn_r.png">
</p>   
  
**RotBack MNIST**
<p align="center">   
  <img width="395" height="100" src="https://github.com/vb690/kerastuner_benchmark/blob/master/data/figures/mn_rb.png">
</p> 

## Methodology

## Results 

### Visual Comparison  
  
<p align="center">   
  <img width="900" height="500" src="https://github.com/vb690/kerastuner_benchmark/blob/master/results/figures/tuners_perfromance.png">
</p> 
  
### Bayesian Generalized Mixed Model - Varying Intercept 
  
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Metric</th>
    <th class="tg-7btt" colspan="4">Random Search</th>
    <th class="tg-7btt" colspan="4">Gaussian Process</th>
    <th class="tg-7btt" colspan="4">HyperBand</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-fymr">Mean</td>
    <td class="tg-fymr">Std</td>
    <td class="tg-fymr">3% hdi</td>
    <td class="tg-fymr">97% hdi</td>
    <td class="tg-fymr">Mean</td>
    <td class="tg-fymr">Std</td>
    <td class="tg-fymr">3% hdi</td>
    <td class="tg-fymr">97% hdi</td>
    <td class="tg-fymr">Mean</td>
    <td class="tg-fymr">Std</td>
    <td class="tg-fymr">3% hdi</td>
    <td class="tg-fymr">97% hdi</td>
  </tr>
  <tr>
    <td class="tg-7btt">Accuracy</td>
    <td class="tg-c3ow"><span style="color:#FE0000">0.150</span></td>
    <td class="tg-c3ow">0.069</td>
    <td class="tg-c3ow">0.021</td>
    <td class="tg-c3ow">0.283</td>
    <td class="tg-c3ow">0.147</td>
    <td class="tg-c3ow">0.069</td>
    <td class="tg-c3ow">0.016</td>
    <td class="tg-c3ow">0.280</td>
    <td class="tg-c3ow">0.149</td>
    <td class="tg-c3ow">0.069</td>
    <td class="tg-c3ow">0.020</td>
    <td class="tg-c3ow">0.283</td>
  </tr>
  <tr>
    <td class="tg-7btt">F1 Score</td>
    <td class="tg-0pky">0.145</td>
    <td class="tg-0pky">0.069</td>
    <td class="tg-0pky">0.012</td>
    <td class="tg-0pky">0.272</td>
    <td class="tg-0pky">0.139</td>
    <td class="tg-0pky">0.069</td>
    <td class="tg-0pky">0.002</td>
    <td class="tg-0pky">0.263</td>
    <td class="tg-0pky">0.143</td>
    <td class="tg-0pky">0.069</td>
    <td class="tg-0pky">0.011</td>
    <td class="tg-0pky">0.271</td>
  </tr>
  <tr>
    <td class="tg-7btt">Precision</td>
    <td class="tg-0pky">0.151</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">0.016</td>
    <td class="tg-0pky">0.277</td>
    <td class="tg-0pky">0.144</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">0.012</td>
    <td class="tg-0pky">0.273</td>
    <td class="tg-0pky">0.152</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">0.016</td>
    <td class="tg-0pky">0.277</td>
  </tr>
  <tr>
    <td class="tg-7btt">Recall</td>
    <td class="tg-0pky">147</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">0.016</td>
    <td class="tg-0pky">0.281</td>
    <td class="tg-0pky">143</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">0.012</td>
    <td class="tg-0pky">0.278</td>
    <td class="tg-0pky">145</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">0.012</td>
    <td class="tg-0pky">0.277</td>
  </tr>
</tbody>
</table>

## Installation

1. Download your local version of the repository
2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
3. Open the Anaconda Powershell Prompt in the repository directory:
```sh
# create anaconda environment
conda create -n tuner_bench_env tensorflow-gpu

# activate the environment
conda activate tuner_bench_env
```

4. At this point install all the requirements with:

```sh
# install the requirements
conda install -c conda-forge --file requirements.txt
```
