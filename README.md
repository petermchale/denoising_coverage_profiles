# Motivation

Clinical labs are moving towards WGS. 

CNVs can be better discovered using depth of coverage profiles from WGS than:  
* exome sequencing (where coverage is biased systematically by hybridization and amplification) 
* array hybridization (where probe density is sparse (1/10000bp)). 

But WGS coverage profiles are subject to systematic biases, confounding the discovery of CNVs. 

Here, we explore a deep-learning approach to characterize these biases. 
Deep learning has proven successful at mapping sequence information onto 
epigenetic information. We tackle the potentially more challenging 
task of deep-learning a mapping from sequence to read depth, 
with the goal of "denoising" coverage profiles.  

Any methods developed ought to be highly generalizable to other seq protocols, e.g., RNA-seq. 

# Summary of work done so far (this project is very much a work in progress)

[Basic Poisson model](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/poisson_model_of_read_depths.ipynb)

[Signal-to-noise analysis](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/signal_to_noise.ipynb)

[Rejection sampling](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/rejection_sampling.ipynb)

## 1D convolution models in Keras

[How conv1d works in Keras](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/conv1d_basic.ipynb)

[Genomics classification task](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/discovering_DNA_motifs_using_convnets_classification.ipynb)

[Genomics regression task](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/discovering_DNA_motifs_using_convnets_regression.ipynb)

## Mixture models

[Mixture of Gaussians](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/gaussian_mixture_distribution_basic.ipynb)

[Mixture of Poissons: toy model](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/poisson_mixture_distribution_basic.ipynb)


