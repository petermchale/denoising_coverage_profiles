# Motivation

Clinical labs are moving towards WGS. 

CNVs can be better discovered using depth coverage profiles from WGS than:  
* exome sequencing (where coverage is biased systematically by hybridization and amplification) 
* array hybridization (where probe density is sparse (1/10000bp)). 

But WGS coverage profiles are noisy, confounding the discovery of CNVs. 

The standard approach to characterize the noise is to do control experiments 
where the signal is purposefully left out.  But this costs money and time. 

Here, we explore a deep-learning approach to characterize the noise. 
Deep learning has proven successful at mapping sequence information onto 
epigenetic information. We tackle the potentially more challenging 
task of deep-learning a mapping from sequence to read depth, 
with the goal of "denoising" coverage profiles.  

Any methods developed ought to be highly generalizable to other seq protocols, e.g., RNA-seq. 

# Summary of work done so far 

[Basic theory](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/theory.ipynb)

[Signal to noise analysis](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/signal_to_noise.ipynb)

[Rejection sampling](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/rejection_sampling.ipynb)

## 1D convolution toy models in Keras

[basic](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/conv1d_basic.ipynb)


