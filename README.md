# Motivation

Clinical labs are moving towards WGS. 

CNVs can be better discovered using depth coverage profiles from WGS than:  
* exome sequencing (where coverage is biased systematically by hybridization and amplification) 
* array hybridization (where probe density is sparse (1/10000bp)). 

But WGS coverage profiles are noisy, confounding the discovery of CNVs. 

The standard approach to characterize the noise is to do control experiments 
where the signal is purposefully left out.  But this costs money and time. 

Here we use deep learning techniques to learn the noise characteristics. 

Any methods developed will be highly generalizable to other seq protocols, e.g., RNA-seq. 



