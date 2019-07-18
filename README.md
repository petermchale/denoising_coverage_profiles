# Bias in sequencing the genome  

Structural variants--large genetic mutations--are often identified 
in the human genome using algorithms that search for hills and 
valleys in read depth of coverage--the number of sequenced 
"reads" that align to each part of the human genome. 

But certain "motif" sequences in 
the genome are known to be associated with 
changes in coverage, even in the absence of a structural variant. 

Such systematic biases in depth of coverage need to be corrected 
before those data are passed to a structural variant caller 
to avoid false calls.  

# Learning and correcting the bias 

Convolutional Neural Networks have recently been used to 
classify genomic sequences. I illustrate the approach 
[here](http://nbviewer.jupyter.org/github/petermchale/denoising_coverage_profiles/blob/master/discovering_DNA_motifs_using_convnets_classification.ipynb)
using toy sequence data. 

We adapted this idea and built a Convolutional Neural Network 
that models the read depth associated with a given sequence 
as a mixture of Poisson distributions.
 
When this model is trained on sequences containing 
an AT-dinucleotide repeat and random sequences, 
it [corrects 
depth of coverage](https://colab.research.google.com/drive/1jIM5OOurbUeP_an0qiQwAsInYIHZClxf)
in sequences harboring the AT-dinucleotide repeat.    


# Next steps 

What is needed now is a training set enriched for 
ALL motifs in the genome that affect coverage. 
With that in hand, the model could be trained to correct 
all systematic biases present in the genome.  

Possible ways to obtain such a training set include: 

* pooling depths across multiple samples, thereby increasing the 
signal-to-noise ratio, and then selecting examples with, e.g., 
lower than expected read depth
* using the HOMER and MEME-CHIP toolsets 
to find motifs that appear in the training set 
more often than in an equal-sized set of random DNA sequences, 
and then retaining only training examples that 
contain one or more of these motifs.


