model=$1
tensorboard --logdir=train:../data/trained_models/$model/tensorboard/train,dev:../data/trained_models/$model/tensorboard/dev
