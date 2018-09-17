#!/usr/bin/env bash

# set -e
# set -uo pipefail
# set -x
# export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

train_data='../data/train_data'
trained_model=$1 
region_start=0
region_end=$2
batch_size=256
learning_rate=$3

source ask.sh

if [ -d "$trained_model" ]; then
	if ! ask "Do you want to overwrite $trained_model?"; then
		echo 'Exiting ...'
		exit 1
	else
		echo 'Overwriting ...'
	fi
else
	mkdir $trained_model
fi

function train {
   	nohup $1 nice -19 python train.py \
	   	--train_data $train_data \
	   	--trained_model $trained_model \
	   	--region_start $region_start \
	   	--region_end $region_end \
	   	--batch_size $batch_size \
	   	--learning_rate $learning_rate \
	   	< /dev/null > $trained_model/train.out 2> $trained_model/train.err & 
}

# cluster computing
if (command -v caffeinate) > /dev/null; then
	echo 'running command with caffeinate!' 
	train "caffeinate -i"
else 
	echo 'running command without caffeinate!'
	train "" 
fi

