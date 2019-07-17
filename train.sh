#!/usr/bin/env bash

# set -e
# set -uo pipefail
# set -x
# export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

trained_model_directory=$1 
filter=$2 
resampling_target_file_name=$3

source ask.sh

if [ -d "$trained_model_directory" ]; then
	if ! ask "Do you want to overwrite $trained_model_directory?"; then
		echo 'Exiting ...'
		exit 1
	else
		echo 'Overwriting ...'
	fi
else
	mkdir $trained_model_directory
fi

function train {
   	nohup $1 nice -19 python train.py \
	   	--trained_model_directory $trained_model_directory \
		--depth_file_name 100.multicov.bin \
	   	--chromosome_number 22 \
	   	--start 20000000 \
	   	--end 50000000 \
	   	--batch_size 256 \
	   	--learning_rate 0.0005 \
	   	--window_half_width 500 \
	   	--fold_reduction_of_sample_size 200 \
		--filter $filter \
		--resampling_target_file_name $resampling_target_file_name \
		< /dev/null > $trained_model_directory/train.out 2> $trained_model_directory/train.err & 
}

# cluster computing
if (command -v caffeinate) > /dev/null; then
	echo 'running command with caffeinate!' 
	train "caffeinate -i"
else 
	echo 'running command without caffeinate!'
	train "" 
fi

