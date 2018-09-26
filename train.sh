#!/usr/bin/env bash

# set -e
# set -uo pipefail
# set -x
# export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

trained_model_directory=$1 
batch_size=$2
learning_rate=$3
number_windows=$4
window_half_width=$5

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
	   	--train_dev_directory ../data/train_dev_data \
	   	--trained_model_directory $trained_model_directory \
	   	--bed_file_processor exactDepth_randomWindow \
	   	--bed_file_name facnn-example-1k.per-base.bed.gz \
	   	--chromosome_number 1 \
	   	--region_start 10000 \
	   	--region_end 149000000 \
	   	--batch_size $batch_size \
	   	--learning_rate $learning_rate \
	   	--number_windows $number_windows \
	   	--window_half_width $window_half_width
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

