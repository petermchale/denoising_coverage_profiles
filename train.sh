#!/usr/bin/env bash

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
fi

nohup caffeinate -i nice -19 python train.py \
	--train_data $train_data \
	--trained_model $trained_model \
	--region_start $region_start \
	--region_end $region_end \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	< /dev/null > train.out 2> train.err &

