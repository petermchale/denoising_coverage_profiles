python test.py \
   	--test_directory $1 \
	--depth_file_name ../data/depths/1.multicov.int32.bin \
   	--trained_model_directory ../data/trained_models/09_100samples_final/09_100samples_humpedDistribution \
   	--chromosome_number 22 \
   	--content_start $2 \
   	--content_end $3 \
	--filter filter1 \
	--number_test_examples 1000 \
	--padding 0.5
