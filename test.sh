python test.py \
   --test_directory ../data/test_data \
   --trained_model_directory ../data/trained_models/08_veryLargeTrain_randomWindow \
   --chromosome_number 1 \
   --region_start $1 \
   --region_end $2 \
   --number_windows $3 \
   --max_y $4 \
   --normalized_depths_only
