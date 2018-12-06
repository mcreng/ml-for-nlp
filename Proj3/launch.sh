#!/bin/bash
echo "Launching a new job with the following configurations."
echo "Job Type: $1"
echo "Epochs: $2"
echo "Dropout: $3"
echo "Hidden: $4"
echo "Embed: $5"
echo
echo "Starting job."
echo
cd ~/NewProj3
python $1.py -epochs $2 -drop $3 -hidden_size $4 -embedding_dim $5 -saved_model models/"$1"_model_"$2"_"$3"_"$4"_"$5".h5 -student_id "$1_$2_$3_$4_$5"
python $1.py -mode test -saved_model models/"$1"_model_"$2"_"$3"_"$4"_"$5".h5 -student_id "$1_$2_$3_$4_$5"
python scorer.py -submission "$1_$2_$3_$4_$5"_valid_result.csv