#!/bin/bash
now=$(date +'%Y-%m-%d %R:%S')
echo "Launching a new job with the following configurations."
echo "Job Type: $1"
echo "Epochs: $2"
echo "Dropout: $3"
echo "Hidden: $4"
echo "Embed: $5"
echo "Batch: $6"
echo "GPU: $7"
echo
echo "Starting job."
echo
cd ~/NewProj3
python $1.py -epochs $2 -drop $3 -hidden_size $4 -embedding_dim $5 -batch_size $6 -saved_model models/"$1"_model_"$2"_"$3"_"$4"_"$5"_"$6".h5 -student_id "$1"_"$2"_"$3"_"$4"_"$5"_"$6"_"$now" -gpu $7
python $1.py -mode test -saved_model models/"$1"_model_"$2"_"$3"_"$4"_"$5"_"$6".h5 -student_id "$1"_"$2"_"$3"_"$4"_"$5"_"$6"_"$now" -score -gpu $7
python $1.py -mode test -saved_model models/"$1"_model_"$2"_"$3"_"$4"_"$5"_"$6".h5 -student_id 20423612 -input test -score -gpu $7