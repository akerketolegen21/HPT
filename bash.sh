#!/usr/bin/bash 

# echo "Start to process data"
# python convert_data.py \
#     --train datas/train_img_captions.json \
#     --val datas/validation_img_captions.json \
#     --output_dir data/meme_captions/ \
#     --dataset_name NGPT \

# echo "Start to train model"
# python train.py \
#     --data data/meme_captions/ \
#     --batch 16 \
#     --name hpt \

# echo "Start to test model"
# python test.py \
#     --name memes-subtask1-hpt \
#     --batch 16

echo "Make predictions"
python predict.py 
echo "Finished successfully"