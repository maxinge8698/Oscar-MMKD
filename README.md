# Project Preparation

```
git clone https://github.com/maxinge8698/Oscar-MMKD.git
cd Oscar-MMKD
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 067923d
touch __init__.py
cd ..
```

# Download

**Note**

It is recommended to download large files with [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) for faster speed. For convenience, the AzCopy executable tool has been downloaded and unzip to the root directory of the project in advance.

**Datasets**

Run command below to obtain the extracted image region features, object tags, and the original text annotations for each downstream tasks, which are provided by [Oscar](https://github.com/microsoft/Oscar).

```
wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1

# VQA
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/vqa.zip ./datasets/
unzip ./datasets/vqa.zip -d ./datasets/
# GQA
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/GQA.zip ./datasets/
unzip ./datasets/GQA.zip -d ./datasets/
# NLVR2
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/nlvr2.zip ./datasets/
unzip ./datasets/nlvr2.zip -d ./datasets/
# Image-Text Retrieval
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_ir.zip ./datasets/
unzip ./datasets/coco_ir.zip -d ./datasets/
```

**Pre-trained Models**

Run command below to obtain the pre-trained models, which are provided by [Oscar](https://github.com/microsoft/Oscar).

```
# OSCAR-base
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-vg-labels.zip ./pretrained_models
unzip ./pretrained_models/base-vg-labels.zip -d ./pretrained_models/
```


# Fine-tuning

Run command below to obtain the fine-tuned teacher for each task.

```
# VQA
python oscar/run_vqa.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 50 \
--data_label_type mask \
--data_dir datasets/vqa/2k \
--model_type bert \
--model_name_or_path pretrained_models/base-vg-labels/ep_107_1192087 \
--task_name vqa_text \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-5 \
--num_train_epochs 25 \
--output_dir model/vqa/teacher \
--label_file datasets/vqa/cache/trainval_ans2label.pkl \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--cls_hidden_scale 3

# GQA
python oscar/run_gqa.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 45 \
--train_data_type all \
--eval_data_type bal \
--test_data_type all \
--data_dir datasets/GQA/0.4true \
--model_type bert \
--model_name_or_path pretrained_models/base-vg-labels/ep_107_1192087 \
--task_name gqa \
--do_train \
--do_lower_case \
--max_seq_length 165 \
--per_gpu_train_batch_size 48 \
--per_gpu_eval_batch_size 48 \
--learning_rate 5e-5 \
--num_train_epochs 5 \
--output_dir model/gqa/teacher \
--label_file datasets/GQA/questions1.2/trainval_testdev_all_ans2label.pkl \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--cls_hidden_scale 3

# NLVR2
python oscar/run_nlvr.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 40 \
--eval_data_type all \
--test_data_type all \
--data_dir datasets/nlvr2/ft_corpus \
--model_type bert \
--model_name_or_path pretrained_models/base-vg-labels/ep_107_1192087 \
--task_name nlvr \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--per_gpu_train_batch_size 72 \
--per_gpu_eval_batch_size 72 \
--learning_rate 3e-5 \
--num_train_epochs 20 \
--output_dir model/nlvr2/teacher  \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--use_pair \
--use_label_seq

# Image-Text Retrieval
python oscar/run_retrieval.py \
--model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
--do_train \
--do_lower_case \
--evaluate_during_training \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--weight_decay 0.05 \
--save_steps 5000 \
--add_od_labels \
--od_label_type vg \
--max_seq_length 70 \
--output_dir model/coco_ir/teacher
```

# Distillation

Run command below to obtain distilled student for each task.

```
# VQA
python oscar/run_vqa_with_mmkd.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 50 \
--data_label_type mask \
--data_dir datasets/vqa/2k \
--model_type bert \
--task_name vqa_text \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--output_dir model/vqa/student \
--label_file datasets/vqa/cache/trainval_ans2label.pkl \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--cls_hidden_scale 3 \
--teacher_model model/vqa/teacher \
--student_model pretrained_models/base-vg-labels/ep_107_1192087 \
--alpha 0.5 \
--temperature 5.0 \
--num_hidden_layers 6 \
--teacher_learning_rate 5e-5 \
--student_learning_rate 5e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25

# GQA
x python oscar/run_gqa_with_mmkd.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 45 \
--train_data_type all \
--eval_data_type bal \
--test_data_type all \
--data_dir datasets/GQA/0.4true \
--model_type bert \
--task_name gqa \
--do_train \
--do_lower_case \
--max_seq_length 165 \
--per_gpu_train_batch_size 48 \
--per_gpu_eval_batch_size 48 \
--num_train_epochs 5 \
--output_dir model/gqa/student \
--label_file datasets/GQA/questions1.2/trainval_testdev_all_ans2label.pkl \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--cls_hidden_scale 3 \
--teacher_model model/gqa/teacher \
--student_model pretrained_models/base-vg-labels/ep_107_1192087 \
--alpha 0.5 \
--temperature 5.0 \
--num_hidden_layers 6 \
--teacher_learning_rate 5e-5 \
--student_learning_rate 5e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25

# NLVR2
python oscar/run_nlvr_with_mmkd.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 40 \
--eval_data_type all \
--test_data_type all \
--data_dir datasets/nlvr2/ft_corpus \
--model_type bert \
--task_name nlvr \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--per_gpu_train_batch_size 72 \
--per_gpu_eval_batch_size 72 \
--num_train_epochs 20 \
--output_dir model/nlvr2/student  \
--seed 88 \
--save_epoch 1 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--use_pair \
--use_label_seq \
--teacher_model model/nlvr2/teacher \
--student_model pretrained_models/base-vg-labels/ep_107_1192087 \
--alpha 0.5 \
--temperature 5.0 \
--num_hidden_layers 6 \
--teacher_learning_rate 3e-5 \
--student_learning_rate 3e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25

# Image-Text Retrieval
python oscar/run_retrieval_with_mmkd.py \
--do_train \
--do_lower_case \
--evaluate_during_training \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--weight_decay 0.05 \
--save_steps 5000 \
--add_od_labels \
--od_label_type vg \
--max_seq_length 70 \
--output_dir model/coco_ir/student \
--teacher_model model/coco_ir/teacher \
--student_model pretrained_models/base-vg-labels/ep_67_588997 \
--alpha 0.5 \
--temperature 5.0 \
--num_hidden_layers 6 \
--teacher_learning_rate 2e-5 \
--student_learning_rate 2e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25
```

# Inference

Run command below to obtain predictions of the distilled student model for each task.

```
# VQA 
python oscar/run_vqa.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 50 \
--data_label_type mask \
--data_dir datasets/vqa/2k \
--model_type bert \
--model_name_or_path model/vqa/student \
--task_name vqa_text \
--do_test \
--do_lower_case \
--max_seq_length 128 \
--per_gpu_eval_batch_size 32 \
--output_dir model/vqa/student \
--label_file datasets/vqa/cache/trainval_ans2label.pkl \
--label2ans_file datasets/vqa/cache/trainval_label2ans.pkl \
--classifier linear \
--cls_hidden_scale 3

# GQA
python oscar/run_gqa.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 45 \
--train_data_type all \
--eval_data_type bal \
--test_data_type all  \
--data_dir datasets/GQA/0.4true \
--model_type bert \
--model_name_or_path model/gqa/student \
--task_name gqa \
--do_test \
--do_lower_case \
--max_seq_length 165 \
--per_gpu_eval_batch_size 48 \
--output_dir model/gqa/student \
--label_file datasets/GQA/questions1.2/trainval_testdev_all_ans2label.pkl \
--label2ans_file datasets/GQA/questions1.2/trainval_testdev_all_label2ans.pkl \
--classifier linear \
--cls_hidden_scale 3 

# NLVR2
python oscar/run_nlvr.py \
--img_feature_dim 2054 \
--img_feature_type faster_r-cnn \
--max_img_seq_length 40 \
--eval_data_type all \
--test_data_type all \
--data_dir datasets/nlvr2/ft_corpus \
--model_type bert \
--model_name_or_path model/nlvr2/student \
--task_name nlvr \
--do_test \
--do_lower_case \
--max_seq_length 55 \
--per_gpu_eval_batch_size 72 \
--output_dir model/nlvr2/student \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--use_pair \
--use_label_seq

# Image-Text Retrieval
# inference on COCO 1k test set
python oscar/run_retrieval.py \
--do_test \
--do_eval \
--test_split test \
--num_captions_per_img_val 5 \
--eval_img_keys_file test_img_keys_1k.tsv \
--cross_image_eval \
--per_gpu_eval_batch_size 32 \
--eval_model_dir model/coco_ir/student
# inference on COCO 5k test set
python oscar/run_retrieval.py \
--do_test \
--do_eval \
--test_split test \
--num_captions_per_img_val 5 \
--eval_img_keys_file test_img_keys.tsv \
--cross_image_eval \
--per_gpu_eval_batch_size 32 \
--eval_model_dir model/coco_ir/student
```



