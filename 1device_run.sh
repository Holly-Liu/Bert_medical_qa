base_path="/home/users/liuhongli/Models/"
prev_trained_model=$base_path/prev_trained_model
output=$base_path/Bert-Chinese-Test-Classification-Pytorch/output/

python run.py \
  --device cuda:1 \
  --model bert_cls \
  --dataset cMedQA \
  --data_method cMedQA_pos_ans_only \
  --train_method cons \
  --eval_method hitn \
  --bert_path $prev_trained_model/mc-bert-torch/ \
  --log_name cMedQA_MC_CLS_HITN_CONS \
  --save_path $output/mc-bert-cMedQA-CLS/ \
  --debug_short_set "-1" \
  --dev_step 1000000 \
  --test_step 1000000 \
  --require_improvement 1000 \
  --num_epochs 20 \
  --batch_size 32 \
  --pad_size 96 \
  --learning_rate 2e-05




