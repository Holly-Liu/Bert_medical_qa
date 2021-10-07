python run.py \
  --device cuda:3 \
  --debug_short_set -1 \
  --model bert_mean_pool \
  --dataset unsup_cMedQA_dev_CHIPSTS \
  --data_method unsup_cMedQA_dev_CHIPSTS \
  --train_method unsup_simcse \
  --eval_method auc \
  --bert_path "/home/users/liuhongli/Models/prev_trained_model/roberta_wwm_ext" \
  --log_name UNSUP_CMEDQA_DEV_CHIPSTS_ROBERTA_WWM_F1_AUC_EP20 \
  --save_path "/home/users/liuhongli/Models/prev_trained_model/roberta_wwm_sts_auc/" \
  --dev_step 20 \
  --test_step 1000000 \
  --require_improvement 1000 \
  --num_epochs 20 \
  --batch_size 96 \
  --pad_size 32 \
  --learning_rate 2e-05




