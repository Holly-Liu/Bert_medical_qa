device=cuda:1
log_name_prefix=MTBERT_CONS_SIMCSE
model=bert_cons_learning


python run.py \
  --log_name ${log_name_prefix}_BS6 \
  --model $model \
  --device $device \
  --learning_rate 2e-5 \
  --gama 0.1 \
  --hitn 10 \
  --require_improvement 5000 \
  --num_epochs 3 \
  --batch_size 6 \
  --pad_size 150 \
  --seed 1 \
  --dev_step 100 \
  --test_step 1000 \
  --train_method simcse










