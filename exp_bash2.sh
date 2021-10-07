device=cuda:2
log_name_prefix=MTBERT_CONS_
model=bert_mean_pool


python run.py \
  --log_name ${log_name_prefix}_bash_exp1_gama_0.1 \
  --model $model \
  --device $device \
  --learning_rate 2e-5 \
  --gama 0.1 \
  --hitn 1 \
  --require_improvement 5000 \
  --num_epochs 3 \
  --batch_size 32 \
  --pad_size 64 \
  --seed 1 \
  --dev_step 500 \
  --test_step 1000 \
  --train_method cons

python run.py \
  --log_name ${log_name_prefix}_bash_exp1_gama_0.3 \
  --model $model \
  --device $device \
  --learning_rate 2e-5 \
  --gama 0.3 \
  --hitn 1 \
  --require_improvement 5000 \
  --num_epochs 3 \
  --batch_size 32 \
  --pad_size 64 \
  --seed 1 \
  --dev_step 500 \
  --test_step 1000 \
  --train_method cons

python run.py \
  --log_name ${log_name_prefix}_bash_exp1_gama_0.6 \
  --model $model \
  --device $device \
  --learning_rate 2e-5 \
  --gama 0.6 \
  --hitn 1 \
  --require_improvement 5000 \
  --num_epochs 3 \
  --batch_size 32 \
  --pad_size 64 \
  --seed 1 \
  --dev_step 500 \
  --test_step 1000 \
  --train_method cons

python run.py \
  --log_name ${log_name_prefix}_bash_exp1_gama_0.9 \
  --model $model \
  --device $device \
  --learning_rate 2e-5 \
  --gama 0.9 \
  --hitn 1 \
  --require_improvement 5000 \
  --num_epochs 3 \
  --batch_size 32 \
  --pad_size 64 \
  --seed 1 \
  --dev_step 500 \
  --test_step 1000 \
  --train_method cons



