device=cuda:2
log_name_prefix=MTBERT_CONS_
model=bert_mean_pool

for exp in {1..10}
do
  python run.py \
    --log_name ${log_name_prefix}_bash_exp${exp}_gama_`0.1 * ${exp}` \
    --model $model \
    --device $device \
    --learning_rate 2e-5 \
    --gama `0.1 * ${exp}` \
    --hitn 1 \
    --require_improvement 5000 \
    --num_epochs 10 \
    --batch_size 32 \
    --pad_size 64 \
    --seed 1 \
    --dev_step 500 \
    --test_step 1000 \
    --train_method cons
done




