conda activate adaptesupar

#### 1. configuration for ADAPT-ESUPAR
### Step 1: DOMAIN TUNING
python run_mlm.py \
    --model_name_or_path KoichiYasuoka/bert-base-japanese-char-extended \
    --train_file adapt_yasuoka_char_cvg_train.txt \
    --per_device_train_batch_size 30 \
    --max_seq_length 50 \
    --eval_steps 100000 \
    --save_steps 100000 \
    --save_total_limit 1 \
    --load_best_model \
    --evaluation_strategy steps \
    --fp16 \
    --do_train \
    --do_eval \
    --output_dir jerrybonnell053122/adapt_yasuoka_char_cvg_50

### Step 2: TASK-SPECIFIC TUNING
python train.py jerrybonnell053122/adapt_yasuoka_char_cvg_50 jerrybonnell053122/adapt_yasuoka_char_cvg_50_upos ../../../../bccwj/bccwj/UD_Japanese-BCCWJGSD

#### 2. configuration for FINETUNED-ESUPAR
### Step 1: TASK-SPECIFIC TUNING ONLY
python train.py KoichiYasuoka/bert-base-japanese-char-extended jerrybonnell/baseline-bert-base-japanese-upos ../../../bccwj/bccwj/UD_Japanese-BCCWJGSD

#### 3. configuration for SUB-ESUPAR
### Step 1: DOMAIN TUNING
python run_mlm.py \
    --model_name_or_path KoichiYasuoka/bert-base-japanese-char-extended \
    --train_file baseline_bccwj_yasuoka_char_cvg_train.txt \
    --per_device_train_batch_size 30 \
    --max_seq_length 50 \
    --eval_steps 10000 \
    --save_steps 10000 \
    --save_total_limit 1 \
    --load_best_model \
    --evaluation_strategy steps \
    --fp16 \
    --do_train \
    --do_eval \
    --output_dir jerrybonnell053122/baseline_bccwj_yasuoka_char_cvg

### Step 2: TASK-SPECIFIC TUNING
python train.py jerrybonnell053122/baseline_bccwj_yasuoka_char_cvg jerrybonnell053122/baseline_bccwj_yasuoka_char_cvg_upos ../../../../bccwj/bccwj/UD_Japanese-BCCWJGSD

#### 4. configuration for OMIT-ESUPAR
### Step 1: DOMAIN TUNING
python run_mlm.py \
    --model_name_or_path KoichiYasuoka/bert-base-japanese-char-extended \
    --train_file baseline_notaiyo_yasuoka_char_cvg_train.txt \
    --per_device_train_batch_size 30 \
    --max_seq_length 50 \
    --eval_steps 10000 \
    --save_steps 10000 \
    --save_total_limit 1 \
    --load_best_model \
    --evaluation_strategy steps \
    --fp16 \
    --do_train \
    --do_eval \
    --output_dir jerrybonnell053122/baseline_notaiyo_yasuoka_char_cvg

### Step 2: TASK-SPECIFIC TUNING
python train.py jerrybonnell053122/baseline_notaiyo_yasuoka_char_cvg jerrybonnell053122/baseline_notaiyo_yasuoka_char_cvg_upos ../../../../bccwj/bccwj/UD_Japanese-BCCWJGSD
