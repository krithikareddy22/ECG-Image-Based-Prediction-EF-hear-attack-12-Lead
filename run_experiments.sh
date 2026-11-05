#!/usr/bin/env bash
# Full pipeline: preprocess -> baseline -> image CNNs -> 1-D CNN / header model / fusion -> comparison.
# On a 4-core CPU this takes ~5 hours; on a GPU a few minutes per model.
# Add --pretrained to the CNN lines when internet access to the timm/HF weights is available.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

[ -f data/manifest.csv ] || python -m ecg_ef.preprocess

run() { local tag=$1; shift; python -m ecg_ef.train "$@" 2>&1 | tee "logs/$tag.log"; }

python -m ecg_ef.baseline 2>&1 | tee logs/hog_svm.log
run ecgnet_filtered          --model ecgnet          --variant filtered
run ecgnet_raw               --model ecgnet          --variant raw
run resnet18_filtered        --model resnet18        --variant filtered
run efficientnet_b0_filtered --model efficientnet_b0 --variant filtered
run convnext_atto_filtered   --model convnext_atto   --variant filtered
run mobilenetv3_filtered     --model mobilenetv3     --variant filtered
run ecgnet_filtered_384      --model ecgnet          --variant filtered --img-size 384 768 --tag ecgnet_filtered_384
run convnext_atto_lr2e-4     --model convnext_atto   --variant filtered --lr 2e-4 --tag convnext_atto_filtered_lr2e-4

python -m ecg_ef.ensemble

# accuracy improvements: digitised waveforms (1-D CNN), header features, fusion
python -m ecg_ef.signals
python -m ecg_ef.train1d --tag resnet1d                    2>&1 | tee logs/resnet1d.log
python -m ecg_ef.train1d --rhythm --tag resnet1d_rhythm    2>&1 | tee logs/resnet1d_rhythm.log
python -m ecg_ef.train1d --seeds 5 --tag resnet1d_x5       2>&1 | tee logs/resnet1d_x5.log
python -m ecg_ef.header                                    2>&1 | tee logs/header.log
python -m ecg_ef.ensemble --members header_logreg resnet1d_x5 --tune-weight --name fusion_header_1d
python -m ecg_ef.ensemble --members header_logreg resnet18_filtered --tune-weight --name fusion_header_resnet18

python -m ecg_ef.compare
