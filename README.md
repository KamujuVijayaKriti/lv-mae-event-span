# Event-Span Masking for Self-Supervised Long-Video Representation Learning and Violence Classification

**Project website:** https://kamujuvijayakriti.github.io/lv-mae-event-span/

Event-span masking for long-video masked autoencoding, built on LV-MAE (Naiman et al., ICCV 2025, https://github.com/amazon-science/lv-mae). During pretraining it finds the events in each video and masks whole, distinctive events. The masking is training-free: it adds no parameters to LV-MAE. The pretrained encoder is then used to classify long videos from XD-Violence into 7 categories.

## Method (`data/event_span_masking.py`)

1. **Change score** for every 5-s segment: `s_i = 1 - cos(e_i, e_{i-1})`, with `s_1 = 0` (LanguageBind 1024-d embeddings).
2. **Adaptive threshold:** `θ = mean(s) + 0.5 * std(s)`.
3. **Events:** a new event starts wherever `s_i > θ`, with boundaries at least 2 segments apart.
4. **Event distinctiveness:** mean-pool the (L2-normalised) segment embeddings of each event and of the whole video; `d_k = 1 - cos(event mean, video mean)`, z-scored across the video's events.
5. **Stochastic event ranking:** Gumbel-top-k sampling from `softmax(z / τ)`, with τ annealed from 2.0 to 0.25 by a cosine schedule over pretraining.
6. **Whole-event masking:** events are masked in that order until 50% of the segments are hidden (at most 40% of the video from one event); any remainder is topped up uniformly. Videos shorter than 8 segments use a per-segment fallback.

## Dataset: XD-Violence

https://roc-ng.github.io/XD-Violence/ (Wu et al., ECCV 2020)

- Official split: 3,954 train / 800 test videos (3,950 / 799 with embeddings here).
- 7 classes: Normal, Fighting (B1), Shooting (B2), Riot (B4), Abuse (B5), Car accident (B6), Explosion (G).
- Labels come from the file name (`..._label_B1-0-0`); multi-label videos use their first (most important) label.
- Video stream only (audio and optical flow are not used).

## Pipeline

Every step is a PBS job; submit from the repository folder.

```bash
conda env create -f environment.yml -n lvmae   # once; scripts activate ${CONDA_ENV:-lvmae}

qsub get_xd.sh             # download videos + train/test lists (Hugging Face mirror jherng/xd-violence)
qsub pbs_extract_xd.sh     # LanguageBind-Video-Huge-V1.5-FT, one 1024-d vector per 5 s (8 frames); resumable
qsub pbs_build_xd_csv.sh   # xd_train.csv / xd_test.csv

XD="LVMAE_TRAIN_CSV=$HOME/XD_Violence/xd_train.csv,LVMAE_VAL_CSV=$HOME/XD_Violence/xd_test.csv"
qsub -v EPOCHS=151,BATCH=4,ACCUM=4,LVMAE_TAG=_xd16,$XD pbs_pretrain_masking.sh
qsub -v M=event_span_xd16,SEED=0,LVMAE_NUM_CLASSES=7,$XD pbs_probe_ap.sh
```

Check job status with `qstat -u $USER`.

- **Pretraining:** 151 epochs (40 warm-up), mask ratio 0.5, effective batch 16 (4 × 4 accumulation), MSE on masked segments, encoder 32 layers / 16 heads, decoder 8 layers / 512-d.
- **Evaluation:** attentive probe (`test_ap.py`), 60 epochs.

## Result: XD-Violence (7 classes, 799 test videos)

| Method | Test accuracy |
|---|---|
| LV-MAE + event-span masking | **87.73%** |

Best test accuracy over 60 probe epochs, following the LV-MAE evaluation protocol (XD-Violence has no separate validation split). Single pretraining run.

## Credits

- Model, pretraining and probe code: LV-MAE (amazon-science/lv-mae), see LICENSE.
- Video encoder: LanguageBind (Zhu et al., ICLR 2024).
- Gumbel-top-k: Kool et al., ICML 2019.
- Change-point idea inspired by ABD (Du et al., CVPR 2022).
- Dataset: XD-Violence (Wu et al., ECCV 2020).
