"""Train/test CSVs for XD-Violence from the official train/test lists.
7 classes: Normal + 6 violence types. Label = FIRST (most important) code in the file name, e.g.
'..._label_B2-G-0' -> Shooting.   Args: DATA_DIR EMB_ROOT OUT_DIR
DATA_DIR holds train_list.txt, test_list.txt and video/.
"""
import glob
import os
import sys

import pandas as pd

CODES = {'A': 'Normal', 'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot',
         'B5': 'Abuse', 'B6': 'Car accident', 'G': 'Explosion'}
LABELS = ['Normal', 'Fighting', 'Shooting', 'Riot', 'Abuse', 'Car accident', 'Explosion']
data_dir, emb_root, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]


def codes_of(name):
    return [c for c in name.rsplit('_label_', 1)[-1].split('-') if c != '0']


def video_names(split):
    lst = os.path.join(data_dir, f'{split}_list.txt')
    if os.path.isfile(lst):
        names = [os.path.basename(l.split()[0]) for l in open(lst) if l.strip()]
    else:                                   # fallback: test = files under a *test* folder
        vids = glob.glob(os.path.join(data_dir, 'video', '**', '*.mp4'), recursive=True)
        names = [os.path.basename(v) for v in vids
                 if ('test' in os.path.basename(os.path.dirname(v)).lower()) == (split == 'test')]
    return sorted({n[:-4] if n.endswith('.mp4') else n for n in names})


def rows(split):
    out, missing, multi = [], [], 0
    for name in video_names(split):
        codes = codes_of(name)
        label_name = CODES.get(codes[0]) if codes else None
        segs = sorted(glob.glob(os.path.join(emb_root, glob.escape(name), '*', 'langbind_output.pt')),
                      key=lambda p: int(os.path.basename(os.path.dirname(p)).split('-')[0]))
        if label_name is None or len(segs) < 1:
            missing.append(name)
            continue
        multi += len(codes) > 1
        out.append(dict(video_filename=name, segment_paths=str(segs), number_of_segments=len(segs),
                        label=LABELS.index(label_name), label_name=label_name))
    return out, missing, multi


os.makedirs(out_dir, exist_ok=True)
for split in ['train', 'test']:
    r, miss, multi = rows(split)
    df = pd.DataFrame(r)
    df.insert(0, 'segment_idx', range(len(df)))
    df.to_csv(os.path.join(out_dir, f'xd_{split}.csv'), index=False)
    print(f'{split}: {len(df)} videos, {df.number_of_segments.sum()} segments, '
          f'multi-label videos {multi}, missing/unlabelled: {len(miss)}')
    print('  per class:', df.label_name.value_counts().sort_index().to_dict())
    if miss:
        print('  first missing:', miss[:5])
