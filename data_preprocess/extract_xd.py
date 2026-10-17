"""LanguageBind features for XD-Violence.
5-second windows, 8 uniformly sampled frames, LanguageBind-Video-Huge-V1.5-FT, one 1024-d vector per window.
Output: <OUT_ROOT>/<video_stem>/<SSSSS-EEEEE>/langbind_output.pt   (finished segments are skipped -> resumable)
Args: VIDEO_ROOT OUT_ROOT TASK_ID NUM_TASKS
"""
import glob, os, sys
import numpy as np
import torch
torch.backends.cudnn.enabled = False   # avoid conv3d segfault on this cluster
SEG, MIN_SEG, BATCH = 5, 1.0, 16


def windows(n_frames, fps, num_frames=8):
    dur = n_frames / fps
    out, k = [], 0
    while k * SEG < dur:
        s, e = k * SEG, min((k + 1) * SEG, dur)
        if e - s >= MIN_SEG:
            f0, f1 = int(round(s * fps)), min(int(round(e * fps)), n_frames)
            if f1 > f0:
                out.append((s, min((k + 1) * SEG, int(dur)), np.linspace(f0, f1 - 1, num_frames, dtype=int)))
        k += 1
    return out

def read_frames(path, wanted):
    import cv2
    cap = cv2.VideoCapture(path)
    wanted, frames, i = set(wanted), {}, 0
    last = max(wanted) if wanted else -1
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if frames:
        lastgood = frames[max(frames)]
        for w in wanted:
            frames.setdefault(w, lastgood)
    return frames

def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    return n, fps


def main():
    video_root, out_root, task, ntask = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    vids = sorted(glob.glob(os.path.join(video_root, '**', '*.mp4'), recursive=True))
    vids = [v for v in vids if not os.path.basename(v).startswith('._')][task::ntask]
    print(f'task {task}/{ntask}: {len(vids)} videos', flush=True)

    from languagebind import LanguageBind, to_device, transform_dict
    torch.backends.cudnn.enabled = False
    model = LanguageBind(clip_type={'video': 'LanguageBind_Video_Huge_V1.5_FT'}, cache_dir='./cache_dir').cuda().eval()
    proc = transform_dict['video'](model.modality_config['video'])
    nf = model.modality_config['video'].vision_config.num_frames

    done = skipped = failed = 0
    for vi, path in enumerate(vids):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            n, fps = video_info(path)
            todo = []
            for s, e, ids in windows(n, fps, nf):
                d = os.path.join(out_root, name, f'{int(s):05d}-{int(e):05d}')
                if os.path.isfile(os.path.join(d, 'langbind_output.pt')):
                    skipped += 1
                else:
                    todo.append((s, e, ids, d))
            for b in range(0, len(todo), 64 * BATCH):                  # read long videos in chunks (memory)
                chunk = todo[b:b + 64 * BATCH]
                frames = read_frames(path, [int(i) for _, _, ids, _ in chunk for i in ids])
                for c in range(0, len(chunk), BATCH):
                    part = chunk[c:c + BATCH]
                    pix = torch.stack([proc.transform(torch.stack([frames[int(i)] for i in ids]).permute(3, 0, 1, 2))
                                       for _, _, ids, _ in part])
                    with torch.no_grad():
                        emb = model({'video': to_device({'pixel_values': pix}, 'cuda')})['video'].float().cpu()
                    for k, (s, e, _, d) in enumerate(part):
                        os.makedirs(d, exist_ok=True)
                        torch.save({'embedding': emb[k].clone(), 'start': int(s), 'end': int(e)},
                                   os.path.join(d, 'langbind_output.pt'))
                        done += 1
        except Exception as ex:
            failed += 1
            print(f'[FAILED] {path}: {ex!r}', flush=True)
        if vi % 20 == 0:
            print(f'{vi + 1}/{len(vids)} videos | new segments {done} | skipped {skipped} | failed videos {failed}', flush=True)
    print(f'TASK DONE: new {done}, skipped {skipped}, failed videos {failed}', flush=True)


if __name__ == '__main__':
    main()
