import os, time, sys
sys.path.insert(0, '/rlwrld1/home/yashu/rlwrld_isaac/gr00t/automating_groot')
from notify import notify_checkpoint_saved

output_dir = "/rlwrld1/home/yashu/output/train/Cylinder_Tube_Place_test_100pct"
dataset    = "Cylinder_Tube_Place_test"
pct        = 100
notified   = set()

print('[watcher] Started, watching:', output_dir, flush=True)

while True:
    time.sleep(30)
    try:
        entries = os.listdir(output_dir)
    except FileNotFoundError:
        continue
    for entry in sorted(entries):
        if entry.startswith('checkpoint-') and entry not in notified:
            step = int(entry.split('-')[1])
            ckpt_path = os.path.join(output_dir, entry)
            print(f'[watcher] New checkpoint: {entry}', flush=True)
            notify_checkpoint_saved(dataset, pct, step, ckpt_path)
            notified.add(entry)
    if os.path.exists(os.path.join(output_dir, '.training_done')):
        print('[watcher] Done flag detected, exiting', flush=True)
        break
