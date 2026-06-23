#!/usr/bin/env bash
# Refined cut-paste IR A/B on fossil's 5060, on CURRENT data.
# B = live-017 (real baseline), E = live-018 (real + species-balanced, non-overlapping cut-paste).
set -e
cd ~/Code/aviary
PY=.venv-train/bin/python
export PYTHONPATH=training:server

echo "[ir-test] $(date) re-prep current data (budgie 150 / cockatiel 300+)"
$PY training/scripts/prepare_dataset.py --source data/annotation/raw --output data/training/datasets/live --model live

echo "[ir-test] $(date) per-class boxes after re-prep:"
cat data/training/datasets/live/labels/*/*.txt | awk '{print $1}' | sort -n | uniq -c

echo "[ir-test] $(date) train B (live-017 baseline)"
$PY training/scripts/train.py --data data/training/datasets/live/dataset.yaml --model yolo11n.pt \
  --name live_017 --export-to data/models/live-017.pt --epochs 200 --patience 50 \
  --batch 8 --imgsz 1280 --seed 0 --cls-pw 0.5 --hsv-h 0.0 --device cuda

echo "[ir-test] $(date) generate species-balanced composites"
$PY training/scripts/cut_paste_ir.py --dataset-dir data/training/datasets/live \
  --output-dir data/training/_cutpaste_stage --count 300 --seed 0
cp data/training/_cutpaste_stage/images/* data/training/datasets/live/images/train/
cp data/training/_cutpaste_stage/labels/* data/training/datasets/live/labels/train/
rm -f data/training/datasets/live/labels/train.cache

echo "[ir-test] $(date) train E (live-018 cut-paste)"
$PY training/scripts/train.py --data data/training/datasets/live/dataset.yaml --model yolo11n.pt \
  --name live_018 --export-to data/models/live-018.pt --epochs 200 --patience 50 \
  --batch 8 --imgsz 1280 --seed 0 --cls-pw 0.5 --hsv-h 0.0 --device cuda

echo "[ir-test] $(date) benchmark B vs E (real IR, val+test, conf 0.4)"
$PY training/scripts/benchmark.py --models live-017 --models live-018 --series live --split test \
  --conf 0.4 --imgsz 1280 --device cuda --output data/models/_irtest_test.json
$PY training/scripts/benchmark.py --models live-017 --models live-018 --series live --split val \
  --conf 0.4 --imgsz 1280 --device cuda --output data/models/_irtest_val.json

$PY - <<'PYEOF'
import json
T={m['name']:m for m in json.load(open('data/models/_irtest_test.json'))['series'][0]['models']}
V={m['name']:m for m in json.load(open('data/models/_irtest_val.json'))['series'][0]['models']}
def agg(model,scope,key):
    tp=fp=fn=0
    for S in (T,V):
        m=S[model]; n=m['labels'].get(key) if scope=='label' else (m['byCategory'].get(key) if scope=='cat' else m['overall'])
        if n: tp+=n['tp']; fp+=n['fp']; fn+=n['fn']
    return (tp/(tp+fn)) if tp+fn else None
print("==== IR TEST: B(live-017) -> E(live-018 balanced cut-paste)  val+test conf0.4 ====")
for lbl,sc,k in [("budgie","label","budgie"),("cockatiel","label","cockatiel"),("lovebird","label","lovebird"),("IR cat","cat","ir"),("day cat","cat","day"),("OVERALL","overall","")]:
    b=agg('live-017',sc,k); e=agg('live-018',sc,k)
    print(f"{lbl:11} " + ("n/a" if b is None or e is None else f"B {b*100:4.0f}% -> E {e*100:4.0f}%   d={(e-b)*100:+.0f}"))
PYEOF
echo "IR_TEST_DONE $(date)"
