#!/usr/bin/env python3
"""Predict synth params from a spectrogram feature using the trained model.

Usage:
    python predict.py --data <dataset_dir> --feat <feature.f32> [--out vals.txt]

Loads <dataset_dir>/inverse.pt (+ meta.txt for T,K,P and param indices), reads the
T*K float32 feature, prints the predicted param values (comma-separated) and the
param indices, and writes them to --out for the C++ --render-params tool.
"""
import argparse, os, re, numpy as np, torch, torch.nn as nn

def read_meta(path):
    txt = open(os.path.join(path, "meta.txt"), encoding="utf-8", errors="replace").read()
    g = lambda k, c: c(re.search(rf"{k}=([^\s,]+)", txt).group(1))
    idx = re.search(r"paramIndices=([0-9,]+)", txt).group(1)
    return g("T", int), g("K", int), g("paramCount", int), [int(x) for x in idx.split(",") if x]

class Net(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(),
                               nn.Dropout(0.2), nn.Linear(256, P), nn.Sigmoid())
    def forward(self, x): return self.f(self.c(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    T, K, P, idx = read_meta(a.data)
    net = Net(P); net.load_state_dict(torch.load(os.path.join(a.data, "inverse.pt"), map_location="cpu"))
    net.eval()
    x = np.fromfile(a.feat, dtype=np.float32)[: T * K].reshape(1, 1, T, K)
    with torch.no_grad():
        pred = net(torch.from_numpy(x)).numpy().reshape(-1)
    vals = ",".join(f"{v:.5f}" for v in pred)
    print("idx=" + ",".join(str(i) for i in idx))
    print("vals=" + vals)
    if a.out:
        open(a.out, "w").write(vals)
        print("wrote", a.out)

if __name__ == "__main__":
    main()
