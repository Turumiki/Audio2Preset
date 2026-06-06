#!/usr/bin/env python3
"""A2: Train a neural inverse-synthesis model (spectrogram -> synth params).

Reads a dataset produced by the C++ A1 generator:
    --gen-dataset <vst3> --out <dir> ...
which writes, in <dir>:
    meta.txt        # T, K, paramCount, param indices/names
    features.f32    # N * (T*K) float32, row-major [T][K]
    params.f32      # N * paramCount float32 in [0,1]

Trains a small 2D CNN (spectrogram image -> parameter vector) and exports ONNX
for the C++ app's "Neural init" (A3).

Usage:
    python train_inverse.py --data path/to/dir [--epochs 60] [--batch 128] [--out model.onnx]

Requires: torch (CUDA build for the RTX 4080), numpy.
"""
import argparse, os, re, numpy as np

def read_meta(path):
    txt = open(os.path.join(path, "meta.txt"), encoding="utf-8", errors="replace").read()
    def grab(key, cast):
        m = re.search(rf"{key}=([^\s,]+)", txt)
        return cast(m.group(1)) if m else None
    T = grab("T", int); K = grab("K", int)
    P = grab("paramCount", int); N = grab("N", int)
    idx_m = re.search(r"paramIndices=([0-9,]+)", txt)
    indices = [int(x) for x in idx_m.group(1).split(",") if x != ""] if idx_m else []
    return dict(T=T, K=K, P=P, N=N, indices=indices)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split

    meta = read_meta(args.data)
    T, K, P = meta["T"], meta["K"], meta["P"]
    feat = np.fromfile(os.path.join(args.data, "features.f32"), dtype=np.float32)
    par  = np.fromfile(os.path.join(args.data, "params.f32"),   dtype=np.float32)
    N = feat.size // (T * K)
    feat = feat[: N * T * K].reshape(N, 1, T, K)   # (N,1,T,K)
    par  = par[: N * P].reshape(N, P)
    print(f"loaded N={N} feat={T}x{K} P={P}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, (torch.cuda.get_device_name(0) if dev == "cuda" else ""))

    ds = TensorDataset(torch.from_numpy(feat), torch.from_numpy(par))
    n_val = max(1, int(N * 0.1))
    tr, va = random_split(ds, [N - n_val, n_val])
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(dev == "cuda"))
    vl = DataLoader(va, batch_size=args.batch)

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

    net = Net(P).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossfn = nn.L1Loss()  # mean abs param error (matches the C++ "param score")

    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = net(x); loss = lossfn(out, y)
            loss.backward(); opt.step(); tot += loss.item() * x.size(0)
        sched.step()
        net.eval(); vtot = 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                vtot += lossfn(net(x), y).item() * x.size(0)
        print(f"epoch {ep+1}/{args.epochs}  train L1={tot/(N-n_val):.4f}  val L1={vtot/n_val:.4f}")

    net.eval()
    ckpt = os.path.join(args.data, "inverse.pt")
    torch.save(net.state_dict(), ckpt)
    print("saved checkpoint ->", ckpt)

    vtot = 0.0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(dev), y.to(dev)
            vtot += lossfn(net(x), y).item() * x.size(0)
    final_val = vtot / n_val
    print(f"==== FINAL val L1 (mean param error) = {final_val:.4f}  (param score ~ {100*(1-final_val):.1f}/100) ====")

    out = args.out or os.path.join(args.data, "inverse.onnx")
    try:
        dummy = torch.zeros(1, 1, T, K, device=dev)
        torch.onnx.export(net, dummy, out, input_names=["spec"], output_names=["params"],
                          dynamic_axes={"spec": {0: "batch"}, "params": {0: "batch"}}, opset_version=17)
        print("exported ONNX ->", out, f"(input 1x1x{T}x{K}, output {P} params)")
    except Exception as ex:
        print("ONNX export failed (checkpoint .pt saved):", ex)
    print("param indices for A3:", meta["indices"])

if __name__ == "__main__":
    main()
