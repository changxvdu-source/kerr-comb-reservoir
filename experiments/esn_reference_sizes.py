"""Software reference for the counter-propagating comparison: ESN with 31 and 62 nodes (the readout dimension of
the one- and two-direction rings), 8 taps, linear ridge readout, input noise swept, paper's fresh-run protocol
(compare_encodings.esn_fresh: hyper-parameters on the validation part, test on new noise realisations).
Mackey-Glass, 10 steps ahead, geometric mean over 3 series.

    python experiments/esn_reference_sizes.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import compare_encodings as ce  # noqa: E402
from combrc import make_task  # noqa: E402
from combrc.evaluate import WARMUP  # noqa: E402

torch.set_num_threads(4)
for size in (31, 62):
    for noise in (0.0, 1e-4, 3e-4, 1e-3, 3e-3):
        v = []
        for seed in (0, 1, 2):
            u_np, y_np = make_task("mackey_glass_h10", WARMUP + ce.N_TOTAL, seed=seed)
            u = torch.tensor(u_np, dtype=torch.float64)
            y = torch.tensor(y_np, dtype=torch.float64)
            v.append(ce.esn_fresh(u, y, size, 8, noise, 6)[0])
        print(f"ESN {size:2d} nodes, input noise {noise:.0e}: {ce.gmean(v):.2e}", flush=True)
