"""Numerical check of the weak-nonlinearity expansion (notes/kerr_fwm_attention_derivation.md): at reduced power,
generated comb lines should scale as P^3 and the Kerr correction of the driven lines as P. Noise off, trial 149 mask."""
import sys, math, torch, numpy as np
sys.path.insert(0, 'experiments'); sys.path.insert(0, '.')
import compare_encodings as ce, counterprop_comb as cc
from combrc import make_task
save = dict(ce.REALISTIC)
for k in ('rin', 'detuning', 'field'): ce.REALISTIC[k] = 0.0
base = {k: v for k, v in ce.sample('mask_drop', 149).items() if k in ce.CFG_KEYS}
u = torch.tensor(make_task('mackey_glass_h10', 700, seed=0)[0], dtype=torch.float64)[:700]
scales = [3e-3, 1e-2, 3e-2, 1e-1]
cfgs = [{**base, 'p_total': base['p_total'] * s} for s in scales]
lin = [{**c, 'group': 'lin_comb_mask'} for c in cfgs]
for c in lin: c['gamma'] = 0.0
L = cc.simulate(cfgs, 'uni', u, 1, 'cuda', 0, 512, 20)[:, :, 0].double()      # [4, 400, 62]
L0 = cc.simulate(lin, 'uni', u, 1, 'cuda', 0, 512, 20)[:, :, 0].double()
c = 31                                                                          # pump index in the 62 kept lines
gen = [c + 12, c - 12, c + 10]                                                   # lines not driven (|mu| > 8)
drv = [c + 3, c - 5, c + 8]
for i, s in enumerate(scales):
    pg = L[i][:, gen].mean().item(); p0g = L0[i][:, gen].mean().item()
    rel = ((L[i][:, drv] - L0[i][:, drv]).abs().mean() / L0[i][:, drv].mean()).item()
    print(f"power x{s:.0e}: generated-line power {pg:.3e} (linear cavity {p0g:.1e}); driven lines, Kerr change rel. {rel:.2e}")
P = np.array([L[i][:, gen].mean().item() for i in range(len(scales))]); R = np.array([((L[i][:, drv] - L0[i][:, drv]).abs().mean() / L0[i][:, drv].mean()).item() for i in range(len(scales))])
print("slope generated lines vs input power:", np.round(np.diff(np.log(P)) / np.diff(np.log(scales)), 2))
print("slope relative Kerr change of driven lines:", np.round(np.diff(np.log(R)) / np.diff(np.log(scales)), 2))
