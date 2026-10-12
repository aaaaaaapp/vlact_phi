#!/usr/bin/env python
import argparse, os, torch
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--mode',choices=['tanh','sigmoid'],required=True); p.add_argument('--alpha-max',type=float,default=.2); a=p.parse_args()
x=torch.load(a.input,map_location='cpu',weights_only=False)
if not isinstance(x,dict): raise TypeError(type(x))
for k,v in list(x.items()):
    if not torch.is_tensor(v) or ('pi_action_layer_gates' not in k and 'pi_physics_layer_gates' not in k): continue
    z=v.float(); z=a.alpha_max*torch.tanh(z) if a.mode=='tanh' else a.alpha_max*torch.sigmoid(z); x[k]=z.to(v.dtype)
os.makedirs(os.path.dirname(a.output),exist_ok=True); torch.save(x,a.output); print(a.mode,a.alpha_max,a.output)
