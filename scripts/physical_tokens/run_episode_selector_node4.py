"""Eight GPU workers; preregistered selector settings and seed lists."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/mnt/jfzn/liujn/code/VLAct-PhysToken')
CKPT = ROOT/'results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt'
PANEL = {
 'handover_block': [100100000,100100005,100100006,100100009,100100015,100100023,100100027,100100028,100100036,100100039],
 'beat_block_hammer': [100150000,100150001,100150004,100150005,100150006,100150008,100150009,100150010,100150011,100150013],
 'click_bell': list(range(100100001,100100011)),
 'place_phone_stand': [100100002,100100003,100100005,100100006,100100007,100100009,100100010,100100013,100100015,100100017]}
HARD = [('click_alarmclock',s,'demo_randomized') for s in [100005,100017,100018,100022]] + [('click_bell',100008,'demo_randomized'),('hanging_mug',100002,'demo_randomized'),('hanging_mug',100003,'demo_randomized'),('press_stapler',100053,'demo_randomized')]

def main():
 stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
 out = ROOT/'logs'/f'episode_selector_node4_{stamp}'
 out.mkdir(parents=True)
 jobs=[]
 for label,threshold,radius in [('strict',0.10,0.20),('wide',0.30,0.50)]:
  for task,seed,cfg in HARD + [(t,s,'demo_clean') for t,seeds in PANEL.items() for s in seeds]:
   jobs.append(dict(label=label,threshold=threshold,radius=radius,task=task,seed=seed,config=cfg))
 (out/'preregistered_jobs.json').write_text(json.dumps(jobs,indent=2))
 print(f'OUT={out}',flush=True)
 def worker(gpu):
  for job in jobs[gpu::8]:
   tag=f"selector_{stamp}_{job['label']}_{job['task']}_{job['seed']}"
   env=dict(os.environ,GPU_ID=str(gpu),PORT=str(7400+gpu),TASK=job['task'],TASK_CONFIG=job['config'],SEED=str(job['seed']),TAG=tag,CKPT=str(CKPT),SELECTOR_THRESHOLD=str(job['threshold']),SELECTOR_RADIUS=str(job['radius']))
   print(f'START gpu={gpu} {tag}',flush=True)
   with (out/f'{tag}.log').open('w') as log:
    proc=subprocess.run(['bash',str(ROOT/'scripts/physical_tokens/run_policy_single_seed.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
   (out/f'{tag}.status.json').write_text(json.dumps(dict(job,returncode=proc.returncode,tag=tag)))
   print(f'END gpu={gpu} rc={proc.returncode} {tag}',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  list(pool.map(worker,range(8)))
 print('ALL_DONE',flush=True)

if __name__=='__main__': main()
