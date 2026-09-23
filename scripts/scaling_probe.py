from pathlib import Path
import json,random,subprocess
ROOT=Path('/home/ubuntu/cnfg_agent'); src=ROOT/'data'; base=ROOT/'data_scaling'; out=ROOT/'results/scaling'; out.mkdir(parents=True,exist_ok=True); train=[json.loads(x) for x in (src/'train.jsonl').read_text().splitlines()]
for frac in [0.01,0.05,0.10,0.25,0.50,1.0]:
 name=f'{int(frac*100)}pct'; d=base/name; d.mkdir(parents=True,exist_ok=True); n=max(1,int(len(train)*frac)); (d/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in train[:n])+'\n'); (d/'validation.jsonl').write_text((src/'validation.jsonl').read_text());
 for f in ['iid_test','novel_layout_test','novel_task_test','long_horizon_test','partial_observation_test']:
  (d/f'{f}.jsonl').write_text((src/f'{f}.jsonl').read_text())
 ck=out/f'cnfg_{name}_seed42'; subprocess.run(['python3',str(ROOT/'scripts/train.py'),'--data',str(d),'--out',str(out),'--model','cnfg','--seed','42','--epochs','5','--batch','1024','--dim','64'],check=True)
 # train.py name is cnfg_seed42; rename to preserve fraction
 for ext in ['pt','json']:
  p=out/f'cnfg_seed42.{ext}'; p.rename(out/f'cnfg_{name}_seed42.{ext}')
 # Evaluate IID with fixed 100 tasks.
 subprocess.run(['python3',str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(out/f'cnfg_{name}_seed42.pt'),'--model','cnfg','--data',str(d),'--out',str(out/f'{name}_eval'),'--split','iid_test','--max-tasks','100'],check=True,stdout=subprocess.DEVNULL)
