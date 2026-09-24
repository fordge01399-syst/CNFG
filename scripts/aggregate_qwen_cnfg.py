import json,glob,statistics
from pathlib import Path
out={}
for split in ['long','independent']:
 vals=[];bad=[]
 for p in sorted(glob.glob('results/qwen_cnfg_dagger/seed*_'+split+'.log')):
  txt=Path(p).read_text(); data=json.loads(txt[txt.index('{'):]); vals.append(data['full_task_success']); bad.append(data['invalid_action_rate'])
 out[split]={'n_seeds':len(vals),'tasks_per_seed':2,'success_mean':statistics.mean(vals) if vals else None,'success_sd':statistics.stdev(vals) if len(vals)>1 else 0.0,'invalid_mean':statistics.mean(bad) if bad else None,'invalid_sd':statistics.stdev(bad) if len(bad)>1 else 0.0}
Path('results/qwen_cnfg_dagger/summary_five_seed.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
