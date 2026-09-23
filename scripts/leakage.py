from __future__ import annotations
import json, hashlib, argparse
from pathlib import Path

def norm_task(r):
 t=r['task']; return json.dumps({k:t[k] for k in ['width','height','walls','start','key','door','target']},sort_keys=True)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); ap.add_argument('--out',default='results/leakage_audit.json'); args=ap.parse_args(); root=Path(args.data)
 splits=['train','validation','iid_test','novel_layout_test','novel_object_test','novel_task_test','long_horizon_test','distractor_test','partial_observation_test','final_independent_test']; tasks={}; seqs={}; sg={}
 for s in splits:
  rows=[json.loads(x) for x in (root/f'{s}.jsonl').read_text().splitlines() if x.strip()]; tasks[s]={norm_task(r) for r in rows}; seqs[s]={tuple(r['expert_action'] for r in [r]) for r in []}; sg[s]={(norm_task(r),r['step']) for r in rows}
 pairs=[]
 for i,a in enumerate(splits):
  for b in splits[i+1:]:
   overlap=len(tasks[a]&tasks[b]);
   if overlap: pairs.append({'a':a,'b':b,'duplicate_tasks':overlap})
 report={'splits':{s:len(tasks[s]) for s in splits},'cross_split_duplicate_tasks':pairs,'final_test_clean':not any(p['b']=='final_independent_test' for p in pairs),'protocol':'audit before final checkpoint selection'}
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
