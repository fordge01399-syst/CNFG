from __future__ import annotations
import sys, json, random, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.environment.gridworld import *
from src.cnfg.agent_model import encode_observation

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data'); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--train',type=int,default=20000); ap.add_argument('--valid',type=int,default=3000); ap.add_argument('--test',type=int,default=1000); ap.add_argument('--final',type=int,default=5000); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); rng=random.Random(args.seed)
    counts={'train':args.train,'validation':args.valid,'iid_test':args.test,'novel_layout_test':args.test,'novel_object_test':args.test,'novel_task_test':args.test,'long_horizon_test':args.test,'distractor_test':args.test,'partial_observation_test':args.test,'final_independent_test':args.final}
    meta={'seed':args.seed,'counts':counts,'transition_schema':'observation/state-goal features -> expert action; no optimal plan in model input'}
    for split,n in counts.items():
        rows=[]; level='medium'
        if split=='long_horizon_test': level='large'
        if split=='final_independent_test': level='large'
        for i in range(n):
            task=make_task(rng,level,i); plan=shortest_plan(task)
            # For long horizon, reject too-short tasks when possible.
            if split in {'long_horizon_test','final_independent_test'}:
                for _ in range(20):
                    if len(plan)>=10: break
                    task=make_task(rng,'large',i); plan=shortest_plan(task)
            obs='partial' if split=='partial_observation_test' or (split in {'train','validation'} and i % 4 == 0) else 'full'
            env=GridWorld(task,observation=obs,rng=random.Random(args.seed+i))
            env.reset(); memless=[]
            for step,action in enumerate(plan):
                row={'task_id':f'{split}-{i:06d}','step':step,'split':split,'seed':args.seed,'task':asdict(task),'features':encode_observation(env),'valid_actions':env.valid_actions(),'expert_action':action,'optimal_steps':len(plan),'observation':env.observe()}
                rows.append(row); env.step(action)
        with (out/f'{split}.jsonl').open('w') as f:
            for r in rows: f.write(json.dumps(r,separators=(',',':'))+'\n')
        meta[split]={'episodes':n,'transitions':len(rows)}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))

def encode_state_goal(env):
    s=env.state;t=env.task;W=max(1,t.width-1);H=max(1,t.height-1)
    return [s.pos[0]/W,s.pos[1]/H,t.target[0]/W,t.target[1]/H,t.key[0]/W,t.key[1]/H,t.door[0]/W,t.door[1]/H,float(s.has_key),float(s.door_open),float(s.pos==t.key),float(s.pos==t.door)]
if __name__=='__main__': main()
