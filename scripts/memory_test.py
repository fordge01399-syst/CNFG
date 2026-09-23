from __future__ import annotations
import sys,json,argparse,random,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.environment.gridworld import *
from src.cnfg.agent_model import CNFGAgent,encode_observation,action_mask
from src.environment.planner import shortest_plan_from_state

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data',default='data'); ap.add_argument('--split',default='iid_test'); ap.add_argument('--gap',type=int,default=4); ap.add_argument('--max-tasks',type=int,default=200); args=ap.parse_args(); model=CNFGAgent(); model.load_state_dict(torch.load(args.checkpoint,map_location='cpu')['model']); model.eval(); rows=[json.loads(x) for x in (Path(args.data)/f'{args.split}.jsonl').read_text().splitlines() if x.strip()][:args.max_tasks]; outcomes=[]
 for i,r in enumerate(rows):
  t=r['task']; task=Task(t['width'],t['height'],tuple(map(tuple,t['walls'])),tuple(t['start']),tuple(t['key']),tuple(t['door']),tuple(t['target']),tuple(map(tuple,t['distractors'])),t.get('task_type','key_door_reach')); env=GridWorld(task,'full',rng=random.Random(i)); env.reset(); memory=None; success=False; correct=0
  for step in range(max(100,(r.get('optimal_steps') or 20)*4)):
   env.observation='full' if step<=args.gap else 'partial'; x=torch.tensor([encode_observation(env)],dtype=torch.float32); mask=action_mask(env).unsqueeze(0); target=(shortest_plan_from_state(task,env.state) or [None])[0]
   with torch.no_grad(): logits,memory=model(x,memory,mask); a=ACTIONS[int(logits.argmax(1))]
   correct+=int(a==target); env.step(a)
   if env.state.done: success=True; break
  outcomes.append({'success':success,'step_accuracy':correct/max(1,step+1),'steps':env.state.steps})
 out={'gap':args.gap,'tasks':len(outcomes),'full_task_success':statistics.mean([int(x['success']) for x in outcomes]),'step_accuracy':statistics.mean([x['step_accuracy'] for x in outcomes]),'mean_steps':statistics.mean([x['steps'] for x in outcomes])}; p=Path('results')/f'memory_gap{args.gap}_seed{Path(args.checkpoint).stem.split('seed')[-1]}.json'; p.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
