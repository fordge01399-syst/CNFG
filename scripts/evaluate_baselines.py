from __future__ import annotations
import sys,json,argparse,random,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.environment.gridworld import *
from src.environment.planner import shortest_plan_from_state
from src.baselines.simple_baselines import RandomAgent,GreedyAgent

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); ap.add_argument('--split',default='iid_test'); ap.add_argument('--agent',choices=['random','greedy'],required=True); ap.add_argument('--max-tasks',type=int,default=200); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',default='results'); args=ap.parse_args()
 rows=[json.loads(x) for x in (Path(args.data)/f'{args.split}.jsonl').read_text().splitlines() if x.strip()][:args.max_tasks]; summaries=[]; Agent=RandomAgent if args.agent=='random' else GreedyAgent
 for i,r in enumerate(rows):
  t=r['task']; task=Task(t['width'],t['height'],tuple(map(tuple,t['walls'])),tuple(t['start']),tuple(t['key']),tuple(t['door']),tuple(t['target']),tuple(map(tuple,t['distractors'])),t.get('task_type','key_door_reach')); env=GridWorld(task,rng=random.Random(args.seed+i)); env.reset(); agent=Agent(args.seed+i); correct=invalid=0; tr=[]
  for step in range(max(100,(r.get('optimal_steps') or 20)*4)):
   target=(shortest_plan_from_state(task,env.state) or [None])[0]; a=agent.act(env); correct+=int(a==target); invalid+=int(a not in env.valid_actions()); _,_,done,info=env.step(a); tr.append({'action':a,'expert_action':target,'info':info})
   if done: break
  summaries.append({'success':env.state.done,'steps':env.state.steps,'optimal_steps':r.get('optimal_steps',0),'path_efficiency':(r.get('optimal_steps',0)/env.state.steps if env.state.done else 0),'invalid_action_rate':invalid/max(1,env.state.steps),'step_success':correct/max(1,len(tr)),'trajectory':tr})
 out=Path(args.out); out.mkdir(parents=True,exist_ok=True); ss={'agent':args.agent,'split':args.split,'tasks':len(summaries),'full_task_success':statistics.mean([int(r['success']) for r in summaries]),'action_accuracy':statistics.mean([r['step_success'] for r in summaries]),'path_efficiency_successes':statistics.mean([r['path_efficiency'] for r in summaries if r['success']] or [0]),'invalid_action_rate':statistics.mean([r['invalid_action_rate'] for r in summaries]),'mean_steps':statistics.mean([r['steps'] for r in summaries])}; (out/f'{args.agent}_{args.split}.json').write_text(json.dumps(ss,indent=2)); (out/f'{args.agent}_{args.split}_trajectories.jsonl').write_text('\n'.join(json.dumps(r) for r in summaries)+'\n'); print(json.dumps(ss,indent=2))
if __name__=='__main__': main()
