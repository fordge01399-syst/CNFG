from __future__ import annotations
import sys,json,argparse,random,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cnfg.agent_model import CNFGAgent,MLPAgent,GRUAgent,encode_observation,action_mask
from src.environment.gridworld import *
from src.environment.planner import shortest_plan_from_state

def load_agent(ck,model_type,dim):
    m={'cnfg':CNFGAgent(dim=dim,memory_dim=dim),'mlp':MLPAgent(dim),'gru':GRUAgent(dim)}[model_type]; m.load_state_dict(torch.load(ck,map_location='cpu')['model']); m.eval(); return m

def run_one(model,task,obs='full',fail_rate=0.,max_steps=100,seed=0,use_mask=True):
    env=GridWorld(task,obs,fail_rate,random.Random(seed)); env.reset(); memory=None; traj=[]; correct=0; invalid=0
    expert=shortest_plan(task) or []
    for t in range(max_steps):
        x=torch.tensor([encode_observation(env)],dtype=torch.float32); mask=action_mask(env).unsqueeze(0)
        with torch.no_grad(): logits,memory=model(x,memory,mask if use_mask else None); pred=int(logits.argmax(1).item())
        a=ACTIONS[pred]; local_plan=shortest_plan_from_state(env.task,env.state); target=local_plan[0] if local_plan else None; correct+=int(a==target);
        if a not in env.valid_actions(): invalid+=1
        old=env.to_json(); ns,r,done,info=env.step(a); traj.append({'t':t,'state':old,'action':a,'expert_action':target,'info':info,'new_state':env.to_json()})
        if done: break
    success=env.state.done; actual=env.state.steps; optimal=len(expert)
    return {'success':success,'steps':actual,'optimal_steps':optimal,'path_efficiency':(optimal/actual if success and actual else 0.),'invalid_action_rate':invalid/max(1,actual),'step_success':correct/max(1,actual),'trajectory':traj}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--model',required=True,choices=['cnfg','mlp','gru']); ap.add_argument('--data',default='data'); ap.add_argument('--out',default='results'); ap.add_argument('--dim',type=int,default=64); ap.add_argument('--split',default='iid_test'); ap.add_argument('--fail-rate',type=float,default=0.0); ap.add_argument('--max-tasks',type=int,default=1000); ap.add_argument('--no-action-mask',action='store_true'); args=ap.parse_args()
    rows=[json.loads(x) for x in (Path(args.data)/f'{args.split}.jsonl').read_text().splitlines() if x.strip()]; rows=rows[:args.max_tasks]; model=load_agent(args.checkpoint,args.model,args.dim); results=[]
    for i,r in enumerate(rows):
        t=r['task']; task=Task(t['width'],t['height'],tuple(map(tuple,t['walls'])),tuple(t['start']),tuple(t['key']),tuple(t['door']),tuple(t['target']),tuple(map(tuple,t['distractors'])),t.get('task_type','key_door_reach')); results.append(run_one(model,task,'partial' if args.split=='partial_observation_test' else 'full',args.fail_rate, max(100,(r.get('optimal_steps') or 20)*4),i,not args.no_action_mask))
    success=[int(r['success']) for r in results]; vals=lambda k:[r[k] for r in results]
    summary={'model':args.model,'split':args.split,'tasks':len(results),'full_task_success':sum(success)/len(success),'action_accuracy':statistics.mean(vals('step_success')),'path_efficiency_successes':statistics.mean([r['path_efficiency'] for r in results if r['success']] or [0]),'invalid_action_rate':statistics.mean(vals('invalid_action_rate')),'mean_steps':statistics.mean(vals('steps')),'failures':[r for r in results if not r['success']][:20]}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/f'{args.model}_{args.split}.json').write_text(json.dumps(summary,indent=2)); (out/f'{args.model}_{args.split}_trajectories.jsonl').write_text('\n'.join(json.dumps(r) for r in results)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
