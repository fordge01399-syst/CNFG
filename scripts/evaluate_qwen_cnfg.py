from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer,AutoModel
from src.environment.gridworld import GridWorld,make_task,ACTIONS
from src.cnfg.agent_model import encode_observation,action_mask
from scripts.qwen_transfer import text_of
from scripts.train_qwen_cnfg_dagger import QwenCNFG, qembed, planner_action, MODEL_ID
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--split',default='long_horizon_test');ap.add_argument('--tasks',type=int,default=5);ap.add_argument('--max-steps',type=int,default=40);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--mask',action='store_true');args=ap.parse_args();ck=torch.load(args.checkpoint,map_location='cpu');back=AutoModel.from_pretrained(MODEL_ID,torch_dtype=torch.float32).eval();tok=AutoTokenizer.from_pretrained(MODEL_ID);model=QwenCNFG();model.load_state_dict(ck['model']);model.eval();rng=random.Random(args.seed+8000);success=invalid=total=0;traj=[]
 for i in range(args.tasks):
  t=make_task(rng,'large',i);env=GridWorld(t,'partial' if 'partial' in args.split else 'full',0.0,rng);env.reset();mem=None;done=False;bad=0
  for step in range(args.max_steps):
   row={'task':{'task_type':t.task_type},'observation':env.observe()};q=qembed(back,tok,[text_of(row)]);x=torch.tensor([encode_observation(env)],dtype=torch.float32);am=action_mask(env).unsqueeze(0)
   logits,mem=model(q,x,mem,am if args.mask else None);raw=ACTIONS[int(logits.argmax())];act=raw
   if args.mask and env.valid_actions():act=max(env.valid_actions(),key=lambda a:float(logits[0,ACTIONS.index(a)]))
   _,_,done,info=env.step(act);bad+=int(not info['valid']);invalid+=int(not info['valid']);total+=1
   if done:success+=1;break
  traj.append({'task':i,'success':done,'steps':env.state.steps,'invalid':bad})
 out={'checkpoint':args.checkpoint,'split':args.split,'tasks':args.tasks,'max_steps':args.max_steps,'mask':args.mask,'full_task_success':success/args.tasks,'invalid_action_rate':invalid/max(1,total),'mean_steps':total/args.tasks,'trajectories':traj};p=Path('results/qwen_cnfg_dagger');p.mkdir(parents=True,exist_ok=True);(p/f'{Path(args.checkpoint).stem}_{args.split}_mask{args.mask}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
