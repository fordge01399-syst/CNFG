from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer,AutoModel
from src.environment.gridworld import Task,GridWorld,ACTIONS,make_task
from scripts.qwen_transfer import text_of,ActionHead
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',default='final_independent_test'); ap.add_argument('--tasks',type=int,default=20); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--mask',action='store_true'); args=ap.parse_args(); device='cpu'; ck=torch.load(args.checkpoint,map_location='cpu'); tok=AutoTokenizer.from_pretrained(MODEL_ID); back=AutoModel.from_pretrained(MODEL_ID,torch_dtype=torch.float32).to(device).eval(); head=ActionHead(back.config.hidden_size); head.load_state_dict(ck['head']); head.eval(); rng=random.Random(args.seed+10000); succ=invalid=steps=0; per=[]
 for i in range(args.tasks):
  t=make_task(rng,'large',f'qwen-{args.split}-{i}'); env=GridWorld(t,'partial' if 'partial' in args.split else 'full',0.0,rng); env.reset(); done=False
  for k in range(140):
   row={'task':{'task_type':t.task_type},'observation':env.observe()}; text=text_of(row)
   z=tok([text],return_tensors='pt',padding=True,truncation=True,max_length=256)
   with torch.no_grad():
    h=back(**z).last_hidden_state; m=z['attention_mask'].unsqueeze(-1); x=(h*m).sum(1)/m.sum(1).clamp_min(1); logits=head(x)[0]
   valid=env.valid_actions(); raw=ACTIONS[int(logits.argmax())]; act=raw
   if args.mask and valid:
    act=max(valid,key=lambda a:float(logits[ACTIONS.index(a)]))
   _,_,done,info=env.step(act); invalid+=int(not info['valid']); steps+=1
   if done: succ+=1; break
  per.append({'task':i,'success':done,'steps':env.state.steps,'invalid_actions':sum(1 for _ in [])})
 out={'checkpoint':args.checkpoint,'split':args.split,'tasks':args.tasks,'mask':args.mask,'full_task_success':succ/args.tasks,'invalid_action_rate':invalid/max(1,steps),'mean_steps':steps/args.tasks,'trajectories':per}; p=Path('results/qwen_closed_loop'); p.mkdir(parents=True,exist_ok=True); (p/f'{Path(args.checkpoint).stem}_{args.split}_mask{args.mask}.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
