from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel
from src.environment.gridworld import GridWorld,make_task,shortest_plan,ACTIONS
from src.cnfg.agent_model import encode_observation,action_mask,FEATURE_DIM
from scripts.qwen_transfer import text_of
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
class QwenCNFG(nn.Module):
 def __init__(self,qdim=896,dim=96,mem=96):
  super().__init__(); self.qproj=nn.Sequential(nn.Linear(qdim,dim),nn.LayerNorm(dim),nn.GELU()); self.unary=nn.Sequential(nn.Linear(FEATURE_DIM,dim),nn.LayerNorm(dim),nn.Tanh()); self.pair=nn.Sequential(nn.Linear(6,dim),nn.LayerNorm(dim),nn.Tanh()); self.gate=nn.Linear(FEATURE_DIM,dim); self.gru=nn.GRUCell(dim*3,mem); self.out=nn.Sequential(nn.LayerNorm(mem+dim),nn.Linear(mem+dim,dim),nn.GELU(),nn.Linear(dim,len(ACTIONS)))
 def forward(self,q,x,m=None,mask=None):
  a=self.qproj(q); z=self.unary(x); p=self.pair(x[:,8:14])*torch.sigmoid(self.gate(x));
  if m is None:m=x.new_zeros((x.size(0),self.gru.hidden_size))
  h=self.gru(torch.cat([a+z,p,m],-1),m); logits=self.out(torch.cat([h,z],-1));
  if mask is not None: logits=logits.masked_fill(~mask.bool(),-1e9)
  return logits,h

@torch.no_grad()
def qembed(back,tok,texts):
 z=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=128); h=back(**z).last_hidden_state; m=z['attention_mask'].unsqueeze(-1); return ((h*m).sum(1)/m.sum(1).clamp_min(1)).detach().clone()
def planner_action(env):
 p=shortest_plan(env.task)
 if not p:return 'wait'
 # derive correct action from actual state by BFS from current state
 from collections import deque
 s=env.state; start=(s.pos,s.has_key,s.door_open); q=deque([(start,[])]); seen={start}
 while q:
  (pos,key,opened),path=q.popleft()
  if pos==env.task.target and opened:return path[0] if path else 'use'
  if len(path)>200:continue
  cand=[]
  if pos==env.task.door and not opened:
   if key:cand=[('open',(pos,key,True))]
  else:
   for a,(dx,dy) in {'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}.items():
    np=(pos[0]+dx,pos[1]+dy)
    if 0<=np[0]<env.task.width and 0<=np[1]<env.task.height and np not in set(env.task.walls):cand.append((a,(np,key,opened)))
   if pos==env.task.key and not key:cand.append(('pickup',(pos,True,opened)))
  for a,ns in cand:
   if ns not in seen:seen.add(ns);q.append((ns,path+[a]))
 return 'wait'
def collect(model,back,tok,rng,episodes,device,eps,max_steps):
 rows=[]
 for i in range(episodes):
  t=make_task(rng,'large',i); env=GridWorld(t,'partial',0.03,rng); env.reset(); mem=None
  for step in range(max_steps):
   obs=env.observe(); dummy={'task':{'task_type':t.task_type},'observation':obs}; q=qembed(back,tok,[text_of(dummy)]); x=torch.tensor([encode_observation(env)],dtype=torch.float32); am=action_mask(env).unsqueeze(0); logits,mem=model(q,x,mem,am); pred=ACTIONS[int(logits.argmax())]; label=planner_action(env); chosen=label if rng.random()<eps else pred; rows.append((q.squeeze(0),x.squeeze(0),ACTIONS.index(label))); _,_,done,_=env.step(chosen)
   if done:break
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='experiments/qwen_cnfg_dagger');ap.add_argument('--max-steps',type=int,default=40);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--rounds',type=int,default=2);ap.add_argument('--episodes',type=int,default=16);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--init',default='');args=ap.parse_args(); random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(8);device='cpu'; out=Path(args.out);out.mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(MODEL_ID);back=AutoModel.from_pretrained(MODEL_ID,torch_dtype=torch.float32).eval();model=QwenCNFG();
 if args.init and Path(args.init).exists():
  ck=torch.load(args.init,map_location='cpu');
  if 'model' in ck:
   own=model.state_dict(); model.load_state_dict({k:v for k,v in ck['model'].items() if k in own and own[k].shape==v.shape},strict=False)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); rng=random.Random(args.seed+100);allrows=[];best=-1; start=time.time()
 for r in range(args.rounds):
  allrows += collect(model,back,tok,rng,args.episodes,device,eps=max(0.2,0.8-0.3*r),max_steps=args.max_steps); random.shuffle(allrows)
  for _ in range(args.epochs):
   for j in range(0,len(allrows),16):
    b=allrows[j:j+16]; q=torch.stack([z[0] for z in b]);x=torch.stack([z[1] for z in b]);y=torch.tensor([z[2] for z in b]);logits,_=model(q,x);loss=nn.functional.cross_entropy(logits,y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  torch.save({'model':model.state_dict(),'seed':args.seed,'round':r,'rows':len(allrows),'parameters':sum(p.numel() for p in model.parameters())},out/f'checkpoint_seed{args.seed}_round{r}.pt')
 summary={'seed':args.seed,'rounds':args.rounds,'episodes_per_round':args.episodes,'rows':len(allrows),'training_seconds':time.time()-start,'model_parameters':sum(p.numel() for p in model.parameters()),'model_id':MODEL_ID};(out/f'summary_seed{args.seed}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
