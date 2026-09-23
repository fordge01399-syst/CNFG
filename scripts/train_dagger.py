from __future__ import annotations
import sys,json,random,argparse,time
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cnfg.agent_model import CNFGAgent,FEATURE_DIM,encode_observation,action_mask
from src.environment.gridworld import *
from src.environment.planner import shortest_plan_from_state

def seed_all(s): random.seed(s); torch.manual_seed(s)
def make_model(dim): return CNFGAgent(dim=dim,memory_dim=dim)
def sample_tasks(rng,n,level='large'):
    out=[]
    for i in range(n): out.append(make_task(rng,level,i))
    return out
def collect(model,tasks,rng,device,model_prob,fail_rate=0.0,max_steps=140):
    episodes=[]; model.eval()
    for task in tasks:
        obs='partial' if rng.random()<0.25 else 'full'; env=GridWorld(task,obs,fail_rate,rng); env.reset(); mem=None; rows=[]
        for _ in range(max_steps):
            feat=encode_observation(env); plan=shortest_plan_from_state(task,env.state); target=plan[0] if plan else 'wait'; rows.append((feat,ACTIONS.index(target)))
            use_model=rng.random()<model_prob
            if use_model:
                x=torch.tensor([feat],dtype=torch.float32,device=device); mask=action_mask(env).unsqueeze(0).to(device)
                with torch.no_grad(): logits,mem=model(x,mem,mask); act=ACTIONS[int(logits.argmax(1).item())]
            else:
                act=target
                with torch.no_grad():
                    x=torch.tensor([feat],dtype=torch.float32,device=device); _,mem=model(x,mem,None)
            env.step(act)
            if env.state.done: break
        episodes.append(rows)
    return episodes
def train_epoch(model,episodes,opt,device):
    random.shuffle(episodes); total=0.; n=0
    for ep in episodes:
        mem=None; loss_terms=[]
        for feat,label in ep:
            x=torch.tensor([feat],dtype=torch.float32,device=device); y=torch.tensor([label],dtype=torch.long,device=device)
            logits,mem=model(x,mem,None); loss_terms.append(nn.functional.cross_entropy(logits,y))
        if loss_terms:
            loss=torch.stack(loss_terms).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=loss.item(); n+=1
    return total/max(1,n)
def eval_bc(model,episodes,device):
    model.eval(); c=t=0
    with torch.no_grad():
        for ep in episodes:
            mem=None
            for feat,label in ep:
                x=torch.tensor([feat],dtype=torch.float32,device=device); logits,mem=model(x,mem,None); c+=int(int(logits.argmax(1))==label); t+=1
    return c/max(1,t)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v4'); ap.add_argument('--out',default='checkpoints_dagger'); ap.add_argument('--init',default=''); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--episodes',type=int,default=512); ap.add_argument('--dim',type=int,default=96); ap.add_argument('--max-steps',type=int,default=140); args=ap.parse_args(); seed_all(args.seed); device='cpu'; rng=random.Random(args.seed+9000); model=make_model(args.dim)
    if args.init: model.load_state_dict(torch.load(args.init,map_location='cpu')['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4); best=(-1,None); hist=[]
    val_tasks=sample_tasks(random.Random(77000+args.seed),256,'large'); val_eps=collect(model,val_tasks,random.Random(88000+args.seed),device,0.0,max_steps=args.max_steps)
    for ep in range(1,args.epochs+1):
        p=min(0.85,0.20+0.07*ep); tasks=sample_tasks(rng,args.episodes,'large'); eps=collect(model,tasks,rng,device,p,0.03,args.max_steps); loss=train_epoch(model,eps,opt,device); val_eps=collect(model,val_tasks,random.Random(88000+args.seed),device,0.0,args.max_steps); va=eval_bc(model,val_eps,device); hist.append({'epoch':ep,'loss':loss,'model_action_probability':p,'validation_action_accuracy':va});
        if va>best[0]: best=(va,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'model_type':'cnfg','dim':args.dim,'seed':args.seed,'training':'dagger_long_horizon'},out/f'cnfg_seed{args.seed}.pt'); (out/f'cnfg_seed{args.seed}.json').write_text(json.dumps({'model':'cnfg','seed':args.seed,'parameters':sum(p.numel() for p in model.parameters()),'best_validation':best[0],'history':hist},indent=2))
if __name__=='__main__': main()
