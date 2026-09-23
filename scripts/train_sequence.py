from __future__ import annotations
import sys,json,random,argparse,time
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cnfg.agent_model import CNFGAgent,MLPAgent,GRUAgent,FEATURE_DIM
from src.environment.gridworld import ACTIONS

def seed_all(s): random.seed(s); torch.manual_seed(s)
def load_episodes(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]; groups={}
    for r in rows: groups.setdefault(r['task_id'],[]).append(r)
    return [sorted(v,key=lambda r:r['step']) for v in groups.values()]
def make_model(kind,dim): return {'cnfg':CNFGAgent(dim=dim,memory_dim=dim),'gru':GRUAgent(dim),'mlp':MLPAgent(dim)}[kind]
def batch_loss(model,eps,device):
    B=len(eps); T=max(len(e) for e in eps); x=torch.zeros(B,T,FEATURE_DIM,device=device); y=torch.full((B,T),-100,dtype=torch.long,device=device); mask=torch.zeros(B,T,dtype=torch.bool,device=device)
    for i,e in enumerate(eps):
        for t,r in enumerate(e): x[i,t]=torch.tensor(r['features']); y[i,t]=ACTIONS.index(r['expert_action']); mask[i,t]=True
    mem=None; logits=[]
    for t in range(T):
        l,mem=model(x[:,t],mem); logits.append(l)
    return nn.functional.cross_entropy(torch.stack(logits,1)[mask],y[mask])
def evaluate(model,eps,device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for start in range(0,len(eps),128):
            batch=eps[start:start+128]; B=len(batch); T=max(len(e) for e in batch); x=torch.zeros(B,T,FEATURE_DIM,device=device); y=torch.full((B,T),-100,dtype=torch.long,device=device); mask=torch.zeros(B,T,dtype=torch.bool,device=device)
            for i,e in enumerate(batch):
                for t,r in enumerate(e): x[i,t]=torch.tensor(r['features']); y[i,t]=ACTIONS.index(r['expert_action']); mask[i,t]=True
            mem=None; out=[]
            for t in range(T): l,mem=model(x[:,t],mem); out.append(l)
            pred=torch.stack(out,1).argmax(-1); correct+=(pred[mask]==y[mask]).sum().item(); total+=mask.sum().item()
    return correct/max(1,total)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v3'); ap.add_argument('--out',default='checkpoints_v3'); ap.add_argument('--model',choices=['cnfg','gru','mlp'],default='cnfg'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--batch-episodes',type=int,default=128); ap.add_argument('--dim',type=int,default=96); args=ap.parse_args(); seed_all(args.seed); device='cpu'; root=Path(args.data); train=load_episodes(root/'train.jsonl'); valid=load_episodes(root/'validation.jsonl'); model=make_model(args.model,args.dim); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=(-1,None); hist=[]; start=time.time()
    for ep in range(1,args.epochs+1):
        random.shuffle(train); model.train(); total=0.; n=0
        for j in range(0,len(train),args.batch_episodes):
            loss=batch_loss(model,train[j:j+args.batch_episodes],device); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=loss.item(); n+=1
        va=evaluate(model,valid,device); hist.append({'epoch':ep,'train_loss':total/max(1,n),'validation_action_accuracy':va})
        if va>best[0]: best=(va,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'model_type':args.model,'dim':args.dim,'seed':args.seed},out/f'{args.model}_seed{args.seed}.pt'); (out/f'{args.model}_seed{args.seed}.json').write_text(json.dumps({'model':args.model,'seed':args.seed,'parameters':sum(p.numel() for p in model.parameters()),'best_validation':best[0],'history':hist,'runtime_seconds':time.time()-start},indent=2))
if __name__=='__main__': main()
