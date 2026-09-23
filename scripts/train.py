from __future__ import annotations
import sys,json,random,argparse,time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cnfg.agent_model import CNFGAgent,MLPAgent,GRUAgent
from src.environment.gridworld import ACTIONS

def seed_all(s): random.seed(s); torch.manual_seed(s)
def load_rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def tensors(rows): return torch.tensor([r['features'] for r in rows],dtype=torch.float32),torch.tensor([ACTIONS.index(r['expert_action']) for r in rows],dtype=torch.long)
def evaluate(model,loader):
    model.eval(); c=n=0; loss=0.; ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            logits,_=model(x); loss+=ce(logits,y).item()*len(y); c+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    return {'accuracy':c/n,'loss':loss/n}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); ap.add_argument('--out',default='checkpoints'); ap.add_argument('--model',choices=['cnfg','mlp','gru'],default='cnfg'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch',type=int,default=512); ap.add_argument('--dim',type=int,default=64); args=ap.parse_args(); seed_all(args.seed)
    root=Path(args.data); train=tensors(load_rows(root/'train.jsonl')); valid=tensors(load_rows(root/'validation.jsonl'))
    tr=DataLoader(TensorDataset(*train),args.batch,shuffle=True); va=DataLoader(TensorDataset(*valid),args.batch)
    model={'cnfg':CNFGAgent(dim=args.dim,memory_dim=args.dim),'mlp':MLPAgent(args.dim),'gru':GRUAgent(args.dim)}[args.model]
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); best=(-1,None); hist=[]; start=time.time()
    for ep in range(1,args.epochs+1):
        model.train(); total=0.; n=0
        for x,y in tr:
            opt.zero_grad(); logits,_=model(x); l=ce(logits,y); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=l.item()*len(y); n+=len(y)
        vm=evaluate(model,va); hist.append({'epoch':ep,'train_loss':total/n,'validation':vm})
        if vm['accuracy']>best[0]: best=(vm['accuracy'],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); ck=out/f'{args.model}_seed{args.seed}.pt'; torch.save({'model':model.state_dict(),'model_type':args.model,'dim':args.dim,'seed':args.seed},ck)
    (out/f'{args.model}_seed{args.seed}.json').write_text(json.dumps({'model':args.model,'seed':args.seed,'parameters':sum(p.numel() for p in model.parameters()),'best_validation':best[0],'history':hist,'runtime_seconds':time.time()-start},indent=2))
if __name__=='__main__': main()
