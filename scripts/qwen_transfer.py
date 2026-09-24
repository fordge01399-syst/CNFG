from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel
ACTIONS=['up','down','left','right','open','pickup','drop','use','wait']
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'

def load_rows(path,n):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip(): rows.append(json.loads(line))
        if len(rows)>=n: break
    return rows

def text_of(r):
    o=r['observation']; t=r['task']
    return ('state position=%s has_key=%s door_open=%s target=%s door=%s visible_key=%s visible_door=%s walls=%s distractors=%s valid_actions=%s goal=reach_target_after_key_door task_type=%s' % (o.get('position'),o.get('has_key'),o.get('door_open'),o.get('target'),o.get('door'),o.get('key'),o.get('door'),o.get('walls'),o.get('distractors'),o.get('valid_actions'),t.get('task_type')))

def tokenize(tok,texts,device): return tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=256).to(device)
def pooled(backbone,z):
    h=backbone(input_ids=z['input_ids'],attention_mask=z['attention_mask']).last_hidden_state; m=z['attention_mask'].unsqueeze(-1); return (h*m).sum(1)/m.sum(1).clamp_min(1)
class ActionHead(nn.Module):
    def __init__(self,h): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,256),nn.GELU(),nn.Linear(256,len(ACTIONS)))
    def forward(self,x): return self.net(x)
class TransferPolicy(nn.Module):
    def __init__(self,h): super().__init__(); self.transfer=nn.Sequential(nn.Linear(h,256),nn.GELU(),nn.LayerNorm(256)); self.memory=nn.GRUCell(256,256); self.head=nn.Linear(256,len(ACTIONS))
    def forward(self,x,mem=None):
        z=self.transfer(x); mem=self.memory(z,mem if mem is not None else torch.zeros(x.size(0),256,device=x.device)); return self.head(mem),mem

def train_head(X,y,kind,epochs,seed):
    torch.manual_seed(seed); model=TransferPolicy(X.shape[1]) if kind=='transfer' else ActionHead(X.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); best=(-1,None)
    for _ in range(epochs):
        model.train(); p=torch.randperm(len(X))
        for j in range(0,len(X),64):
            ix=p[j:j+64]; logits,_=model(X[ix]) if kind=='transfer' else (model(X[ix]),None); loss=nn.functional.cross_entropy(logits,y[ix]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad(): logits,_=model(X) if kind=='transfer' else (model(X),None); acc=(logits.argmax(1)==y).float().mean().item()
        if acc>best[0]: best=(acc,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); return model,best[0]

def evaluate_head(model,X,y,kind):
    model.eval();
    with torch.no_grad(): logits,_=model(X) if kind=='transfer' else (model(X),None)
    return (logits.argmax(1)==y).float().mean().item()

def train_qwen(backbone,tok,train_text,train_y,valid_text,valid_y,mode,epochs,seed,device):
    torch.manual_seed(seed); head=ActionHead(backbone.config.hidden_size); params=list(head.parameters())
    for p in backbone.parameters(): p.requires_grad=False
    if mode=='finetuned':
        for p in backbone.layers[-1].parameters(): p.requires_grad=True
        params += list(backbone.layers[-1].parameters())
    opt=torch.optim.AdamW(params,lr=2e-5 if mode=='finetuned' else 2e-3,weight_decay=1e-4); best=(-1,None)
    for _ in range(epochs):
        backbone.train() if mode=='finetuned' else backbone.eval(); head.train()
        order=list(range(len(train_text))); random.Random(seed+_).shuffle(order)
        for j in range(0,len(order),8):
            ix=order[j:j+8]; z=tokenize(tok,[train_text[i] for i in ix],device); x=pooled(backbone,z); logits=head(x); y=train_y[ix].to(device); loss=nn.functional.cross_entropy(logits,y); opt.zero_grad(); loss.backward(); opt.step()
        backbone.eval(); head.eval();
        with torch.no_grad(): x=pooled(backbone,tokenize(tok,valid_text,device)); va=(head(x).argmax(1)==valid_y.to(device)).float().mean().item()
        if va>best[0]: best=(va,{'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'last':{k:v.detach().cpu().clone() for k,v in backbone.layers[-1].state_dict().items()} if mode=='finetuned' else None})
    head.load_state_dict(best[1]['head']);
    if mode=='finetuned': backbone.layers[-1].load_state_dict(best[1]['last'])
    return head,best[0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data_v4'); ap.add_argument('--out',default='experiments/pretrained_backbone'); ap.add_argument('--train-rows',type=int,default=512); ap.add_argument('--valid-rows',type=int,default=128); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); random.seed(args.seed); torch.set_num_threads(8); device='cpu'; start=time.time(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID); backbone=AutoModel.from_pretrained(MODEL_ID,torch_dtype=torch.float32).to(device); hidden=backbone.config.hidden_size
    train=load_rows(Path(args.data)/'train.jsonl',args.train_rows); valid=load_rows(Path(args.data)/'validation.jsonl',args.valid_rows); tt=[text_of(r) for r in train]; vt=[text_of(r) for r in valid]; y=torch.tensor([ACTIONS.index(r['expert_action']) for r in train]); vy=torch.tensor([ACTIONS.index(r['expert_action']) for r in valid])
    results={'model_id':MODEL_ID,'seed':args.seed,'hidden_size':hidden,'train_rows':len(train),'validation_rows':len(valid),'training_seconds':0.0,'no_solution_leakage':'text excludes expert_action, optimal_steps, and action sequences'}
    for mode in ['frozen','finetuned']:
        head,va=train_qwen(backbone,tok,tt,y,vt,vy,mode,args.epochs,args.seed,device); p=out/f'{mode}_qwen_seed{args.seed}.pt'; torch.save({'backbone_id':MODEL_ID,'head':head.state_dict(),'mode':mode,'parameters':sum(x.numel() for x in head.parameters())+(sum(x.numel() for x in backbone.layers[-1].parameters()) if mode=='finetuned' else 0)},p); results[f'{mode}_qwen']={'checkpoint':str(p),'validation_action_accuracy':va}
    backbone.eval();
    with torch.no_grad(): X=pooled(backbone,tokenize(tok,tt,device)).cpu(); VX=pooled(backbone,tokenize(tok,vt,device)).cpu()
    tm,ta=train_head(X,y,'transfer',args.epochs,args.seed); p=out/f'transfer_seed{args.seed}.pt'; torch.save({'backbone_id':MODEL_ID,'model':tm.state_dict(),'mode':'cnfg_qwen_transfer','parameters':sum(x.numel() for x in tm.parameters())},p); results['cnfg_qwen_transfer']={'checkpoint':str(p),'train_action_accuracy':ta,'validation_action_accuracy':evaluate_head(tm,VX,vy,'transfer')}
    results['training_seconds']=time.time()-start; (out/f'summary_seed{args.seed}.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
