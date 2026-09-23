from __future__ import annotations
import torch
from torch import nn
from src.environment.gridworld import ACTIONS, DIRS

FEATURE_DIM = 32

class CNFGAgent(nn.Module):
    """Structured closed-loop CNFG-Agent.

    Unary state/goal features and ordered pair-relative features are composed with
    a recurrent memory. The input contains observations only; expert plans are
    never encoded. Memory is intended to persist across rollout steps.
    """
    def __init__(self, grid_size=9, dim=96, memory_dim=96):
        super().__init__(); self.grid_size=grid_size; self.dim=dim
        self.unary=nn.Sequential(nn.Linear(FEATURE_DIM,dim),nn.LayerNorm(dim),nn.Tanh())
        self.pair=nn.Sequential(nn.Linear(6,dim),nn.LayerNorm(dim),nn.Tanh())
        self.relation_gate=nn.Linear(FEATURE_DIM,dim)
        self.update=nn.GRUCell(dim*3,memory_dim)
        self.action=nn.Sequential(nn.LayerNorm(memory_dim+dim),nn.Linear(memory_dim+dim,dim),nn.GELU(),nn.Dropout(0.05),nn.Linear(dim,len(ACTIONS)))
        self.goal_head=nn.Linear(memory_dim,3)
    def forward(self,x,memory=None,action_mask=None):
        z=self.unary(x); p=self.pair(x[:,8:14]); gate=torch.sigmoid(self.relation_gate(x)); p=p*gate
        if memory is None: memory=x.new_zeros((x.size(0),self.update.hidden_size))
        h=self.update(torch.cat([z,p,memory],-1),memory); logits=self.action(torch.cat([h,z],-1))
        if action_mask is not None: logits=logits.masked_fill(~action_mask.bool(),-1e9)
        return logits,h
    def act(self,x,memory=None,action_mask=None):
        with torch.no_grad(): logits,m=self.forward(x,memory,action_mask); return logits.argmax(-1),m

class MLPAgent(nn.Module):
    def __init__(self,dim=96):
        super().__init__(); self.net=nn.Sequential(nn.Linear(FEATURE_DIM,dim),nn.LayerNorm(dim),nn.GELU(),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,len(ACTIONS)))
    def forward(self,x,memory=None,action_mask=None):
        logits=self.net(x)
        if action_mask is not None: logits=logits.masked_fill(~action_mask.bool(),-1e9)
        return logits,memory

class GRUAgent(nn.Module):
    def __init__(self,dim=96):
        super().__init__(); self.inp=nn.Linear(FEATURE_DIM,dim); self.gru=nn.GRUCell(dim,dim); self.out=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,len(ACTIONS)))
    def forward(self,x,memory=None,action_mask=None):
        if memory is None: memory=x.new_zeros(x.size(0),self.gru.hidden_size)
        h=self.gru(torch.nn.functional.gelu(self.inp(x)),memory); logits=self.out(h)
        if action_mask is not None: logits=logits.masked_fill(~action_mask.bool(),-1e9)
        return logits,h

def encode_observation(env):
    s=env.state; t=env.task; W=max(1,t.width-1); H=max(1,t.height-1)
    def pos(p): return (p[0]/W,p[1]/H)
    cx,cy=pos(s.pos); tx,ty=pos(t.target); kx,ky=pos(t.key); dx,dy=pos(t.door)
    visible=env.observation!='partial'; ob=env.observe()
    key_obs=ob.get('key'); door_obs=ob.get('door')
    if key_obs is None: kx,ky=-1.,-1.
    else: kx,ky=pos(key_obs)
    if door_obs is None: dx,dy=-1.,-1.
    else: dx,dy=pos(door_obs)
    rel=[tx-cx,ty-cy,kx-cx,ky-cy,dx-cx,dy-cy]
    walls=[]
    for a in ['up','down','left','right']:
        dd=DIRS[a]; q=(s.pos[0]+dd[0],s.pos[1]+dd[1]); walls.append(float((not env.in_bounds(q)) or env.blocked(q)))
    avail=[float(a in env.valid_actions()) for a in ACTIONS]
    flags=[float(s.has_key),float(s.door_open),float(s.pos==t.key),float(s.pos==t.door),float(visible),float(env.observation=='partial')]
    return [cx,cy,tx,ty,kx,ky,dx,dy]+rel+walls+flags+avail

def action_mask(env):
    mask=torch.zeros(len(ACTIONS),dtype=torch.bool)
    for a in env.valid_actions(): mask[ACTIONS.index(a)]=True
    return mask
