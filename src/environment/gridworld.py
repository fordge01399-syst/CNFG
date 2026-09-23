from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
import json, random
from typing import Dict, List, Tuple, Optional

ACTIONS = ['up','down','left','right','pickup','open','use','wait']
DIRS = {'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}
ACTION_TO_ID = {a:i for i,a in enumerate(ACTIONS)}
ID_TO_ACTION = {i:a for a,i in ACTION_TO_ID.items()}

@dataclass(frozen=True)
class Task:
    width:int; height:int; walls:Tuple[Tuple[int,int],...]; start:Tuple[int,int]
    key:Tuple[int,int]; door:Tuple[int,int]; target:Tuple[int,int]
    distractors:Tuple[Tuple[int,int],...]; task_type:str='key_door_reach'

@dataclass
class State:
    pos:Tuple[int,int]; has_key:bool=False; door_open:bool=False; done:bool=False; steps:int=0

class GridWorld:
    def __init__(self, task:Task, observation='full', fail_rate=0.0, rng=None):
        self.task=task; self.observation=observation; self.fail_rate=fail_rate; self.rng=rng or random.Random(0); self.state=None
    def reset(self):
        self.state=State(pos=self.task.start); return self.state
    def clone(self):
        e=GridWorld(self.task,self.observation,self.fail_rate,self.rng); e.state=State(**asdict(self.state)); return e
    def in_bounds(self,p): return 0<=p[0]<self.task.width and 0<=p[1]<self.task.height
    def blocked(self,p): return p in set(self.task.walls)
    def valid_actions(self):
        s=self.state; out=[]
        if s.pos==self.task.door and not s.door_open:
            return ['open'] if s.has_key else []
        for a,(dx,dy) in DIRS.items():
            q=(s.pos[0]+dx,s.pos[1]+dy)
            if self.in_bounds(q) and not self.blocked(q): out.append(a)
        if s.pos==self.task.key and not s.has_key: out.append('pickup')
        if s.pos==self.task.target and s.door_open: out.append('use')
        return out
    def step(self, action):
        if self.state.done: return self.state,0.0,True,{'valid':False,'reason':'done'}
        valid=action in self.valid_actions(); info={'valid':valid,'action':action}
        self.state.steps+=1
        if not valid:
            info['reason']='invalid_action'; return self.state,-0.2,False,info
        if self.rng.random()<self.fail_rate and action in DIRS:
            info['reason']='environment_action_failure'; return self.state,-0.15,False,info
        if action in DIRS:
            dx,dy=DIRS[action]; self.state.pos=(self.state.pos[0]+dx,self.state.pos[1]+dy)
        elif action=='pickup': self.state.has_key=True
        elif action=='open': self.state.door_open=True
        elif action=='use': self.state.done=True; return self.state,1.0,True,info
        reward=-0.01
        return self.state,reward,False,info
    def observe(self):
        s=self.state; visible={'position':s.pos,'has_key':s.has_key,'door_open':s.door_open,'target':self.task.target,'door':self.task.door}
        if self.observation=='partial':
            visible['walls']=[p for p in self.task.walls if abs(p[0]-s.pos[0])+abs(p[1]-s.pos[1])<=2]
            visible['key']=self.task.key if abs(self.task.key[0]-s.pos[0])+abs(self.task.key[1]-s.pos[1])<=3 else None
        else:
            visible['walls']=list(self.task.walls); visible['key']=self.task.key
        visible['distractors']=list(self.task.distractors); visible['valid_actions']=self.valid_actions()
        return visible
    def to_json(self): return {'task':asdict(self.task),'state':asdict(self.state),'observation':self.observe()}

def shortest_plan(task:Task, max_steps=200):
    start=(task.start,False,False); q=deque([(start,[])]) ; seen={start}; walls=set(task.walls)
    while q:
        (pos,key,opened), path=q.popleft()
        if pos==task.target and opened: return path+['use']
        if len(path)>=max_steps: continue
        candidates=[]
        if pos==task.door and not opened:
            if key: candidates.append(('open',(pos,key,True)))
        else:
            for a,(dx,dy) in DIRS.items():
                np=(pos[0]+dx,pos[1]+dy)
                if not (0<=np[0]<task.width and 0<=np[1]<task.height) or np in walls: continue
                candidates.append((a,(np,key,opened)))
            if pos==task.key and not key: candidates.append(('pickup',(pos,True,opened)))
        for a,ns in candidates:
            if ns not in seen: seen.add(ns); q.append((ns,path+[a]))
    return None

def make_task(rng, level='medium', task_id=0):
    size={'small':5,'medium':7,'large':9}.get(level,7); cells=[(x,y) for x in range(size) for y in range(size)]
    for _ in range(1000):
        start,key,door,target=rng.sample(cells,4); walls=set(rng.sample([p for p in cells if p not in {start,key,door,target}], int(size*size*0.10 if level!='small' else 1)))
        t=Task(size,size,tuple(sorted(walls)),start,key,door,target,tuple(rng.sample([p for p in cells if p not in walls and p not in {start,key,door,target}], min(4, max(1,size//2)))))
        if shortest_plan(t): return t
    raise RuntimeError('failed to generate solvable task')

def task_to_row(task, split, seed, task_id):
    plan=shortest_plan(task); return {'task_id':task_id,'split':split,'seed':seed,'task':asdict(task),'optimal_plan':plan,'optimal_steps':len(plan) if plan else None}
