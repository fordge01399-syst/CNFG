from __future__ import annotations
import random
from src.environment.gridworld import ACTIONS,DIRS
class RandomAgent:
    def __init__(self,seed=0): self.rng=random.Random(seed)
    def act(self,env):
        valid=env.valid_actions(); return self.rng.choice(valid) if valid else 'wait'
class GreedyAgent:
    def __init__(self,seed=0): pass
    def act(self,env):
        s=env.state;t=env.task
        if s.pos==t.key and not s.has_key: return 'pickup'
        if s.pos==t.door and s.has_key and not s.door_open: return 'open'
        target=t.key if not s.has_key else (t.door if not s.door_open else t.target)
        valid=env.valid_actions(); moves=[a for a in valid if a in DIRS]
        if not moves: return valid[0] if valid else 'wait'
        def score(a):
            dx,dy=DIRS[a]; q=(s.pos[0]+dx,s.pos[1]+dy); return abs(q[0]-target[0])+abs(q[1]-target[1])
        return min(moves,key=score)
