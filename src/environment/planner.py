from collections import deque
from .gridworld import DIRS

def shortest_plan_from_state(task,state,max_steps=200):
    start=(state.pos,state.has_key,state.door_open); q=deque([(start,[])]); seen={start}; walls=set(task.walls)
    while q:
        (pos,key,opened),path=q.popleft()
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
    return []
