import sys, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.environment.gridworld import *
from src.cnfg.agent_model import CNFGAgent,encode_observation,action_mask

def test_task_and_plan():
 t=make_task(random.Random(4),'small'); p=shortest_plan(t); assert p and p[-1]=='use'
def test_transition_validity():
 t=make_task(random.Random(5),'small'); e=GridWorld(t); e.reset(); assert all(a in ACTIONS for a in e.valid_actions()); _,_,_,info=e.step('wait'); assert not info['valid']
def test_encoding_and_forward_memory():
 t=make_task(random.Random(6),'medium'); e=GridWorld(t); e.reset(); x=torch.tensor([encode_observation(e)],dtype=torch.float32); m=CNFGAgent(); logits,h=m(x,None,action_mask(e).unsqueeze(0)); assert logits.shape==(1,len(ACTIONS)); assert h.shape[0]==1
def test_rollout_interface():
 t=make_task(random.Random(7),'small'); e=GridWorld(t); e.reset(); assert isinstance(e.observe(),dict); assert isinstance(e.valid_actions(),list)
