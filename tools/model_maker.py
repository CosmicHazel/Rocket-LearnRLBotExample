import torch
import os

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, ReLU
from rocket_learn.utils.util import SplitLayer


# TODO add your network here
def get_actor(split, state_dim):
    return DiscretePolicy(Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, state_dim),
        SplitLayer(splits=split)
    ), split)


# TODO set your split and obs length
split = (3, 3, 3, 3, 3, 2, 2, 2)

# TOTAL SIZE OF THE INPUT DATA
state_dim = 107

actor = get_actor(split, state_dim)

# PPO REQUIRES AN ACTOR/CRITIC AGENT

cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, "checkpoint.pt"))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
torch.jit.save(torch.jit.script(actor), 'jit.pt')

exit(0)
