"""
Train two groups of agents, one to attack the building, one to defense
"""

import argparse
import logging as log
import time

import magent
from magent import utility
from models import buffer
from models.mx_model import DeepQNetwork as RLModel

def load_config(size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": size, "map_height": size})
    cfg.set({"minimap_mode": True})

    agent = cfg.register_agent_type(
        name="agent",
        attr={
            "width": 1,
            "length": 1,
            "hp": 3,
            "speed": 3,
            "view_range": gw.CircleRange(7),
            "attack_range": gw.CircleRange(1),
            "damage": 6,
            "step_recover": 0,
            "step_reward": -0.01,
            "dead_penalty": -1,
            "attack_penalty": -0.1,
            "attack_in_group": 1,
        },
    )

    food = cfg.register_agent_type(
        name="food",
        attr={
            "width": 1,
            "length": 1,
            "hp": 25,
            "speed": 0,
            "view_range": gw.CircleRange(1),
            "attack_range": gw.CircleRange(0),
            "kill_reward": 5,
        },
    )

    g_f = cfg.add_group(food)
    g_a = cfg.add_group(agent)
    g_d = cfg.add_group(agent)

    a = gw.AgentSymbol(g_f, index='any')
    b = gw.AgentSymbol(g_a, index='any')
    c = gw.AgentSymbol(g_d, index='any')


    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.5)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=c, value=-0.5)

    return cfg