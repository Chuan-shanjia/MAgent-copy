""" battle of two armies """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(12), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1, 'kill_supply': 2,

         'step_reward': -0.005,  'kill_reward': 10, 'dead_penalty': -0.1, 'attack_penalty': -0.1,
         })

    #small是智能体的属性，可以改

    g0 = cfg.add_group(small)   #group_handle : int，handle的标号
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.5)    #Event是gridworld中的EventNode类
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.5)

    return cfg
