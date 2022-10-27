"""
Train battle, two models in two processes
"""

import argparse
import time
import logging as log
import math

import numpy as np

import magent
from magent import utility
from models import buffer
from model import ProcessingModel
# import calculate_pos as cp

def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {
            "width": 1,
            "length": 1,
            "hp": 10,
            "speed": 2,
            "view_range": gw.CircleRange(6),
            "attack_range": gw.CircleRange(1.5),
            "damage": 2,
            "step_recover": 0.1,
            "step_reward": -0.005,
            "kill_reward": 5,
            "dead_penalty": -0.1,
            "attack_penalty": -0.1,
        },
    )

    food = cfg.register_agent_type(
        "food",
        {
            "width": 1,
            "length": 1,
            "hp": 25,
            "speed": 0,
            "view_range": gw.CircleRange(1),
            "attack_range": gw.CircleRange(0),
            "kill_reward": 5,
        },
    )

    g0 = cfg.add_group(small)   #group_handle : int，handle的标号
    g1 = cfg.add_group(small)
    gf1 = cfg.add_group(food)
    gf2 = cfg.add_group(food)

    f1 = gw.AgentSymbol(gf1, index='any')
    f2 = gw.AgentSymbol(gf2, index='any')
    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.2)    #Event是gridworld中的EventNode类
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.2)

    cfg.add_reward_rule(gw.Event(b, 'attack', f1), receiver=b, value=1)
    cfg.add_reward_rule(gw.Event(b, 'attack', f1), receiver=a, value=-1)
    cfg.add_reward_rule(gw.Event(a, 'attack', f1), receiver=a, value=-50)

    cfg.add_reward_rule(gw.Event(b, 'attack', f2), receiver=b, value=1)
    cfg.add_reward_rule(gw.Event(b, 'attack', f2), receiver=a, value=-1)
    cfg.add_reward_rule(gw.Event(a, 'attack', f2), receiver=a, value=-50)

    return cfg

leftID = 0
rightID = 1
targetID1 = 2
targetID2 = 3

def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""

    # global leftID, rightID
    # leftID, rightID = rightID, leftID

    #target_1
    pos = []
    for x in range(5, 10):
        for y in range(5, 10):
            pos.append([x, y, 0])
    env.add_agents(handles[targetID1], method="custom", pos=pos)

    #target_2
    pos = []
    for x in range(5, 10):
        for y in range(40, 45):
            pos.append([x, y, 0])
    env.add_agents(handles[targetID2], method="custom", pos=pos)

    # left
    # n = 100
    pos = []
    for x in range(15, 25, 2):
        for y in range(5, 45, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    # n = 50
    pos = []
    for x in range(30, 40, 2):
        for y in range(5, 45, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)

def generate_map2(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""

    # global leftID, rightID
    # leftID, rightID = rightID, leftID

    #target_1
    pos = []
    for x in range(5, 10):
        for y in range(5, 10):
            pos.append([x, y, 0])
    env.add_agents(handles[targetID1], method="custom", pos=pos)

    #target_2
    pos = []
    for x in range(5, 10):
        for y in range(40, 45):
            pos.append([x, y, 0])
    env.add_agents(handles[targetID2], method="custom", pos=pos)

    # left
    # n = 100
    pos = []
    for x in range(15, 25, 2):
        for y in range(5, 45, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    # n = 50
    pos = []
    for x in range(30, 40, 2):
        for y in range(25, 45):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def play_a_round(k, env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    """play a ground and train"""
    env.reset()
    generate_map2(env, map_size, handles)
    # generate_right2(env, handles)

    step_ct = 0 #每次采样的最大轮数（帧数）
    done = False

    n = len(handles) - 2
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %.2f number %s " % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            # let models infer action in parallel (non-blocking)
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps, block=False)

        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            #包围加奖励reward加系数
            pos = env.get_pos(handles[i])
            if train:
                alives = env.get_alive(handles[i])
                # store samples in replay buffer (non-blocking)
                models[i].sample_step(rewards, alives, block=False)
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        # clear dead agents
        env.clear_dead()

        # check return message of previous called non-blocking function sample_step()
        if args.train:
            for model in models:
                model.check_done()

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s " %
                  (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2)))

        step_ct += 1
        if step_ct > 1000:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train models in parallel
        for i in range(n):
            models[i].train(print_every=100, block=False)
        for i in range(n):
            total_loss[i], value[i] = models[i].fetch_train()

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    def round_list(l): return [round(x, 2) for x in l]
    return round_list(total_loss), nums, round_list(total_reward), round_list(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=50)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="battle")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # set logger
    buffer.init_logger(args.name)

    # init the game
    env = magent.GridWorld(get_config(map_size=args.map_size))
    env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()
    #handles 是阵营，battle模式下有handle[LeftID],handles[RightID]

    # sample eval observation set
    eval_obs = [None, None]
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, handles)
        for i in range(len(handles)):
            eval_obs[i] = buffer.sample_observation(env, handles, 2048, 500)

    # load models
    batch_size = 1024
    unroll_step = 8
    target_update = 1200
    train_freq = 5

    if args.alg == 'dqn':
        from models.tf_model import DeepQNetwork
        RLModel = DeepQNetwork
        base_args = {'batch_size': batch_size,
                     'memory_size': 8 * 625, 'learning_rate': 1e-4,
                     'target_update': target_update, 'train_freq': train_freq}
    elif args.alg == 'drqn':
        from models.tf_model import DeepRecurrentQNetwork
        RLModel = DeepRecurrentQNetwork
        base_args = {'batch_size': batch_size / unroll_step, 'unroll_step': unroll_step,
                     'memory_size': 8 * 625, 'learning_rate': 1e-4,
                     'target_update': target_update, 'train_freq': train_freq}
    elif args.alg == 'a2c':
        # see train_against.py to know how to use a2c
        raise NotImplementedError

    # init models
    names = [args.name + "-l", args.name + "-r"]
    models = []

    for i in range(len(names)):
        model_args = {'eval_obs': eval_obs[i]}
        model_args.update(base_args)
        models.append(ProcessingModel(env, handles[i], names[i], 20000+i, 1000, RLModel, **model_args))

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print state info
    print(args)
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = buffer.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05]) if not args.greedy else 0

        loss, num, reward, value = play_a_round(k, env, args.map_size, handles, models,
                                                train=args.train, print_every=50,
                                                render=args.render or (k+1) % args.render_every == 0,
                                                eps=eps)  # for e-greedy

        log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)

    # send quit command
    for model in models:
        model.quit()

