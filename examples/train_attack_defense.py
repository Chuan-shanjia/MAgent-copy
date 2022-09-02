"""
Train two groups of agents, one to attack the building, one to defense
"""

import argparse
import logging as log
import time

import magent
from magent import utility
from models import buffer
from model import ProcessingModel

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
    cfg.add_reward_rule(gw.Event(b, 'attack', c), receiver=b, value=0.2)
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=0.2)

    return cfg

def generate_map(env, map_size, food_handle, handles):
    center_x, center_y = map_size // 3, map_size // 3
    gap = 3
    side = 40

    def add_square(pos, side, gap):
        side = int(side)
        for x in range(center_x - side//2, center_x + side//2 + 1, gap):
            pos.append([x, center_y - side//2])
            pos.append([x, center_y + side//2])
        for y in range(center_y - side//2, center_y + side//2 + 1, gap):
            pos.append([center_x - side//2, y])
            pos.append([center_x + side//2, y])

    # agent_attack
    pos = []
    for x in range(map_size // 2 + gap, map_size // 2 + gap + side, 4):
        for y in range((map_size - side) // 2, (map_size - side) // 2 + side, 4):
            pos.append([x, y, 0])
    env.add_agents(handles[1], method="custom", pos=pos)

    # agent_defense
    pos = []
    add_square(pos, map_size * 0.2, 2)
    add_square(pos, map_size * 0.15, 2)
    env.add_agents(handles[0], method="custom", pos=pos)

    # food
    pos = []
    add_square(pos, map_size * 0.1, 1)
    add_square(pos, map_size * 0.1 - 2, 1)
    add_square(pos, map_size * 0.1 - 4, 1)
    add_square(pos, map_size * 0.1 - 6, 1)
    env.add_agents(food_handle, method="custom", pos=pos)

    # legend

    # legend = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #     [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
    #     [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
    #     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
    #     [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
    #     [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
    #     [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    # ]
    #
    # org = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    # ]
    #
    # def draw(base_x, base_y, scale, data):
    #     w, h = len(data), len(data[0])
    #     pos = []
    #     for i in range(w):
    #         for j in range(h):
    #             if data[i][j] == 1:
    #                 start_x = i * scale + base_x
    #                 start_y = j * scale + base_y
    #                 for x in range(start_x, start_x + scale):
    #                     for y in range(start_y, start_y + scale):
    #                         pos.append([y, x])
    #
    #     env.add_agents(food_handle, method="custom", pos=pos)
    #
    # scale = 1
    # w, h = len(legend), len(legend[0])
    # offset = -3
    # draw(offset + map_size // 2 - w // 2 * scale, map_size // 2 - h // 2 * scale, scale, legend)
    # draw(offset + map_size // 2 - w // 2 * scale + len(legend), map_size // 2 - h // 2 * scale, scale, org)


def play_a_round(env, map_size, food_handle, handles, models, train=0,
                 print_every=10, record=False, render=False, eps=None):
    env.reset()
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    total_reward = 0
    done = False

    pos_reward_ct = set()

    n = len(handles)
    obs  = [None for _ in range(n)]
    ids  = [None for _ in range(n)]
    acts = [None for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps, block=False)
            # acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            # env.set_action(handles[i], acts[i])

        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            # 包围加奖励reward加系数
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
        food_num = env.get_num(food_handle)

        # clear dead agents
        env.clear_dead()

        # check return message of previous called non-blocking function sample_step()
        if args.train:
            for model in models:
                model.check_done()


        if step_ct % print_every == 0:
            print("step %3d,  num %s,  reward %.2f,  total_reward: %.2f, non_zero: %d" %
                  (step_ct, [food_num,nums], np.around(step_reward, 2), np.around(total_reward, 2), len(pos_reward_ct)))
        step_ct += 1

        if step_ct > 1000:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    if record:
        with open("reward-hunger.txt", "a") as fout:
            fout.write(str(nums[0]) + "\n")

    # train
    total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train models in parallel
        for i in range(n):
            models[i].train(print_every=1000, block=False)
        for i in range(n):
            total_loss[i], value[i] = models[i].fetch_train()

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    def round_list(l): return [round(x, 2) for x in l]
    return round_list(total_loss), nums, round_list(total_reward), round_list(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=1500)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--map_size", type=int, default=200)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="gather")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init env
    env = magent.GridWorld(load_config(size=args.map_size))
    env.set_render_dir("build/render")

    handles = env.get_handles()
    food_handle = handles[0]
    player_handles = handles[1:]

    # sample eval observation set
    eval_obs = [None, None]
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, food_handle, player_handles)
        for i in range(len(player_handles)):
            eval_obs = buffer.sample_observation(env, player_handles[i], 2048, 500)

    # load models
    batch_size = 1024
    unroll_step = 8
    target_update = 1200
    train_freq = 5

    if args.alg == 'dqn':
        from models.tf_model import DeepQNetwork

        RLModel = DeepQNetwork
        base_args = {'batch_size': batch_size,
                     'memory_size': 16 * 625, 'learning_rate': 1e-4,
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
        models.append(ProcessingModel(env, handles[i], names[i], 20000 + i, 1000, RLModel, **model_args))

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print('view_space', env.get_view_space(player_handles[0]))
    print('feature_space', env.get_feature_space(player_handles[0]))
    print('view2attack', env.get_view2attack(player_handles[0]))

    if args.record:
        for k in range(4, 999 + 5, 5):
            eps = 0
            for model in models:
                model.load(save_dir, start_from)
                play_a_round(env, args.map_size, food_handle, player_handles, models,
                             -1, record=True, render=False,
                             print_every=args.print_every, eps=eps)
    else:
        # play
        start = time.time()
        for k in range(start_from, start_from + args.n_round):
            tic = time.time()
            eps = buffer.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05]) if not args.greedy else 0
            loss, reward, value, pos_reward_ct = \
                play_a_round(env, args.map_size, food_handle, player_handles, models,
                             train=args.train, record=False,
                             render=args.render or (k + 1) % args.render_every == 0,
                             print_every=args.print_every, eps=eps)

            log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
            print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

            # save models
            if (k + 1) % args.save_every == 0 and args.train:
                print("save model... ")
                for model in models:
                    model.save(savedir, k)