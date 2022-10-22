"""play the video"""

import matplotlib.pyplot as plt
import numpy as np
import os


files_name = os.listdir("./build/render")

def file_filter(f):
    if f[-4:] in ['.txt']:
        return True
    else:
        return False

files_name = list(filter(file_filter, files_name))
print(files_name)



for filename in files_name:
    text = []       #txt的所有内容
    agent = []      #F行的内容
    num_agent = []  #F的行数和此帧的智能体数

    n_pic = 0
    n = 0
    file_name = "./build/render/"+filename

    with open(file_name,'r') as f:
        for line in f:
            n = n + 1
            text.append(list(line.strip(' ').split()))
            if line[0] == 'F':
                n_pic = n_pic + 1
                agent.append(list(line.strip(' ').split()))
                num_agent.append([n,int(agent[n_pic-1][1])])


    n_wall = int(text[0][1])
    print(n_wall)
    print(text[n_wall+1][0])

    #得到画墙的信息
    x_wall = []
    y_wall = []
    for i in range(1,n_wall+1):
        x_wall.append(int(text[i][0]))
        y_wall.append(int(text[i][1]))

    text_agent_all = [] #每一帧的智能体的信息
    text_agent = [] #需要画的agent

    #得到画agent的信息
    for index, num in num_agent:
        for i in range(num):
            text_agent.append(list(map(int, text[index+i])))
        text_agent = []
        text_agent_all.append(text_agent)

    x1_agent = []
    y1_agent = []
    x2_agent = []
    y2_agent = []
    x_food = []
    y_food = []

    for i in range(n_pic):
        for li in text_agent_all[i]:
            if li[5] == 0:
                x1_agent.append(li[3])
                y1_agent.append(li[4])
            elif li[5] == 1:
                x2_agent.append(li[3])
                y2_agent.append(li[4])
            elif li[5] == 2 or li[5] == 3:
                x_food.append(li[3])
                y_food.append(li[4])
        plt.figure(figsize=(10,10))
        plt.plot(x_wall, y_wall, 'ks', x1_agent, y1_agent, 'rs', x2_agent, y2_agent, 'bs', x_food, y_food, 'y*', markersize=4)
        name = filename[:filename.index('.')]
        if not os.path.exists('./build/'+name):
            os.mkdir('./build/'+name)
        plt.savefig('./build/{}/pic-{}.png'.format(name,i + 1))
        # plt.show()
        plt.close()
        x1_agent = []
        y1_agent = []
        x2_agent = []
        y2_agent = []
        x_food = []
        y_food = []


    # plt.plot(x_wall,y_wall,'bs')
    # plt.show()




