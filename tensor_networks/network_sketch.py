#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

fig_dir = "./figures/"

params = { "font.family" : "sans-serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : 25 }
plt.rcParams.update(params)


def circ(center, radius = 0.3, tick_width = 2):
    circle = plt.Circle(center, radius = radius, linewidth = tick_width,
                        facecolor = "white", edgecolor = "black", zorder = 3)
    plt.gca().add_patch(circle)


plt.figure(figsize = (4,3))

node_nums = [ 4, 3, 4, 2 ]
layers = len(node_nums)

def pos(layer, node, vscale = 1, hscale = 1.2):
    assert(node < node_nums[layer])
    return [ ( layer % layers ) * hscale,
             ( node_nums[layer]/2 - node ) * vscale ]

for layer in range(layers):
    for node in range(node_nums[layer]):
        circ(pos(layer,node))

for layer in range(layers-1):
    for fst in range(node_nums[layer]):
        fst_pos = pos(layer, fst)
        for snd in range(node_nums[layer+1]):
            snd_pos = pos(layer+1, snd)
            plt.plot(*list(zip(fst_pos,snd_pos)), "k")

for node in range(node_nums[0]):
    node_pos = pos(0, node)
    plt.plot([ node_pos[0], node_pos[0]-0.5 ], [node_pos[1]]*2, "k")

for node in range(node_nums[-1]):
    node_pos = pos(-1, node)
    plt.plot([ node_pos[0], node_pos[0]+0.5 ], [node_pos[1]]*2, "k")

plt.axis("scaled")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "network.pdf")
