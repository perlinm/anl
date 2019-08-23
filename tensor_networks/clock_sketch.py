#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

spokes = 12

fig_dir = "./figures/"

params = { "font.family" : "sans-serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : 25 }
plt.rcParams.update(params)

angles = np.arange(spokes) * 2*np.pi/spokes

def spoke_loc(spoke):
    return np.array([ np.sin(angles[spoke]), np.cos(angles[spoke]) ])

def plot_clock(center, spoke, tick_width = 3,
               arrow_width = 0.3, tick_length = 0.2):
    center = np.array(center)

    circle = plt.Circle(center, radius = 1, linewidth = tick_width,
                        facecolor = "white", edgecolor = "black", zorder = 3)
    plt.gca().add_patch(circle)

    for tick in range(spokes):
        xx = center + spoke_loc(tick)
        yy = center + spoke_loc(tick) * (1-tick_length)
        plt.plot(*list(zip(xx,yy)), "k", linewidth = tick_width, zorder = 4)

    arrow_tip = spoke_loc(spoke)
    arrow = plt.Arrow(*center, *arrow_tip, width = arrow_width,
                      color = "red", zorder = 5)
    plt.gca().add_patch(arrow)

### single clock

plt.figure(figsize = (2,2))

plot_clock([0,0], 4)

plt.axis("scaled")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "clock.pdf")

### network of clocks

plt.figure(figsize = (4,4))

step = 2
waves = 3
wave_amp = 0.3

color = "C0"

basewidth = 1
strong = basewidth * 2.5
weak = basewidth

# main clocks
plot_clock([-step,-step],-1, 2, 0.4)
plot_clock([-step,+step], 1, 2, 0.4)
plot_clock([+step,-step], 0, 2, 0.4)
plot_clock([+step,+step], 2, 2, 0.4)

wave_vals = np.linspace(0,waves) * 2*np.pi
wave_peaks = wave_amp * np.sin(wave_vals)
wave_vals *= 2*step/wave_vals[-1]

# interactions between clocks
plt.plot(wave_vals-step, wave_peaks-step, color, linewidth = weak)
plt.plot(wave_vals-step, wave_peaks+step, color, linewidth = weak)
plt.plot(wave_peaks-step, wave_vals-step, color, linewidth = strong)
plt.plot(wave_peaks+step, wave_vals-step, color, linewidth = strong)

num_vals = len(wave_vals)*2//3
wave_vals = wave_vals[:num_vals]
wave_peaks = wave_peaks[:num_vals]

# external interactions -- horizontal
plt.plot(-wave_vals-step, wave_peaks-step, color, linewidth = strong)
plt.plot(+wave_vals+step, wave_peaks-step, color, linewidth = weak)
plt.plot(-wave_vals-step, wave_peaks+step, color, linewidth = weak)
plt.plot(+wave_vals+step, wave_peaks+step, color, linewidth = strong)

# external interactions -- vertical
plt.plot(wave_peaks-step, -wave_vals-step, color, linewidth = weak)
plt.plot(wave_peaks-step, +wave_vals+step, color, linewidth = strong)
plt.plot(wave_peaks+step, -wave_vals-step, color, linewidth = strong)
plt.plot(wave_peaks+step, +wave_vals+step, color, linewidth = weak)

mid = np.array([ 0, -3*step ])
vec = step * spoke_loc(1)

start = mid-vec/2
arrow = plt.Arrow(*start, *vec, width = 0.6, color = "C0")
plt.gca().add_patch(arrow)

plt.text(start[0]+step/2, start[1], r"$h$")

plt.axis("scaled")
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "clock_network.pdf")
