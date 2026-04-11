import matplotlib.pyplot as plt
import numpy as np

from find_lines import find_straight, find_diagonal, Line

def find_closer(ini_circles, fin_circles):
    # ini_left, ini_right = ini_circles
    # fin_left, fin_right = fin_circles

    smallest_distance = float('inf')
    smallest_idx = None

    for idx_ini, ini_circle in enumerate(ini_circles):
        for idx_fin, fin_circle in enumerate(fin_circles):

            ini_x, ini_y = ini_circle.center
            fin_x, fin_y = fin_circle.center
            dis = (ini_x - fin_x)**2 + (ini_y - fin_y)**2

            if dis < smallest_distance:
                smallest_distance = dis
                smallest_idx = (idx_ini, idx_fin)

    return ini_circles[smallest_idx[0]], fin_circles[smallest_idx[1]]



fig = plt.gcf()
ax = fig.gca()
ax.set_aspect(1)


initl_conf = (0, 0, np.deg2rad(90))
final_conf = (2, 2, np.deg2rad(0))

L = 1
max_turn_angle = 20

min_turn_r = L / np.tan(max_turn_angle)
# print(f'{min_turn_r=}')

ini_x, ini_y, ini_a = initl_conf

ini_left = plt.Circle((ini_x-min_turn_r*np.sin(ini_a), ini_y+min_turn_r*np.cos(ini_a)), min_turn_r)
ini_right = plt.Circle((ini_x+min_turn_r*np.sin(ini_a), ini_y-min_turn_r*np.cos(ini_a)), min_turn_r)

ax.add_patch(ini_left)
ax.add_patch(ini_right)

plt.plot([ini_x, ini_x+min_turn_r*np.cos(ini_a)], [ini_y, ini_y+min_turn_r*np.sin(ini_a)], color='r')


fin_x, fin_y, fin_a = final_conf

fin_left = plt.Circle((fin_x-min_turn_r*np.sin(fin_a), fin_y+min_turn_r*np.cos(fin_a)), min_turn_r)
fin_right = plt.Circle((fin_x+min_turn_r*np.sin(fin_a), fin_y-min_turn_r*np.cos(fin_a)), min_turn_r)

ax.add_patch(fin_left)
ax.add_patch(fin_right)

plt.plot([fin_x, fin_x+min_turn_r*np.cos(fin_a)], [fin_y, fin_y+min_turn_r*np.sin(fin_a)], color='r')
xs = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)

ini_circles = [ini_left, ini_right]
fin_circles = [fin_left, fin_right]

print(f'{ini_left}, {fin_right}')
# correct_straight_1, correct_straight_2 = find_straight(ini_circles, fin_circles, initl_conf, final_conf, None)
# plt.plot(xs, correct_straight_1(xs))
# plt.plot(xs, correct_straight_2(xs))

# ini_closer, fin_closer = find_closer([ini_left, ini_right], [fin_left, fin_right])
# print(fin_closer)

# dx, dy = ini_closer.center[0] - fin_closer.center[0], ini_closer.center[1] - fin_closer.center[1]
# beta = np.arctan(1/(dx/dy))
# straight_tangent = np.tan(beta)*(xs - min_turn_r + min_turn_r*np.sin(beta)) + min_turn_r*np.cos(beta)
# other_straight_tangent = np.tan(beta) * (xs - min_turn_r - min_turn_r*np.sin(beta)) - min_turn_r*np.cos(beta)

# plt.plot(xs, np.tan(beta)*xs - min_turn_r, color='g')
# plt.plot(xs, straight_tangent)
# plt.plot(xs, other_straight_tangent)

# ini_point = (ini_closer.center[0] - min_turn_r*np.sin(beta), ini_closer.center[1] + min_turn_r*np.cos(beta))
# print(ini_point)
# plt.scatter([ini_point[0]], [ini_point[1]], color='red')

correct_diagonal_1, correct_diagonal_2 = find_diagonal(ini_circles, fin_circles, initl_conf, final_conf, None)
plt.plot(xs, correct_diagonal_1(xs))
plt.plot(xs, correct_diagonal_2(xs))

plt.show()