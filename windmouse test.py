import numpy as np
import matplotlib.pyplot as plt

sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)

# Bewegungsfunktionen
def move_left(positions):
    positions.append((positions[-1][0] - 1, positions[-1][1]))

def move_right(positions):
    positions.append((positions[-1][0] + 1, positions[-1][1]))

def move_up(positions):
    positions.append((positions[-1][0], positions[-1][1] - 1))

def move_down(positions):
    positions.append((positions[-1][0], positions[-1][1] + 1))

def move_up_right(positions):
    positions.append((positions[-1][0] + 1, positions[-1][1] - 1))

def move_up_left(positions):
    positions.append((positions[-1][0] - 1, positions[-1][1] - 1))

def move_down_right(positions):
    positions.append((positions[-1][0] + 1, positions[-1][1] + 1))

def move_down_left(positions):
    positions.append((positions[-1][0] - 1, positions[-1][1] + 1))

def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=5, D_0=12, tolerance=1, move_mouse=lambda x, y: None):
    '''
    WindMouse algorithm. Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational force
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    tolerance - tolerance radius for the destination
    '''
    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0
    positions = [(current_x, current_y)]

    while (dist := np.hypot(dest_x - start_x, dest_y - start_y)) >= tolerance:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
            W_y = W_y / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random() * 3 + 3
            else:
                M_0 /= sqrt5

        v_x += W_x + G_0 * (dest_x - start_x) / dist
        v_y += W_y + G_0 * (dest_y - start_y) / dist
        v_mag = np.hypot(v_x, v_y)

        if v_mag > M_0:
            v_clip = M_0 / 2 + np.random.random() * M_0 / 2
            v_x = (v_x / v_mag) * v_clip
            v_y = (v_y / v_mag) * v_clip

        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))

        if current_x != move_x or current_y != move_y:
            while current_x < move_x and current_y < move_y:
                move_down_right(positions)
                current_x += 1
                current_y += 1
            while current_x < move_x and current_y > move_y:
                move_up_right(positions)
                current_x += 1
                current_y -= 1
            while current_x > move_x and current_y < move_y:
                move_down_left(positions)
                current_x -= 1
                current_y += 1
            while current_x > move_x and current_y > move_y:
                move_up_left(positions)
                current_x -= 1
                current_y -= 1
            while current_x < move_x:
                move_right(positions)
                current_x += 1
            while current_x > move_x:
                move_left(positions)
                current_x -= 1
            while current_y < move_y:
                move_down(positions)
                current_y += 1
            while current_y > move_y:
                move_up(positions)
                current_y -= 1

    # Ensure final position is exactly at the destination
    positions.append((dest_x, dest_y))
    return positions


# Beispielanwendung und Plotting der Pfade
fig = plt.figure(figsize=[13, 13])
plt.axis('off')
for y in np.linspace(-200, 200, 25):
    positions = wind_mouse(0, y, 500, y)
    positions = np.asarray(positions)
    plt.plot(positions[:, 0], positions[:, 1])
plt.xlim(-50, 550)
plt.ylim(-250, 250)

# Speichern der Grafik in einer Datei
plt.savefig("wind_mouse_paths.png")
print("Die Grafik wurde als 'wind_mouse_paths.png' gespeichert.")
