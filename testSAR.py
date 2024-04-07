import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Aircraft:
    def __init__(self, position, velocity, radar_range):
        self.position = position
        self.velocity = velocity
        self.radar_range = radar_range

    def emit_radar(self):
        return np.random.rand() * self.radar_range

    def move(self, time_interval):
        self.position += self.velocity * time_interval


reflective_spot_positions = np.array([[5.0,7.0,0.0],[3.0,6.0,0.0],[7.0,5.0,0.0]])

def simulate_flight(duration, time_interval, radar_range, radar_samples):
    aircraft = Aircraft(position=np.array([0.0,0.0,0.0]), velocity=np.array([1.0,0.0,0.0]), radar_range=radar_range)
    num_steps = int(duration / time_interval)
    echo_data = []

    for step in range(num_steps):
    #for step in range(num_steps):
        # Emit radar and record echo intensity
        echo_intensity = aircraft.emit_radar()

        lf = (lambda x : np.linalg.norm(aircraft.position - x))
        echo_dists = np.apply_along_axis(lf, axis=1, arr=reflective_spot_positions)
        #echo_dists = [np.linalg.norm(aircraft.position - reflective_spot_position)]
        echo_vals = np.zeros(radar_samples)
        #print(echo_dists)
        for d in echo_dists:
            index = int(d * radar_samples/(radar_range))
            if(not(index >= radar_samples or index < 0)):
                echo_vals[index] = 1

        # Add the echo data
        echo_data.append(echo_vals)

    # Move the aircraft
        aircraft.move(time_interval)

    return echo_data

def make_original_image(points):
    w, h = 10,10
    res = 100
    img = np.zeros((res,res))
    for p in points:
        img[int(p[0]*res/w),int(p[1]*res/h)] = 1
    return img


def draw_circle(img, center, radius, thickness=1.0, intensity=1.0):
    for index in np.ndindex(img.shape):
        if abs(np.linalg.norm(index - center) - radius) <= thickness/2:
            img[index] += intensity

def reconstruct(echo_data):
    w, h = 10, 10
    res = 100
    img = np.zeros((res, res))
    center_init = np.array((0.0,0.0))
    ed = np.array(echo_data)
    for index in np.ndindex(ed.shape):
        if ed[index] != 0:
            center = center_init + np.array([index[0]*res/10*0.1,0])
            radius = index[1]*res/(10 * 10)
            draw_circle(img, center, radius)
    return img


def plot_echo_data(echo_data, unmask):

    ed = np.zeros_like(echo_data)
    ed[0:unmask] = echo_data[0:unmask]

    orig_img = make_original_image(reflective_spot_positions)
    rec_img = reconstruct(ed)

    fig, ax = plt.subplots(1,3, figsize=(16,9))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    #ax[0].imshow(orig_img.T[100:200,50:200], cmap='gray', aspect='auto', origin='lower')
    ax[0].imshow(orig_img.T, cmap='gray', origin='lower', interpolation='nearest')
    ax[2].imshow(rec_img.T, cmap='gray', origin='lower', interpolation='nearest')

    im = ax[1].imshow(np.array(ed).T, cmap='gray', origin='lower', interpolation='nearest')
    ax[1].set_xlabel('Pulse emission time')
    ax[1].set_ylabel('Echo return time')
    ax[1].set_title('Echo Data')
    ax[0].set_title('Terrain')
    ax[2].set_title('Reconstruction of terrain')


    #fig.colorbar(im, ax=ax[0], label='Intensity')
    plt.savefig(f"radarimg{unmask}", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    duration = 15  # seconds
    time_interval = 0.1  # seconds
    radar_range = 10  # arbitrary units
    radar_samples = 100  # arbitrary units
    #terrain_width = 200  # arbitrary units

    echo_data = simulate_flight(duration, time_interval, radar_range, radar_samples)
    echo_data = np.array(echo_data)

    plot_echo_data(echo_data, 0)
    plot_echo_data(echo_data, 1)
    plot_echo_data(echo_data, 2)
    plot_echo_data(echo_data, 3)
    plot_echo_data(echo_data, 5)
    plot_echo_data(echo_data, 10)
    plot_echo_data(echo_data, 20)
    plot_echo_data(echo_data, 50)
    plot_echo_data(echo_data, 100)
    plot_echo_data(echo_data, np.array(echo_data).shape[0])
