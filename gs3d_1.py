import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import laplace
from tqdm import tqdm
import cv2
import os
from matplotlib.colors import LinearSegmentedColormap

size = 128
Du, Dv = 0.16, 0.08
F, k = 0.037, 0.0625
dt = 1.0
steps = 1000
plot_interval = 10
threshold = 0.2
frame_dir = "frames"
video_name = "grayscott3d_minimal.mp4"
os.makedirs(frame_dir, exist_ok=True)

colors = [(0.02, 0.02, 0.1), (0.1, 0.0, 0.3), (0.0, 0.2, 0.8), (0.0, 0.8, 0.9), (0.4, 1.0, 0.6), (1.0, 0.8, 0.0), (1.0, 0.2, 0.8)]
custom_cmap = LinearSegmentedColormap.from_list("cyberpunk", colors, N=256)

U = np.ones((size, size, size))
V = np.zeros((size, size, size))

center = size // 2
radius = 8
z, y, x = np.ogrid[:size, :size, :size]
mask = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= radius**2
sphere = np.exp(-0.5 * ((x - center)**2 + (y - center)**2 + (z - center)**2) / (radius/2)**2)

V += 0.5 * sphere
U -= 0.5 * sphere

def step(U, V, Du, Dv, F, k, dt):
    Lu = laplace(U, mode='wrap')
    Lv = laplace(V, mode='wrap')
    reaction = U * V * V
    dU = Du * Lu - reaction + F * (1 - U)
    dV = Dv * Lv + reaction - (F + k) * V
    U += dU * dt
    V += dV * dt
    np.clip(U, 0, 1, out=U)
    np.clip(V, 0, 1, out=V)
    return U, V

def render_and_save(V, frame_num, threshold=0.2):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    x, y, z = np.where(V > threshold)

    if len(x) > 0:
        c = V[x, y, z]
        norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
        colors_mapped = custom_cmap(norm)
        ax.scatter(x, y, z, c=colors_mapped, marker='o', alpha=0.8, s=12, edgecolors='none')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    ax.axis('off')

    plt.tight_layout()
    path = os.path.join(frame_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(path, dpi=150, facecolor='black')
    plt.close()
    return path

saved_frames = []
for t in tqdm(range(steps)):
    U, V = step(U, V, Du, Dv, F, k, dt)
    if t % plot_interval == 0:
        frame_path = render_and_save(V, t, threshold)
        saved_frames.append(frame_path)

sample = cv2.imread(saved_frames[0])
h, w, _ = sample.shape
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
for path in saved_frames:
    img = cv2.imread(path)
    out.write(img)
out.release()

for frame_path in saved_frames:
    os.remove(frame_path)
os.rmdir(frame_dir)
