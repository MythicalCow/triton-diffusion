import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import laplace
from tqdm import tqdm
import cv2
import os

# --- Parameters ---
size = 64
Du, Dv = 0.16, 0.08
F, k = 0.060, 0.062
dt = 1.0
steps = 1000
plot_interval = 10  # more granular now
threshold = 0.2
frame_dir = "frames"
video_name = "grayscott3d.mp4"
os.makedirs(frame_dir, exist_ok=True)

# --- Initialize ---
U = np.ones((size, size, size))
V = np.zeros((size, size, size))
s = size // 8
U[size//2 - s:size//2 + s, size//2 - s:size//2 + s, size//2 - s:size//2 + s] = 0.50
V[size//2 - s:size//2 + s, size//2 - s:size//2 + s, size//2 - s:size//2 + s] = 0.25

# --- Step Function ---
def step(U, V, Du, Dv, F, k, dt):
    Lu = laplace(U, mode='wrap')
    Lv = laplace(V, mode='wrap')
    reaction = U * V * V
    dU = Du * Lu - reaction + F * (1 - U)
    dV = Dv * Lv + reaction - (F + k) * V
    U += dU * dt
    V += dV * dt
    return U, V

# --- Render + Save Frame ---
def render_and_save(V, frame_num, threshold=0.2):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    x, y, z = np.where(V > threshold)
    c = V[x, y, z]
    norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
    ax.scatter(x, y, z, c=norm, cmap='inferno', marker='o', alpha=0.4, s=5)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(frame_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# --- Simulation Loop ---
saved_frames = []
for t in tqdm(range(steps)):
    U, V = step(U, V, Du, Dv, F, k, dt)
    if t % plot_interval == 0:
        frame_path = render_and_save(V, t, threshold)
        saved_frames.append(frame_path)

# --- Make Video ---
sample = cv2.imread(saved_frames[0])
h, w, _ = sample.shape
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
for path in saved_frames:
    img = cv2.imread(path)
    out.write(img)
out.release()
print(f"✅ Saved video: {video_name}")
