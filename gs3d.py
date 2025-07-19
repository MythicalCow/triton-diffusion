import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import laplace, gaussian_filter
from tqdm import tqdm
import cv2
import os
import time
from matplotlib.colors import LinearSegmentedColormap

# --- Parameters ---
size = 64
Du, Dv = 0.16, 0.08
F, k = 0.037, 0.0625  # Using mitosis parameters from 2D version
dt = 1.0
steps = 1000
plot_interval = 10
threshold = 0.2
frame_dir = "frames"
video_name = "grayscott3d_enhanced.mp4"
os.makedirs(frame_dir, exist_ok=True)

# Vibrant cyberpunk colormap from 2D version
colors = [
    (0.02, 0.02, 0.1),    # Deep dark blue-black
    (0.1, 0.0, 0.3),      # Dark purple
    (0.0, 0.2, 0.8),      # Electric blue
    (0.0, 0.8, 0.9),      # Bright cyan
    (0.4, 1.0, 0.6),      # Bright green-cyan
    (1.0, 0.8, 0.0),      # Electric yellow
    (1.0, 0.2, 0.8)       # Hot pink
]
custom_cmap = LinearSegmentedColormap.from_list("cyberpunk", colors, N=256)

# --- Initialize ---
U = np.ones((size, size, size))
V = np.zeros((size, size, size))

# Adapted 3D shape creation functions
def create_organic_shape_3d(center_x, center_y, center_z, base_size, num_points=8):
    """Create a 3D organic shape using spherical coordinates"""
    # Create random radii for different directions
    phi_angles = np.linspace(0, np.pi, num_points//2)  # elevation
    theta_angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)  # azimuth

    # Create 3D mask
    mask = np.zeros((size, size, size), dtype=np.float32)
    z, y, x = np.ogrid[:size, :size, :size]

    # Create organic shape by varying radius in different directions
    for i in range(size):
        for j in range(size):
            for k in range(size):
                dx = k - center_x
                dy = j - center_y
                dz = i - center_z

                if dx == 0 and dy == 0 and dz == 0:
                    mask[i, j, k] = 1.0
                    continue

                # Convert to spherical coordinates
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                theta = np.arctan2(dy, dx)
                phi = np.arccos(dz / (r + 1e-8))

                # Create organic variation
                organic_radius = base_size * (0.7 + 0.5 * np.sin(3*theta) * np.cos(2*phi) +
                                            0.3 * np.cos(5*theta) * np.sin(3*phi))

                # Soft falloff
                if r <= organic_radius:
                    mask[i, j, k] = np.exp(-0.5 * (r / organic_radius)**2)

    return mask

def create_3d_spiral(center_x, center_y, center_z, max_radius, height_range, turns=2):
    """Create a 3D spiral shape"""
    mask = np.zeros((size, size, size), dtype=np.float32)

    # Create spiral points
    t = np.linspace(0, turns * 2 * np.pi, 500)
    spiral_r = max_radius * t / (turns * 2 * np.pi)
    spiral_x = center_x + spiral_r * np.cos(t)
    spiral_y = center_y + spiral_r * np.sin(t)
    spiral_z = center_z + height_range * (t / (turns * 2 * np.pi) - 0.5)

    # Create thickness that varies along the spiral
    thickness = max_radius * 0.3 * (1 + 0.5 * np.sin(t * 3))

    for i, (sx, sy, sz) in enumerate(zip(spiral_x, spiral_y, spiral_z)):
        if 0 <= sx < size and 0 <= sy < size and 0 <= sz < size:
            z, y, x = np.ogrid[:size, :size, :size]
            dist = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
            mask += np.exp(-0.5 * (dist / thickness[i])**2)

    return np.clip(mask, 0, 1)

def create_3d_torus(center_x, center_y, center_z, major_radius, minor_radius):
    """Create a 3D torus shape"""
    mask = np.zeros((size, size, size), dtype=np.float32)
    z, y, x = np.ogrid[:size, :size, :size]

    # Torus equation: (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
    dx = x - center_x
    dy = y - center_y
    dz = z - center_z

    # Distance from z-axis
    rho = np.sqrt(dx**2 + dy**2)

    # Torus distance
    torus_dist = np.sqrt((rho - major_radius)**2 + dz**2)

    # Soft falloff
    mask = np.exp(-0.5 * (torus_dist / minor_radius)**2)

    return mask

# Create artistic 3D initial conditions
np.random.seed(int(time.time()))
num_shapes = 3

for i in range(num_shapes):
    # Random center for each shape
    cx = np.random.randint(size//4, 3*size//4)
    cy = np.random.randint(size//4, 3*size//4)
    cz = np.random.randint(size//4, 3*size//4)

    # Choose random shape type
    shape_type = np.random.choice(['organic', 'spiral', 'torus'])

    if shape_type == 'organic':
        base_size = np.random.uniform(size//8, size//4)
        blob = create_organic_shape_3d(cx, cy, cz, base_size)
    elif shape_type == 'spiral':
        max_radius = np.random.uniform(size//6, size//3)
        height_range = np.random.uniform(size//4, size//2)
        turns = np.random.uniform(1.5, 3.0)
        blob = create_3d_spiral(cx, cy, cz, max_radius, height_range, turns)
    else:  # torus
        major_radius = np.random.uniform(size//6, size//3)
        minor_radius = np.random.uniform(size//12, size//6)
        blob = create_3d_torus(cx, cy, cz, major_radius, minor_radius)

    # Random intensity for each shape
    intensity = np.random.uniform(0.6, 1.0)

    # Add to V and subtract from U
    V += intensity * blob
    U -= 0.4 * intensity * blob

# Clip to valid ranges
np.clip(U, 0, 1, out=U)
np.clip(V, 0, 1, out=V)

# Add small random noise everywhere
U += 0.005 * (np.random.rand(size, size, size) - 0.5).astype(np.float32)
V += 0.005 * (np.random.rand(size, size, size) - 0.5).astype(np.float32)

# Final clipping
np.clip(U, 0, 1, out=U)
np.clip(V, 0, 1, out=V)

print(f"Created {num_shapes} random artistic 3D shapes")
print(f"Initial U range: [{U.min():.4f}, {U.max():.4f}]")
print(f"Initial V range: [{V.min():.4f}, {V.max():.4f}]")

# --- Step Function ---
def step(U, V, Du, Dv, F, k, dt):
    Lu = laplace(U, mode='wrap')
    Lv = laplace(V, mode='wrap')
    reaction = U * V * V
    dU = Du * Lu - reaction + F * (1 - U)
    dV = Dv * Lv + reaction - (F + k) * V
    U += dU * dt
    V += dV * dt
    np.clip(U, 0, 1, out=U)  # Add clipping like in 2D version
    np.clip(V, 0, 1, out=V)
    return U, V

# --- Enhanced Render + Save Frame ---
def render_and_save(V, frame_num, threshold=0.2):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Find points above threshold
    x, y, z = np.where(V > threshold)

    if len(x) > 0:
        c = V[x, y, z]

        # Normalize colors similar to 2D version
        v_min, v_max = c.min(), c.max()
        if v_max - v_min > 1e-6:
            norm = (c - v_min) / (v_max - v_min)
        else:
            norm = np.zeros_like(c)

        # Apply gamma correction like in 2D version
        norm = norm ** 0.5

        # Add contrast boost
        norm = np.clip(norm * 1.2 - 0.1, 0, 1)

        # Use the cyberpunk colormap
        colors_mapped = custom_cmap(norm)

        # Create scatter plot with enhanced visual settings
        ax.scatter(x, y, z, c=colors_mapped, marker='o', alpha=0.6, s=8,
                  edgecolors='none')

    # Set limits and styling
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    ax.axis('off')

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=frame_num * 2)  # Slowly rotate the view

    plt.tight_layout()
    path = os.path.join(frame_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(path, dpi=150, facecolor='black')
    plt.close()
    return path

# --- Simulation Loop ---
saved_frames = []
for t in tqdm(range(steps)):
    U, V = step(U, V, Du, Dv, F, k, dt)

    # Debug: print values every 100 steps like in 2D version
    if t % 100 == 0:
        print(f"Step {t}: U range [{U.min():.4f}, {U.max():.4f}], V range [{V.min():.4f}, {V.max():.4f}]")

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
print(f"Final V range: [{V.min():.4f}, {V.max():.4f}]")
print(f"Final U range: [{U.min():.4f}, {U.max():.4f}]")

# Clean up frame files
for frame_path in saved_frames:
    os.remove(frame_path)
os.rmdir(frame_dir)
print("Cleaned up temporary frame files")
