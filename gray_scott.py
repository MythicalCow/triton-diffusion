import numpy as np
import cv2
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import time

# Increased resolution
width, height = 512, 512
Du, Dv = 0.16, 0.07
F, k = 0.037, 0.0625 # Classic mitosis parameters
dt = 1.0
video_fps = 200
steps = video_fps * 30 * 6
video_name = "grayscott_open_shapes.mp4"

# Initialize fields - start with the equilibrium state
U = np.ones((height, width), dtype=np.float32)
V = np.zeros((height, width), dtype=np.float32)

# Create more varied, artistic blob seeds with general shapes
np.random.seed(int(time.time()))  # Use current time for true randomness
num_blobs = 1  # Slightly more blobs for higher resolution

def create_organic_shape(center_x, center_y, base_size, num_points=8, open_shape=False):
    """Create an organic, irregular shape using polar coordinates"""
    if open_shape:
        # Create an open organic curve
        angles = np.linspace(0, np.pi * np.random.uniform(0.5, 1.5), num_points)
        radii = base_size * (0.5 + 0.7 * np.random.rand(num_points))

        # Add some smoothing to make it more organic
        radii = gaussian_filter(np.concatenate([radii, radii, radii]), sigma=0.8)[num_points:2*num_points]

        # Create curve points
        curve_x = center_x + radii * np.cos(angles)
        curve_y = center_y + radii * np.sin(angles)

        # Create mask from curve points
        mask = np.zeros((height, width), dtype=np.float32)
        thickness = base_size * 0.3

        for px, py in zip(curve_x, curve_y):
            if 0 <= px < width and 0 <= py < height:
                y, x = np.ogrid[:height, :width]
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                mask += np.exp(-0.5 * (dist / thickness)**2)

        return np.clip(mask, 0, 1)
    else:
        # Original closed organic shape
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        # Create random radii for each angle to make irregular shapes
        radii = base_size * (0.5 + 0.7 * np.random.rand(num_points))

        # Add some smoothing to make it more organic
        radii = gaussian_filter(np.concatenate([radii, radii, radii]), sigma=0.8)[num_points:2*num_points]

        # Create a mask for this shape
        mask = np.zeros((height, width), dtype=np.float32)
        y, x = np.ogrid[:height, :width]

        # For each pixel, check if it's inside the organic shape
        for i in range(height):
            for j in range(width):
                dx = j - center_x
                dy = i - center_y

                if dx == 0 and dy == 0:
                    mask[i, j] = 1.0
                    continue

                # Convert to polar coordinates
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)

                # Find the interpolated radius at this angle
                theta_norm = (theta + np.pi) / (2 * np.pi) * num_points
                idx = int(theta_norm) % num_points
                next_idx = (idx + 1) % num_points
                frac = theta_norm - int(theta_norm)

                interp_radius = radii[idx] * (1 - frac) + radii[next_idx] * frac

                # Create a soft falloff
                if r <= interp_radius:
                    mask[i, j] = np.exp(-0.5 * (r / interp_radius)**2)

        return mask

def create_squiggly_line(start_x, start_y, length, direction, thickness, frequency=3, amplitude=20):
    """Create a squiggly/wavy line shape"""
    mask = np.zeros((height, width), dtype=np.float32)

    # Create points along the main direction
    num_points = int(length)
    t = np.linspace(0, 1, num_points)

    # Base line points
    base_x = start_x + length * np.cos(direction) * t
    base_y = start_y + length * np.sin(direction) * t

    # Add squiggly variation perpendicular to the main direction
    perp_direction = direction + np.pi/2
    squiggle_offset = amplitude * np.sin(frequency * 2 * np.pi * t)

    # Final squiggly points
    squiggle_x = base_x + squiggle_offset * np.cos(perp_direction)
    squiggle_y = base_y + squiggle_offset * np.sin(perp_direction)

    # Vary thickness along the squiggle
    thickness_variation = thickness * (0.5 + 0.5 * np.sin(frequency * 1.5 * 2 * np.pi * t))

    # Create mask from squiggly points
    for i, (px, py) in enumerate(zip(squiggle_x, squiggle_y)):
        if 0 <= px < width and 0 <= py < height:
            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            current_thickness = thickness_variation[i]
            mask += np.exp(-0.5 * (dist / current_thickness)**2)

    return np.clip(mask, 0, 1)

def create_random_path(start_x, start_y, num_segments, segment_length, thickness):
    """Create a random meandering path"""
    mask = np.zeros((height, width), dtype=np.float32)

    current_x, current_y = start_x, start_y
    current_direction = np.random.uniform(0, 2*np.pi)

    all_points = [(current_x, current_y)]

    for _ in range(num_segments):
        # Add some randomness to direction
        current_direction += np.random.uniform(-np.pi/3, np.pi/3)

        # Move in current direction
        next_x = current_x + segment_length * np.cos(current_direction)
        next_y = current_y + segment_length * np.sin(current_direction)

        # Keep within bounds
        next_x = np.clip(next_x, thickness, width - thickness)
        next_y = np.clip(next_y, thickness, height - thickness)

        all_points.append((next_x, next_y))
        current_x, current_y = next_x, next_y

    # Create mask from all points
    for i in range(len(all_points) - 1):
        x1, y1 = all_points[i]
        x2, y2 = all_points[i + 1]

        # Interpolate between points
        steps = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        if steps > 0:
            xs = np.linspace(x1, x2, steps)
            ys = np.linspace(y1, y2, steps)

            for px, py in zip(xs, ys):
                y, x = np.ogrid[:height, :width]
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                mask += np.exp(-0.5 * (dist / thickness)**2)

    return np.clip(mask, 0, 1)

def create_zigzag_line(start_x, start_y, length, direction, thickness, zigzag_amplitude=30, zigzag_frequency=5):
    """Create a zigzag line shape"""
    mask = np.zeros((height, width), dtype=np.float32)

    # Create points along the main direction
    num_points = int(length)
    t = np.linspace(0, 1, num_points)

    # Base line points
    base_x = start_x + length * np.cos(direction) * t
    base_y = start_y + length * np.sin(direction) * t

    # Add zigzag variation perpendicular to the main direction
    perp_direction = direction + np.pi/2
    zigzag_offset = zigzag_amplitude * np.sign(np.sin(zigzag_frequency * 2 * np.pi * t))

    # Final zigzag points
    zigzag_x = base_x + zigzag_offset * np.cos(perp_direction)
    zigzag_y = base_y + zigzag_offset * np.sin(perp_direction)

    # Create mask from zigzag points
    for px, py in zip(zigzag_x, zigzag_y):
        if 0 <= px < width and 0 <= py < height:
            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            mask += np.exp(-0.5 * (dist / thickness)**2)

    return np.clip(mask, 0, 1)

def create_brush_stroke(start_x, start_y, end_x, end_y, thickness):
    """Create a brush stroke shape"""
    mask = np.zeros((height, width), dtype=np.float32)

    # Create points along the stroke
    num_points = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2))
    if num_points < 2:
        num_points = 2

    x_points = np.linspace(start_x, end_x, num_points)
    y_points = np.linspace(start_y, end_y, num_points)

    # Vary thickness along the stroke for more organic feel
    thickness_variation = thickness * (0.3 + 0.7 * np.random.rand(num_points))

    for i, (px, py) in enumerate(zip(x_points, y_points)):
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        current_thickness = thickness_variation[i]

        # Add to mask with gaussian falloff
        mask += np.exp(-0.5 * (dist / current_thickness)**2)

    return np.clip(mask, 0, 1)

def create_spiral_shape(center_x, center_y, max_radius, turns=2):
    """Create a spiral shape"""
    mask = np.zeros((height, width), dtype=np.float32)
    y, x = np.ogrid[:height, :width]

    # Create spiral points
    t = np.linspace(0, turns * 2 * np.pi, 1000)
    spiral_r = max_radius * t / (turns * 2 * np.pi)
    spiral_x = center_x + spiral_r * np.cos(t)
    spiral_y = center_y + spiral_r * np.sin(t)

    # Create thickness that varies along the spiral
    thickness = max_radius * 0.2 * (1 + 0.5 * np.sin(t * 3))

    for i, (sx, sy) in enumerate(zip(spiral_x, spiral_y)):
        if 0 <= sx < width and 0 <= sy < height:
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            mask += np.exp(-0.5 * (dist / thickness[i])**2)

    return np.clip(mask, 0, 1)

# Create different types of artistic shapes
for i in range(num_blobs):
    # Random center for each blob (scaled for higher resolution)
    cx = np.random.randint(160, width-160)
    cy = np.random.randint(160, height-160)

    # Choose random shape type - now includes more open shapes
    shape_type = np.random.choice(['organic', 'organic_open'])

    if shape_type == 'organic':
        # Organic irregular shape (closed)
        base_size = np.random.uniform(30, 80)  # Scaled for higher resolution
        blob = create_organic_shape(cx, cy, base_size, open_shape=False)

    elif shape_type == 'organic_open':
        # Organic irregular shape (open)
        base_size = np.random.uniform(30, 80)  # Scaled for higher resolution
        blob = create_organic_shape(cx, cy, base_size, open_shape=True)

    elif shape_type == 'brush':
        # Brush stroke shape
        # Create a random brush stroke
        angle = np.random.uniform(0, 2*np.pi)
        length = np.random.uniform(60, 150)  # Scaled for higher resolution
        end_x = cx + length * np.cos(angle)
        end_y = cy + length * np.sin(angle)
        thickness = np.random.uniform(15, 35)  # Scaled for higher resolution
        blob = create_brush_stroke(cx, cy, end_x, end_y, thickness)

    elif shape_type == 'spiral':
        # Spiral shape
        max_radius = np.random.uniform(40, 90)  # Scaled for higher resolution
        turns = np.random.uniform(1.5, 3.0)
        blob = create_spiral_shape(cx, cy, max_radius, turns)

    elif shape_type == 'squiggly':
        # Squiggly line shape
        direction = np.random.uniform(0, 2*np.pi)
        length = np.random.uniform(80, 200)  # Scaled for higher resolution
        thickness = np.random.uniform(10, 25)  # Scaled for higher resolution
        frequency = np.random.uniform(2, 5)
        amplitude = np.random.uniform(15, 40)  # Scaled for higher resolution
        blob = create_squiggly_line(cx, cy, length, direction, thickness, frequency, amplitude)

    elif shape_type == 'random_path':
        # Random meandering path
        num_segments = np.random.randint(5, 15)
        segment_length = np.random.uniform(20, 50)  # Scaled for higher resolution
        thickness = np.random.uniform(8, 20)  # Scaled for higher resolution
        blob = create_random_path(cx, cy, num_segments, segment_length, thickness)

    else:  # zigzag
        # Zigzag line shape
        direction = np.random.uniform(0, 2*np.pi)
        length = np.random.uniform(100, 250)  # Scaled for higher resolution
        thickness = np.random.uniform(12, 28)  # Scaled for higher resolution
        zigzag_amplitude = np.random.uniform(20, 50)  # Scaled for higher resolution
        zigzag_frequency = np.random.uniform(3, 8)
        blob = create_zigzag_line(cx, cy, length, direction, thickness, zigzag_amplitude, zigzag_frequency)

    # Random intensity for each blob
    intensity = np.random.uniform(0.6, 1.0)

    # Add to V and subtract from U
    V += intensity * blob
    U -= 0.4 * intensity * blob

# Clip to valid ranges
np.clip(U, 0, 1, out=U)
np.clip(V, 0, 1, out=V)

# Add small random noise everywhere (scaled for higher resolution)
U += 0.005 * (np.random.rand(height, width) - 0.5).astype(np.float32)
V += 0.005 * (np.random.rand(height, width) - 0.5).astype(np.float32)

# Final clipping
np.clip(U, 0, 1, out=U)
np.clip(V, 0, 1, out=V)

print(f"Created {num_blobs} random artistic blob seeds")
print(f"Initial U range: [{U.min():.4f}, {U.max():.4f}]")
print(f"Initial V range: [{V.min():.4f}, {V.max():.4f}]")

def laplacian(Z):
    return (
        -4 * Z +
        np.roll(Z, 1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) +
        np.roll(Z, -1, axis=1)
    )

def update(U, V):
    Lu = laplacian(U)
    Lv = laplacian(V)
    reaction = U * V * V
    U += (Du * Lu - reaction + F * (1 - U)) * dt
    V += (Dv * Lv + reaction - (F + k) * V) * dt
    np.clip(U, 0, 1, out=U)
    np.clip(V, 0, 1, out=V)
    return U, V

# Vibrant cyberpunk colormap
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

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, video_fps, (width, height))

for step in tqdm(range(steps)):
    if step % 500 == 0:
        if np.random.randint(0,3) != 0:
            F -= 0.0014
            k += 0.0014
        else:
            F += 0.0016
            k -= 0.0016

    U, V = update(U, V)

    # Debug: print values every 100 steps
    if step % 100 == 0:
        print(f"Step {step}: U range [{U.min():.4f}, {U.max():.4f}], V range [{V.min():.4f}, {V.max():.4f}]")

    if step % 5 == 0:
        # Use V for visualization
        img = V.copy()

        # Check if we have any variation
        v_min, v_max = V.min(), V.max()
        if v_max - v_min > 1e-6:
            img = (img - v_min) / (v_max - v_min)
        else:
            # If no variation, try visualizing U instead
            img = U.copy()
            u_min, u_max = U.min(), U.max()
            if u_max - u_min > 1e-6:
                img = (img - u_min) / (u_max - u_min)
            else:
                img = img * 0

        # Enhanced contrast and saturation
        img = img ** 0.5  # Less gamma correction for more vibrant colors

        # Add slight contrast boost
        img = np.clip(img * 1.2 - 0.1, 0, 1)

        # Convert to color
        img_color = custom_cmap(img)[..., :3]
        img_uint8 = (img_color * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

video.release()
print(f"Video saved as {video_name}")

# Save final frame for debugging
final_frame = V.copy()
if V.max() - V.min() > 1e-6:
    final_frame = (final_frame - V.min()) / (V.max() - V.min())
else:
    final_frame = U.copy()
    if U.max() - U.min() > 1e-6:
        final_frame = (final_frame - U.min()) / (U.max() - U.min())

final_frame = (final_frame * 255).astype(np.uint8)
cv2.imwrite("final_frame.png", final_frame)
print("Final frame saved as final_frame.png")
print(f"Final V range: [{V.min():.4f}, {V.max():.4f}]")
print(f"Final U range: [{U.min():.4f}, {U.max():.4f}]")
