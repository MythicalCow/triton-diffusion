import modal
from io import BytesIO
import triton
import triton.language as tl

# Create Modal app
app = modal.App("gray-scott-triton")

# Define image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "torch",
    "triton",
    "opencv-python",
    "tqdm",
    "matplotlib",
    "scipy",
    "numpy"
]).apt_install([
    "libgl1-mesa-glx",
    "libglib2.0-0"
])

@app.function(image=image, gpu="T4", timeout=1800)
def run_gray_scott_simulation():
    """Run Gray-Scott simulation - ported from original code"""

    # All imports happen inside Modal function
    import torch
    import cv2
    from tqdm import tqdm
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    import time
    import numpy as np
    import tempfile
    import os

    # Define Triton kernel after imports
    @triton.jit
    def test_triton_kernel(
        input_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        output = x * 2.0  # Simple test operation
        tl.store(output_ptr + offsets, output, mask=mask)

    # Set device for GPU acceleration for Triton test
    gpu_device = torch.device("cuda")
    print(f"Testing Triton on device: {gpu_device}")

    # Test Triton installation
    print("Testing Triton...")
    test_tensor = torch.randn(1024, device=gpu_device)
    output_tensor = torch.zeros_like(test_tensor)

    grid = lambda meta: (triton.cdiv(test_tensor.numel(), meta['BLOCK_SIZE']),)
    test_triton_kernel[grid](test_tensor, output_tensor, test_tensor.numel(), BLOCK_SIZE=256)

    print(f"Triton test - Input mean: {test_tensor.mean():.4f}, Output mean: {output_tensor.mean():.4f}")
    print("Triton working correctly!")

    # Set device to CPU for the actual simulation
    device = torch.device("cpu")
    print(f"Running Gray-Scott simulation on device: {device}")

    # Original simulation parameters
    width, height = 512, 512
    Du, Dv = 0.16, 0.07
    F, k = 0.037, 0.0625
    dt = 1.0
    video_fps = 200

    steps = 6000
    video_name = "grayscott.mp4"

    # Initialize fields - start with the equilibrium state
    U = torch.ones((height, width), dtype=torch.float32, device=device)
    V = torch.zeros((height, width), dtype=torch.float32, device=device)

    # Create more varied, artistic blob seeds with general shapes
    torch.manual_seed(int(time.time()))  # Use current time for true randomness
    num_blobs = 1  # Slightly more blobs for higher resolution

    def create_organic_shape(center_x, center_y, base_size, num_points=8, open_shape=False):
        """Create an organic, irregular shape using polar coordinates"""
        if open_shape:
            # Create an open organic curve
            angles = torch.linspace(0, np.pi * np.random.uniform(0.5, 1.5), num_points, device=device)
            radii = base_size * (0.5 + 0.7 * torch.rand(num_points, device=device))

            # Add some smoothing to make it more organic
            radii_np = radii.cpu().numpy()
            radii_smoothed = gaussian_filter(np.concatenate([radii_np, radii_np, radii_np]), sigma=0.8)[num_points:2*num_points]
            radii = torch.tensor(radii_smoothed, device=device)

            # Create curve points
            curve_x = center_x + radii * torch.cos(angles)
            curve_y = center_y + radii * torch.sin(angles)

            # Create mask from curve points
            mask = torch.zeros((height, width), dtype=torch.float32, device=device)
            thickness = base_size * 0.3

            for px, py in zip(curve_x, curve_y):
                px, py = px.item(), py.item()
                if 0 <= px < width and 0 <= py < height:
                    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
                    dist = torch.sqrt((x - px)**2 + (y - py)**2)
                    mask += torch.exp(-0.5 * (dist / thickness)**2)

            return torch.clamp(mask, 0, 1)
        else:
            # Original closed organic shape
            angles = torch.linspace(0, 2*np.pi, num_points, device=device)
            # Create random radii for each angle to make irregular shapes
            radii = base_size * (0.5 + 0.7 * torch.rand(num_points, device=device))

            # Add some smoothing to make it more organic
            radii_np = radii.cpu().numpy()
            radii_smoothed = gaussian_filter(np.concatenate([radii_np, radii_np, radii_np]), sigma=0.8)[num_points:2*num_points]
            radii = torch.tensor(radii_smoothed, device=device)

            # Create a mask for this shape
            mask = torch.zeros((height, width), dtype=torch.float32, device=device)
            y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')

            # For each pixel, check if it's inside the organic shape
            for i in range(height):
                for j in range(width):
                    dx = j - center_x
                    dy = i - center_y

                    if dx == 0 and dy == 0:
                        mask[i, j] = 1.0
                        continue

                    # Convert to polar coordinates
                    r = torch.sqrt(torch.tensor(dx**2 + dy**2, device=device))
                    theta = torch.atan2(torch.tensor(dy, device=device), torch.tensor(dx, device=device))

                    # Find the interpolated radius at this angle
                    theta_norm = (theta + np.pi) / (2 * np.pi) * num_points
                    idx = int(theta_norm.item()) % num_points
                    next_idx = (idx + 1) % num_points
                    frac = theta_norm.item() - int(theta_norm.item())

                    interp_radius = radii[idx] * (1 - frac) + radii[next_idx] * frac

                    # Create a soft falloff
                    if r <= interp_radius:
                        mask[i, j] = torch.exp(-0.5 * (r / interp_radius)**2)

            return mask

    # Create different types of artistic shapes
    for i in range(num_blobs):
        # Random center for each blob (scaled for higher resolution)
        cx = width / 2
        cy = height / 2

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

        # Random intensity for each blob
        intensity = np.random.uniform(0.6, 1.0)

        # Add to V and subtract from U
        V += intensity * blob
        U -= 0.4 * intensity * blob

    # Clip to valid ranges
    U = torch.clamp(U, 0, 1)
    V = torch.clamp(V, 0, 1)

    # Add small random noise everywhere (scaled for higher resolution)
    U += 0.005 * (torch.rand(height, width, device=device) - 0.5)
    V += 0.005 * (torch.rand(height, width, device=device) - 0.5)

    # Final clipping
    U = torch.clamp(U, 0, 1)
    V = torch.clamp(V, 0, 1)

    print(f"Created {num_blobs} random artistic blob seeds")
    print(f"Initial U range: [{U.min():.4f}, {U.max():.4f}]")
    print(f"Initial V range: [{V.min():.4f}, {V.max():.4f}]")


    # creates block of size BLOCK_SIZE x BLOCK_SIZE

    @triton.jit
    def laplacian_kernel(
        input_ptr, output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        def idx(x, y): return y * N + x
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)

        x_start = pid_x * BLOCK_SIZE
        y_start = pid_y * BLOCK_SIZE

        offs_x = tl.arange(0, BLOCK_SIZE)
        offs_y = tl.arange(0, BLOCK_SIZE)

        x = x_start + offs_x
        y = y_start + offs_y

        mask = (x > 0) & (x < N - 1) & (y > 0) & (y < N - 1)

        center = tl.load(input_ptr + idx(x, y), mask=mask, other=0)
        left   = tl.load(input_ptr + idx(x - 1, y), mask=mask, other=0)
        right  = tl.load(input_ptr + idx(x + 1, y), mask=mask, other=0)
        up     = tl.load(input_ptr + idx(x, y - 1), mask=mask, other=0)
        down   = tl.load(input_ptr + idx(x, y + 1), mask=mask, other=0)

        # Example computation: Laplacian
        out = -4 * center + left + right + up + down

        # Store result
        tl.store(output_ptr + idx(x, y), out, mask=mask)

    def laplacian(Z):
        return (
            -4 * Z +
            torch.roll(Z, 1, dims=0) +
            torch.roll(Z, -1, dims=0) +
            torch.roll(Z, 1, dims=1) +
            torch.roll(Z, -1, dims=1)
        )

    def update(U, V):
        Lu = laplacian(U)
        Lv = laplacian(V)
        reaction = U * V * V
        U += (Du * Lu - reaction + F * (1 - U)) * dt
        V += (Dv * Lv + reaction - (F + k) * V) * dt
        U = torch.clamp(U, 0, 1)
        V = torch.clamp(V, 0, 1)
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

    # Create video in memory
    frames = []

    start = time.time()
    for step in tqdm(range(steps)):
        U, V = update(U, V)

        # Debug: print values every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: U range [{U.min():.4f}, {U.max():.4f}], V range [{V.min():.4f}, {V.max():.4f}]")

        if step % 5 == 0:
            # Use V for visualization - move to CPU for numpy operations
            img = V.cpu().numpy()

            # Check if we have any variation
            v_min, v_max = np.min(img), np.max(img)
            if v_max - v_min > 1e-6:
                img = (img - v_min) / (v_max - v_min)
            else:
                # If no variation, try visualizing U instead
                img = U.cpu().numpy()
                u_min, u_max = np.min(img), np.max(img)
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
            frames.append(img_uint8)

    print(f"Generated {len(frames)} frames")

    # Create video using cv2 and save to bytes
    buffer = BytesIO()

    # Use a temporary file approach since cv2 needs a file path
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(tmp_path, fourcc, video_fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()

        # Read the file back into memory
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()

        os.unlink(tmp_path)  # Clean up temp file

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

    end = time.time()
    print(f"Elapsed Time: {end-start} seconds")
    print("Video generated successfully!")

    return video_bytes

@app.local_entrypoint()
def main():
    """Local entrypoint to run the simulation"""
    print("Running Gray-Scott simulation on Modal...")

    # Run simulation
    video_bytes = run_gray_scott_simulation.remote()

    # Save output locally
    with open("grayscott_modal.mp4", "wb") as f:
        f.write(video_bytes)

    print("Simulation complete! Output saved as grayscott_modal.mp4")

if __name__ == "__main__":
    with app.run():
        main()
