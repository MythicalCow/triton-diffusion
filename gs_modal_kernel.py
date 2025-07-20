import modal
from io import BytesIO
import triton
import triton.language as tl
from triton.testing import do_bench

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
    device = torch.device("cuda")
    print(f"Running Gray-Scott simulation on device: {device}")

    # simulation parameters
    N = 512
    BLOCK_SIZE = 32 # creates block of size BLOCK_SIZE x BLOCK_SIZE
    width, height = N, N
    Du, Dv = 0.16, 0.07
    F, k = 0.012, 0.05
    # F, k = 0.037, 0.0625
    dt = 1.0
    video_fps = 200

    steps = 6000
    video_name = "grayscott.mp4"

    # Initialize fields - start with the equilibrium state
    U = torch.ones((height, width), dtype=torch.float32, device='cuda')
    V = torch.zeros((height, width), dtype=torch.float32, device='cuda')

    images_per_chunk = 10

    U_CHUNK_GPU = torch.zeros((images_per_chunk, height, width), dtype=torch.float32, device='cuda')
    V_CHUNK_GPU = torch.zeros((images_per_chunk, height, width), dtype=torch.float32, device='cuda')

    U_pinned = torch.empty_like(U_CHUNK_GPU, device='cpu', pin_memory=True)
    V_pinned = torch.empty_like(V_CHUNK_GPU, device='cpu', pin_memory=True)

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
        # Random center for each blob
        cx = width / 2
        cy = height / 2

        # Choose random shape type - now includes more open shapes
        shape_type = np.random.choice(['organic', 'organic_open'])

        if shape_type == 'organic':
            # Organic irregular shape (closed)
            base_size = np.random.uniform(30 * (N // 512), 80 * (N // 512))  # Scaled for higher resolution
            blob = create_organic_shape(cx, cy, base_size, open_shape=False)

        elif shape_type == 'organic_open':
            # Organic irregular shape (open)
            base_size = np.random.uniform(30 * (N // 512), 80 * (N // 512))  # Scaled for higher resolution
            blob = create_organic_shape(cx, cy, base_size, open_shape=True)

        # Random intensity for each blob
        intensity = torch.empty(1, device='cuda').uniform_(0.6, 1.0)

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

    @triton.jit
    def laplacian_kernel(
        input_ptr, output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)

        x_start = pid_x * BLOCK_SIZE
        y_start = pid_y * BLOCK_SIZE

        offs_x = tl.arange(0, BLOCK_SIZE)
        offs_y = tl.arange(0, BLOCK_SIZE)

        x = x_start + offs_x[:, None]
        y = y_start + offs_y[None, :]
        mask = (x > 0) & (x < N - 1) & (y > 0) & (y < N - 1)


        center_idx = y * N + x
        left_idx = y * N + (x - 1)
        right_idx = y * N + (x + 1)
        up_idx = (y - 1) * N + x
        down_idx = (y + 1) * N + x

        center = tl.load(input_ptr + center_idx, mask=mask, other=0)
        left   = tl.load(input_ptr + left_idx, mask=mask, other=0)
        right  = tl.load(input_ptr + right_idx, mask=mask, other=0)
        up     = tl.load(input_ptr + up_idx, mask=mask, other=0)
        down   = tl.load(input_ptr + down_idx, mask=mask, other=0)

        # Example computation: Laplacian
        out = -4 * center + left + right + up + down

        # Store result
        tl.store(output_ptr + center_idx, out, mask=mask)

    @triton.jit
    def laplacian_kernel_uv(
        U_ptr, V_ptr, Lu_ptr, Lv_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)

        x_start = pid_x * BLOCK_SIZE
        y_start = pid_y * BLOCK_SIZE

        offs_x = tl.arange(0, BLOCK_SIZE)
        offs_y = tl.arange(0, BLOCK_SIZE)

        x = x_start + offs_x[:, None]
        y = y_start + offs_y[None, :]
        mask = (x > 0) & (x < N - 1) & (y > 0) & (y < N - 1)


        center_idx = y * N + x
        left_idx = y * N + (x - 1)
        right_idx = y * N + (x + 1)
        up_idx = (y - 1) * N + x
        down_idx = (y + 1) * N + x

        #U, V
        U_center, V_center = tl.load(U_ptr + center_idx, mask=mask, other=0), tl.load(V_ptr + center_idx, mask=mask, other=0)
        U_left, V_left = tl.load(U_ptr + left_idx, mask=mask, other=0), tl.load(V_ptr + left_idx, mask=mask, other=0)
        U_right, V_right = tl.load(U_ptr + right_idx, mask=mask, other=0), tl.load(V_ptr + right_idx, mask=mask, other=0)
        U_up, V_up = tl.load(U_ptr + up_idx, mask=mask, other=0), tl.load(V_ptr + up_idx, mask=mask, other=0)
        U_down, V_down = tl.load(U_ptr + down_idx, mask=mask, other=0), tl.load(V_ptr + down_idx, mask=mask, other=0)

        # Compute Laplacians
        Lu_out = -4 * U_center + U_left + U_right + U_up + U_down
        Lv_out = -4 * V_center + V_left + V_right + V_up + V_down

        # Store results
        tl.store(Lu_ptr + center_idx, Lu_out, mask=mask)
        tl.store(Lv_ptr + center_idx, Lv_out, mask=mask)

    def laplacian(Z):
        return (
            -4 * Z +
            torch.roll(Z, 1, dims=0) +
            torch.roll(Z, -1, dims=0) +
            torch.roll(Z, 1, dims=1) +
            torch.roll(Z, -1, dims=1)
        )

    # def laplacian_gpu_caller(Z):
    #     output = torch.zeros_like(Z)
    #     laplacian_kernel[grid](Z, output, N, BLOCK_SIZE)
    #     return output

    @triton.jit
    def gs_update_kernel(
        U_ptr, V_ptr,
        Lu_ptr, Lv_ptr,
        F, k, dt, Du, Dv,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)

        x_start = pid_x * BLOCK_SIZE
        y_start = pid_y * BLOCK_SIZE

        offs_x = tl.arange(0, BLOCK_SIZE)
        offs_y = tl.arange(0, BLOCK_SIZE)

        x = x_start + offs_x[:, None]
        y = y_start + offs_y[None, :]
        mask = (x >= 0) & (x <= N-1) & (y >= 0) & (y <= N-1)

        U = tl.load(U_ptr + y*N + x, mask=mask, other=0)
        V = tl.load(V_ptr + y*N + x, mask=mask, other=0)
        Lu = tl.load(Lu_ptr + y*N + x, mask=mask, other=0)
        Lv = tl.load(Lv_ptr + y*N + x, mask=mask, other=0)
        reaction = U * V * V

        updateU = (Du * Lu - reaction + F * (1 - U)) * dt
        updateV = (Dv * Lv + reaction - (F + k) * V) * dt

        newU = U + updateU
        newV = V + updateV

        # Clip values between 0 and 1
        clippedU = tl.where(newU > 1.0, 1.0, tl.where(newU < 0.0, 0.0, newU))
        clippedV = tl.where(newV > 1.0, 1.0, tl.where(newV < 0.0, 0.0, newV))

        tl.store(U_ptr + y*N + x, clippedU, mask=mask)
        tl.store(V_ptr + y*N + x, clippedV, mask=mask)

    grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # Ceiling division
    grid_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x, grid_y)
    Lu = torch.zeros_like(U)
    Lv = torch.zeros_like(V)
    def update(U, V):
        laplacian_kernel_uv[grid](U, V, Lu, Lv, N, BLOCK_SIZE)
        gs_update_kernel[grid](U, V, Lu, Lv, F, k, dt, Du, Dv, N, BLOCK_SIZE)
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
    samples = np.linspace(0, 1, 256)
    colormap_rgb = custom_cmap(samples)[:, :3]

    #GPU friendly linear interpolation mapper buffer
    torch_colormap = torch.tensor(colormap_rgb, dtype=torch.float32, device="cuda").flatten()

    @triton.jit
    def cyberpunk_colormap_kernel(
        img_ptr, # batch of 10 images of size N * N
        img_rgb_ptr, # batch of 10 rgb output images of size 3 * N * N
        torch_colormap_ptr, # cyberpunk style linear interpolation map
        img_min_ptr, # array of 10 mins
        img_max_ptr, # array of 10 maxs
        N: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        img_index = tl.program_id(1)
        img_offset = img_index * N * N
        rgb_offset = img_index * N * N * 3

        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * N  # flat index

        # Reading the minimums for this array
        img_min = tl.load(img_min_ptr + img_index)
        img_max = tl.load(img_max_ptr + img_index)

        val = tl.load(img_ptr + img_offset + offsets, mask=mask, other=0)
        val = tl.where(img_max - img_min > 1e-6, (val - img_min)/(img_max - img_min), val)
        val = tl.sqrt(val)

        val = val * 1.2 - 0.1
        val = tl.where(val > 1.0, 1.0, tl.where(val < 0.0, 0.0, val))

        idx = (val * 255).to(tl.uint32)
        base_offset = idx * 3
        r = tl.load(torch_colormap_ptr + base_offset + 0, mask=mask) * 255
        g = tl.load(torch_colormap_ptr + base_offset + 1, mask=mask) * 255
        b = tl.load(torch_colormap_ptr + base_offset + 2, mask=mask) * 255

        tl.store(img_rgb_ptr + rgb_offset + offsets * 3 + 0, r.to(tl.uint8), mask=mask)
        tl.store(img_rgb_ptr + rgb_offset + offsets * 3 + 1, g.to(tl.uint8), mask=mask)
        tl.store(img_rgb_ptr + rgb_offset + offsets * 3 + 2, b.to(tl.uint8), mask=mask)

    # Create video in memory
    frames = []

    start = time.time()
    grid_batched = ((N * N + BLOCK_SIZE - 1) // BLOCK_SIZE, images_per_chunk,)
    img_batch = torch.empty(images_per_chunk, N * N, device="cuda")
    img_rgb_batch = torch.empty(images_per_chunk, N * N * 3, dtype=torch.uint8, device="cuda")
    for step in tqdm(range(steps)):
        U, V = update(U, V)

        # Debug: print values every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: U range [{U.min():.4f}, {U.max():.4f}], V range [{V.min():.4f}, {V.max():.4f}]")

        # chunk the gpu outputs
        if step % 5 == 0:
            # calculate within chunk position
            idx = step % (5*images_per_chunk)
            idx = idx // 5
            V_CHUNK_GPU[idx] = V

        # batch coloration of images for rendering (will convert this to gpu)

        if step % (5*images_per_chunk) == 49:
            img_min_tensor = torch.tensor([float(v.min()) for v in V_CHUNK_GPU], device="cuda")
            img_max_tensor = torch.tensor([float(v.max()) for v in V_CHUNK_GPU], device="cuda")
            for i in range(images_per_chunk):
                img_batch[i] = V_CHUNK_GPU[i].flatten()

            cyberpunk_colormap_kernel[grid_batched](img_batch, img_rgb_batch, torch_colormap, img_min_tensor, img_max_tensor, N, BLOCK_SIZE)

            frames.extend([img_rgb_batch[i].view(N, N, 3).cpu().numpy() for i in range(images_per_chunk)])

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
