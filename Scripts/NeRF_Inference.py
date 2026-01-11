
import math
import imageio
import os
import re
from datetime import datetime

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    # Check file size
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Model saved to {filename}, size: {size_mb:.2f} MB")


def create_gif(image_folder, gif_path, duration=5):
    def sort_key(filename):
        number = re.search(r"(\d+)", filename)
        return int(number.group(1)) if number else 0

    filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')],
                       key=sort_key)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(gif_path, images, duration=duration)

def load_data():
    data = np.load(f"data/lego_200x200.npz")
    images_train = data["images_train"] / 255.0
    c2ws_train = data["c2ws_train"]
    images_val = data["images_val"] / 255.0
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    focal = data["focal"]
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal

def transform(c2w, x_c):
    B, H, W, _ = x_c.shape
    x_c_homogeneous = torch.cat([x_c, torch.ones(B, H, W, 1, device=x_c.device)], dim=-1)

    x_w_homogeneous_reshaped = x_c_homogeneous.view(B, -1, 4)  # [100, 40000, 4]
    x_w_homogeneous_reshaped = x_w_homogeneous_reshaped.permute(0, 2, 1)
    x_w_homogeneous_reshaped = c2w.bmm(x_w_homogeneous_reshaped)
    x_w_homogeneous = x_w_homogeneous_reshaped.permute(0, 2, 1).view(B, H, W, 4)
    x_w = x_w_homogeneous[:, :, :, :3]
    return x_w

def intrinsic_matrix(fx, fy, ox, oy):
    K = torch.tensor([[fx,  0, ox],
                      [ 0, fy, oy],
                      [ 0,  0,  1]])
    return K

def pixel_to_camera(K, uv, s):
    B, H, W, C = uv.shape
    uv_reshaped = uv.view(B, -1, 3).permute(0, 2, 1)
    uv_homogeneous_reshaped = torch.cat([uv_reshaped[:, 1:], torch.ones((B, 1, H*W), device=uv.device)], dim=1)
    K_inv = torch.inverse(K)
    uv_homogeneous_reshaped = torch.stack((uv_homogeneous_reshaped[:, 1], uv_homogeneous_reshaped[:, 0], uv_homogeneous_reshaped[:, 2]), dim=1)
    x_c_homogeneous_reshaped = K_inv.bmm(uv_homogeneous_reshaped)
    x_c_homogeneous = x_c_homogeneous_reshaped.permute(0, 2, 1).view(B, H, W, 3)
    x_c = x_c_homogeneous * s
    
    return x_c

def pixel_to_ray(K, c2w, uv):
    # find x_c
    B, H, W, C = uv.shape 
    x_c = pixel_to_camera(K, uv, torch.ones((B, H, W, 1), device=uv.device))
    
    w2c = torch.inverse(c2w)
    R = w2c[:, :3, :3]
    R_inv = torch.inverse(R)
    T = w2c[:, :3, 3]
    # ray origins
    r_o = -torch.bmm(R_inv, T.unsqueeze(-1)).squeeze(-1)
    
    # ray directions
    x_w = transform(c2w, x_c)
    r_o = r_o.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
    r_d = (x_w - r_o) / torch.norm((x_w - r_o), dim=-1, keepdim=True)
    
    return r_o, r_d

def sample_along_rays(r_o, r_d, perturb=True, near=2.0, far=6.0, n_samples=64):
    t = torch.linspace(near, far, n_samples, device=r_o.device)
    if perturb:
        t = t + torch.rand_like(t) * (far - near) / n_samples
    x = r_o + r_d * t.unsqueeze(-1).unsqueeze(-1)
    return x

class RaysData:
    def __init__(self, images, K, c2w, device='cuda'):
        self.images = images
        self.K = K
        self.c2w = c2w
        self.device = device
        
        self.height = images.shape[1]
        self.width = images.shape[2]
        
        
        self.uv = torch.stack(torch.meshgrid(torch.arange(self.images.shape[0]), torch.arange(self.height), torch.arange(self.width)), dim=-1).to(device).float()
        # add 0.5 offset to each pixel
        self.uv[..., 1] += 0.5
        self.uv[..., 2] += 0.5
        self.uv_flattened = self.uv.reshape(-1, 3)
        
        self.r_o, self.r_d = pixel_to_ray(K, c2w, self.uv)
        self.pixels = images.reshape(-1, 3)
        self.r_o_flattened = self.r_o.reshape(-1, 3)
        self.r_d_flattened = self.r_d.reshape(-1, 3)
        
    def sample_rays(self, batch_size):
        # sample rays
        idx = torch.randint(0, self.pixels.shape[0], (batch_size,), device=self.pixels.device)
        return self.r_o_flattened[idx], self.r_d_flattened[idx], self.pixels[idx]
    
    
    def sample_rays_single_image(self, image_index=None):
        if image_index is None:
            image_index = torch.randint(0, self.c2w.shape[0], (1,), device=self.device).item()
        start_idx = image_index * self.height * self.width
        end_idx = start_idx + self.height * self.width

        r_o_single = self.r_o_flattened[start_idx:end_idx]
        r_d_single = self.r_d_flattened[start_idx:end_idx]
        pixels_single = self.pixels[start_idx:end_idx]
        
        return r_o_single, r_d_single, pixels_single
        
def volrend(sigmas, rgbs, step_size):
    B, N, _ = sigmas.shape
    # transmittance of first ray is 1
    T_i = torch.cat([torch.ones((B, 1, 1), device=rgbs.device), torch.exp(-step_size * torch.cumsum(sigmas, dim=1)[:, :-1])], dim=1)
    alpha = 1 - torch.exp(-sigmas * step_size)
    weights = alpha * T_i
    
    rendered_colors = torch.sum(weights * rgbs, dim=1)# + accumulated_transmittance.squeeze(1) * torch.ones((B, 3), device=rgbs.device)
    return rendered_colors

def volrend_depth(sigmas, step_size):
    depths = torch.linspace(2.0, 6.0, sigmas.shape[1], device=sigmas.device)
    T_i = torch.exp(-step_size * torch.cumsum(sigmas, dim=1))
    T_i = torch.cat([torch.ones((sigmas.shape[0], 1, 1), device=sigmas.device), T_i[:, :-1]], dim=1)
    alpha = 1 - torch.exp(-sigmas * step_size)
    weights = alpha * T_i
    rendered_depths = torch.sum(weights * depths.unsqueeze(0).unsqueeze(-1), dim=1)

    return rendered_depths

def positional_encoding(x, L):
    freqs = 2.0 ** torch.arange(L).float().to(x.device)
    x_input = x.unsqueeze(-1) * freqs * 2 * torch.pi
    encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
    encoding = torch.cat([x, encoding.reshape(*x.shape[:-1], -1)], dim=-1) # add to original input    
    return encoding

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
      
        self.fc1_block1 = nn.Linear(2 * 3 * 10 + 3, 128)
        self.fc2_block1 = nn.Linear(128, 128)
        self.fc3_block1 = nn.Linear(128, 128)
        self.fc4_block1 = nn.Linear(128, 128)
        
       
        self.fc1_d = nn.Linear(2 * 3 * 4 + 3, 128)
        
        
        self.fc1_block2 = nn.Linear(128 + 2 * 3 * 10 + 3, 128)
        self.fc2_block2 = nn.Linear(128, 128)
        self.fc3_block2 = nn.Linear(128, 128)
        self.fc4_block2 = nn.Linear(128, 128)
        
       
        self.linear_density = nn.Linear(128, 1)
        
        
        self.fc1_block3 = nn.Linear(128, 128)
        self.fc2_block3 = nn.Linear(128 + 2 * 3 * 4 + 3, 128)
        
        self.linear_rgb = nn.Linear(128, 3)
        
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, r_d):
        # Positional encoding
       
        x_encoded = positional_encoding(x, L=10) #[64, 10000, 63]
        r_d_encoded = positional_encoding(r_d, L=4) #[10000, 27]
        
        
        x = self.relu(self.fc1_block1(x_encoded))
        x = self.relu(self.fc2_block1(x))
        x = self.relu(self.fc3_block1(x))
        x = self.relu(self.fc4_block1(x))
        
        
        x = torch.cat([x, x_encoded], dim=-1)
        
        x = self.relu(self.fc1_block2(x))
        x = self.relu(self.fc2_block2(x))
        x = self.relu(self.fc3_block2(x))
        x = self.fc4_block2(x)
        
        
        density = self.relu(self.linear_density(x))
        
        # Process ray direction
        x = self.fc1_block3(x)
        
        x = torch.cat([x, r_d_encoded], dim=-1)
        # Process after concatenation
        x = self.relu(self.fc2_block3(x))
        rgb = self.linear_rgb(x)
        rgb = self.sigmoid(rgb)
        #print(f"x shape: {x.shape}, r_d shape: {r_d.shape}")
        return rgb, density



def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()

def inference(model, test_dataset, chunk_size=5000):
    model.eval()
    num_images = test_dataset.c2w.shape[0]
    latencies = []
    throughputs = []
    total_time = 0.0

    with torch.no_grad():
        for i in range(num_images):
            start_time = time.time()

            rays_o, rays_d, _ = test_dataset.sample_rays_single_image()

            comp_rgb = []
            for chunk_start in range(0, rays_o.shape[0], chunk_size):
                rays_o_chunk = rays_o[chunk_start:chunk_start + chunk_size]
                rays_d_chunk = rays_d[chunk_start:chunk_start + chunk_size]

                points = sample_along_rays(rays_o_chunk, rays_d_chunk)
                points = points.permute(1, 0, 2)
                rays_d_chunk = rays_d_chunk.unsqueeze(1).repeat(1, points.shape[1], 1)
                rgb, sigmas = model_trt(points, rays_d_chunk)

                comp_rgb_chunk = volrend(sigmas, rgb, step_size=(6.0 - 2.0) / 64)
                comp_rgb.append(comp_rgb_chunk)
                
            comp_rgb = torch.cat(comp_rgb, dim=0)
            image = comp_rgb.reshape(200, 200, 3).cpu().numpy()
            plt.imsave(f"final_render/render_{i}.jpg", image)

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            latencies.append(elapsed_time)
            throughputs.append(1.0 / elapsed_time)

            print(f"Image {i} inference time: {elapsed_time:.2f} seconds")

    create_gif('final_render', 'final_render/inference.gif')
    plot_metrics(latencies, throughputs, total_time, output_dir='graphs')



def plot_metrics(latencies, throughputs, total_time, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot latency
    plt.figure()
    plt.plot(range(len(latencies)), latencies, marker='o', linestyle='-', color='b')
    plt.title('Latency per Image')
    plt.xlabel('Image Index')
    plt.ylabel('Latency (seconds)')
    plt.savefig(os.path.join(output_dir, 'latency_per_image.png'))
    plt.close()

    # Plot throughput
    plt.figure()
    plt.plot(range(len(throughputs)), throughputs, marker='o', linestyle='-', color='r')
    plt.title('Throughput per Second')
    plt.xlabel('Image Index')
    plt.ylabel('Throughput (images/second)')
    plt.savefig(os.path.join(output_dir, 'throughput_per_sec.png'))
    plt.close()

    # Plot total execution time
    plt.figure()
    plt.bar(['Total Execution Time'], [total_time], color='g')
    plt.title('Total Execution Time')
    plt.ylabel('Time (seconds)')
    plt.savefig(os.path.join(output_dir, 'total_execution_time.png'))
    plt.close()

    print(f"Graphs saved to {output_dir}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    # Load data
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_data()

    # Prepare data
    images_train = torch.tensor(images_train).float().to(device)
    c2ws_train = torch.tensor(c2ws_train).float().to(device)
    images_val = torch.tensor(images_val).float().to(device)
    c2ws_val = torch.tensor(c2ws_val).float().to(device)
    c2ws_test = torch.tensor(c2ws_test).float().to(device)
    focal = torch.tensor(focal).float().to(device)
    K_test = intrinsic_matrix(focal.item(), focal.item(), images_val.shape[1] / 2, images_val.shape[2] / 2).unsqueeze(0).repeat(c2ws_test.shape[0], 1, 1).to(device)
    
    batch_size = 1  
    n_points = 64   
    n_features_x = 3  
    n_features_r_d = 4 
    import torch
    from torch2trt import torch2trt
    # Load model
    model = MLP().to(device)
    print(f"x shape: {x.shape}, r_d shape: {r_d.shape}")
    checkpoint_filename = "checkpoints/nerf_checkpoint_20240928_114649.pt"  
    load_model(model, checkpoint_filename)

 
    dummy_x = torch.randn(batch_size, n_points, n_features_x).to(device)
    dummy_r_d = torch.randn(batch_size, n_points, n_features_r_d).to(device)
    
    model_trt = torch2trt(model, [dummy_x, dummy_r_d], fp16=True) 

    test_dataset = RaysData(images_train[:60], K_test, c2ws_test)

    inference(model_trt, test_dataset)  

    end_time=time.time()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.2f} seconds")
