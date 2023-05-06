import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch.optim import Adam
from infrastructure.utils import *
from nn import Unet
from infrastructure.forward_diffusion import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset

# Load dataset and set hyperparameters
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128
epochs = 10

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 800

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data pre-processing pipeline
def process_data(input_data):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    input_data["pixel_values"] = [transform(image.convert("L")) for image in input_data["image"]]
    del input_data["image"]
    return input_data

transformed_dataset = dataset.with_transform(process_data).remove_columns("label")
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

# Initialize new model
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

# Loss function using huber loss
def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    loss = F.smooth_l1_loss(noise, predicted_noise)
    return loss

# Training loop
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        # Sample a random time step for each example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Calculate loss and backpropagate    
        loss = p_losses(model, batch, t)
        if step % 100 == 0:
            print("Loss:", loss.item())
        
        loss.backward()
        optimizer.step()

# Define sampling functions
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 – Use our model to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 (Line 4)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


# Sample from model and save a random image
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)
random_index = 3
selected_image = samples[-1][random_index]
selected_image_tensor = torch.tensor(selected_image).unsqueeze(0)
save_image(selected_image_tensor, str(results_folder / f'sample.png'))

