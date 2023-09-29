"""
adapted from DDNM. Please also cite this paper.
https://github.com/wyhuai/DDNM
"""
import argparse
import yaml
import os
join = os.path.join
from skimage import io
import numpy as np
import tqdm
import torch
from functions import data_transform, inverse_data_transform
from functions.svd_ddnm import ddnm_diffusion
import torchvision.utils as tvu
from guided_diffusion.script_util import create_model, model_and_diffusion_defaults


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default='config.yml', required=False, help="Path to the config file"
    )
    parser.add_argument(
        "--deg", type=str, default='sr_averagepooling', required=False, help="Degradation: sr_averagepooling, denoising"
    )
    parser.add_argument("--deg_scale", type=float, default=4, help="deg_scale")    
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default='./example/MR', 
        required=False,
        help="Path of the test dataset.",
    )    
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="./results",
        help="The folder name of samples",
    )    
    parser.add_argument('--model_path', type=str, default='./ckpt/MR.pt', help='Path to the model')

    parser.add_argument("--seed", type=int, default=2023, help="Set different seeds for diverse results")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta")      
    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    args.output_folder = join(args.output_folder, 'MR')
    os.makedirs(args.output_folder, exist_ok=True)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


args, config = parse_args_and_config()

ddpm_ori = model_and_diffusion_defaults()
defaults = dict(image_size=256,
        batch_size=1,
        num_channels=64,
        num_res_blocks=3,
        num_heads=1,
        diffusion_steps=1000,
        noise_schedule='linear',
        lr=1e-4,
        clip_denoised=False,
        num_samples=1, # 10000
        use_ddim=True,
        model_path="",
    )
ddpm_ori.update(defaults)

model = create_model(**ddpm_ori)
ddpm_ori = dict2namespace(ddpm_ori)
if config.model.use_fp16:
    model.convert_to_fp16()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)

betas = np.linspace(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.num_diffusion_timesteps, dtype=np.float64)
betas = torch.from_numpy(betas).float().to(device)

img_names = sorted(os.listdir(args.input_folder))
print(f'Dataset img num: {len(img_names)}')

# get degradation matrix
deg = args.deg
A_funcs = None
if deg == 'denoising':
    from functions.svd_operators import Denoising
    A_funcs = Denoising(config.data.channels, ddpm_ori.image_size, device)
elif deg == 'sr_averagepooling':
    blur_by = int(args.deg_scale)
    from functions.svd_operators import SuperResolution
    A_funcs = SuperResolution(config.data.channels, ddpm_ori.image_size, blur_by, device)
else:
    raise ValueError("degradation type not supported")



for img_name in tqdm.tqdm(img_names):
    img_np = io.imread(join(args.input_folder, img_name))
    img_np = img_np/np.max(img_np)
    x_orig = torch.from_numpy(img_np.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
    x_orig = data_transform(config, x_orig)

    y = A_funcs.A(x_orig)
    Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, ddpm_ori.image_size, ddpm_ori.image_size)
    for i in range(len(Apy)):
        tvu.save_image(
            inverse_data_transform(config, Apy[i]),
            os.path.join(args.output_folder, 'baseline_' + img_name)
        )
        tvu.save_image(
            inverse_data_transform(config, x_orig[i]),
            os.path.join(args.output_folder, 'original_' + img_name)
        )

    model.eval()
    with torch.no_grad():
        torch.manual_seed(args.seed)
        x = torch.randn(y.shape[0], config.data.channels, ddpm_ori.image_size, ddpm_ori.image_size, device=device)
        assert x.shape == (1,3,256,256), "x shape is not correct" + x.shape
        x, _ = ddnm_diffusion(x, model, betas, args.eta, A_funcs, y, cls_fn=None, classes=None, config=config)

    x = [inverse_data_transform(config, xi) for xi in x]
    tvu.save_image(x[0][0], os.path.join(args.output_folder, 'output_' + img_name))
