# DPM-MedImgEnhance
Pre-trained Diffusion Models for Plug-and-Play Medical Image Enhancement

## Installation

- install `torch=1.3.1` and `torchvision=0.14.1` based on [pytorch guidance](https://pytorch.org/get-started/previous-versions/)
- install [guided diffusion](https://github.com/openai/guided-diffusion)

## Dataset

- [Low dose CT](https://www.aapm.org/grandchallenge/lowdosect/)
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
- [MMs](https://www.ub.edu/mnms/)
- [CMRxMotion](http://cmr.miccai.cloud/)

Resize each 2D slice to `256x256x3` and save it as a `PNG` image.

## Training

- CT model

```bash
MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 3 --num_heads 1"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"

python scripts/image_train.py --data_dir ../NormalDose_png_data_path --log_dir ./work_dir/CT256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

- Heart MR model

```bash
MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 3 --num_heads 1"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"

python scripts/image_train.py --data_dir ../ACDC-MMs_png_data_path --log_dir ./work_dir/MR256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```


## Inference

Download the checkpoints [here](https://drive.google.com/drive/folders/1_7yIdrR3Io8tH-hCkkuFd2PEvtijGEGB?usp=sharing) and put them to `ckpt` folder

- CT model

Run

```bash
python CT_main.py
```

- Heart MR model

Run 

```bash
python MR_main.py
```


## Acknowledgment

We thank the [IDDPM](https://github.com/openai/improved-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), and [DDNM](https://github.com/wyhuai/DDNM) as their implementation served as the basis for our work. We highly appreciate Jiwen Yu, who provided invaluable guidance and support. We also thank the organizers of AAPM Low Dose CT Grand Challenge, ACDC, MMs, and CMRxMothion for making the datasets publicly available.

```
@InProceedings{DPM-MedImgEnhance,
	author="Ma, Jun
	and Zhu, Yuanzhi
	and You, Chenyu
	and Wang, Bo",
	editor="Greenspan, Hayit
	and Madabhushi, Anant
	and Mousavi, Parvin
	and Salcudean, Septimiu
	and Duncan, James
	and Syeda-Mahmood, Tanveer
	and Taylor, Russell",
	title="Pre-trained Diffusion Models for Plug-and-Play Medical Image Enhancement",
	booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
	year="2023",
	publisher="Springer Nature Switzerland",
	address="Cham",
	pages="3--13",
}
```
