import diffusers
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image, make_image_grid
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
from copy import deepcopy
import os
from pathlib import Path

from diffusers.image_processor import IPAdapterMaskProcessor

import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL, StableDiffusionXLControlNetPipeline

import numpy as np