# Florence2 Caption Batch
This tool uses the [VLM Florence2](https://huggingface.co/microsoft/Florence-2-large) from Microsoft to caption images in an input folder. Thanks to their team for training this great model.

It's a very fast and fairly robust captioning model that can produce good outputs in 3 different levels of detail.

## Requirements
* Python 3.10 or above.
  * It's been tested with 3.10, 3.11 and 3.12.
  * It does not work with 3.8.

* Cuda 12.1.
  * It may work with other versions. Untested.
 
To use CUDA / GPU speed captioning, you'll need ~6GB VRAM or more.

## Setup
1. Create a virtual environment. Use the included `venv_create.bat` to automatically create it. Use python 3.10 or above.
2. Install the libraries in requirements.txt. `pip install -r requirements.txt`. This is done by step 1 when asked if you use `venv_create`.
3. Install [Pytorch for your version of CUDA](https://pytorch.org/). It's only been tested with version 12.1 but may work with others.
4. Open `batch.py` in a text editor and change the BATCH_SIZE = 7 value to match the level of your GPU.

>   For a 6gb VRAM GPU, use 1.
  
>   For a 24gb VRAM GPU, use 7.

## How to use
1. Activate the virtual environment. If you installed with `venv_create.bat`, you can run `venv_activate.bat`.
2. Run `python batch.py` from the virtual environment.

This runs captioning on all images in the /input/-folder.

## Detail Mode
You can edit the variable `DETAIL_MODE` to 1, 2 or 3.

Here's an example:

![airplane 001](https://github.com/user-attachments/assets/61219c96-5ed1-4bb6-acee-17ddef62fe52)

DETAIL_MODE = 1:
```
A toy airplane flying through the clouds in the sky.
```

DETAIL_MODE = 2:
```
The image shows a toy airplane flying through the sky with white fluffy clouds in the background.
```


DETAIL_MODE = 3:
```
The image shows a toy airplane flying above the clouds. The airplane is made of gray yarn and has two propellers on either side. It appears to be in mid-flight, with its wings spread wide and its nose pointing upwards. The clouds below are white and fluffy, and the sky is a light blue with a few wispy clouds. In the background, there is a body of water visible. The overall mood of the image is peaceful and serene.
```
