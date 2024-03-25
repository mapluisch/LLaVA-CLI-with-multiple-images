<h1 align="center">LLaVA CLI with multiple images</h1>
<div align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/88cd732a-fe3f-4ba4-94a5-3935371c5d34" alt="banner" style="width:50%">
  <p><em>LLaVA inference combining multiple images into one for streamlined processing and cross-image analysis.</em></p>
</div>


## Setup
0. You should follow the LLaVA tutorial, so that you have the pretrained model / checkpoint shards ready.
1. Then, `cd` into your LLaVA root directory.
2. Clone my repo (and optionally remove the test-images):
```
git clone https://github.com/mapluisch/LLaVA-CLI-with-multiple-images.git && \
(cd LLaVA-CLI-with-multiple-images && \
rm -rf test-images && \
cp -a . ../) && \
rm -rf LLaVA-CLI-with-multiple-images
```

This command simply clones the repo, removes the test-images folder, copies all the files into the actual working directory (your LLaVA root directory), and finally removes the repo's directory.

## Usage
While in your LLaVA directory, first activate the conda environment via `conda activate llava`.
Then, simply call my script via `python` or `python3` with your preferred arguments.
```
python llava-multi-images.py [ARGS]
```

### Arguments

Given that this project is based on LLaVA's `cli.py`, the following base arguments can be specified:
```
--model-path, default="liuhaotian/llava-v1.6-vicuna-13b"
--model-base, default=None
--device, default="cuda"
--conv-mode, default=None
--temperature, default=0.2
--max-new-tokens, default=512
--load-8bit, action="store_true"
--load-4bit, action="store_true"
--debug, action="store_true"
```

Additionally added args:
```
--images
--save-image, action="store_true"
--concat-strategy, default="vertical", choices=["vertical", "horizontal", "grid"]
--dist-images, default=20
--grid-resolution, default=2560,1440
```

Using `--images /some/img1.jpg /some/img2.jpg /some/img_n.jpg`, or alternatively `--images /some/img{1-n}.jpg` if the images have the same location and prefix, you can specify as many images as you want for inference. These input images will get concatenated using PIL.

Using `--save-image`, the resulting concatenated image gets stored in the LLaVA directory as `concat-image.jpg`.

Using `--concat-strategy`, you can specify the arrangement of the concatenated images (see [Examples](##Examples)).

Using `--dist-images`, you can specify the spacing between images.

Using `--grid-resolution`, you can specify the output image's resolution when using grid placement.

## Examples

### Vertical Placement

#### Input
For these two input images, `img1.jpg` & `img2.jpg`:

<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/66bda775-156b-404b-9171-41fa21e3c0c6" height="250" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/51719905-0265-4333-8e5a-798598f7f79b" height="250">   
</p>

#### Image Concatenation
```
python3 llava-multi-images.py --load-4bit --save-image --images test-images/img1.jpg test-images/img2.jpg --dist-images 250 --concat-strategy vertical
```

results in this concatenated image being used for LLaVA's CLI conversation (empty space in the lower right due to resolution mismatch):
<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/cdd3573f-b3e1-4e87-af8c-4266583691a4" width="250">
</p>

#### LLaVA Response
LLaVA can now answer questions about the merged image, understanding that it consists of separate images:

> USER: Analyze and explain which image is the most visually intriguing.

> ASSISTANT: Both images are visually striking, but they present different types of scenes and evoke different emotions.
> 
> The top image captures a dramatic and serene landscape. It features a snow-covered mountain under a sky with the Aurora Borealis, also known as the Northern Lights. The aurora creates a vibrant, almost otherworldly glow that contrasts with the cool, white snow on the mountain. This image evokes a sense of tranquility and awe at the natural beauty of the scene.
> 
> The bottom image shows a solitary tree in a field with dry grass. The tree stands out against the flat landscape, and the dry grass suggests a season like autumn or early spring. This image conveys a sense of solitude and the quiet beauty of nature.
> 
> Both images are visually intriguing, but they appeal to different aspects of the viewer's experience. The top image is more dramatic and otherworldly, while the bottom image is more grounded and evokes a sense of solitude. The choice of which image is more visually intriguing would depend on personal preference and the emotions one wishes to evoke.


### Horizontal Placement

#### Input
Given these two input images, `img3.jpg` & `img4.jpg`:

<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/04b32564-78f1-4cbe-85cd-9eb113968e72" height="250" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/26a852e2-583c-47ca-9d97-7ccac7e2596e" height="250">
</p>


#### Image Concatenation
```
python3 llava-multi-images.py --load-4bit --save-image --images test-images/img3.jpg test-images/img4.jpg --dist-images 250 --concat-strategy horizontal
```

LLaVA receives this image (again, empty space in the lower right due to resolution mismatch):
<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/01873626-9b5a-40b0-90b8-2eb96160f5d6" width="500">
</p>

#### LLaVA Response
...and can once again answer promptly:

> USER: Describe the main parts of each image.

> ASSISTANT: The image on the left shows a building with a sign that reads "HOTEL" prominently displayed. The building has a distinctive architectural style with a tower-like structure and a series of windows. The sky is overcast, and the lighting suggests it could be early morning or late afternoon.
> 
> The image on the right depicts a road lined with trees on both sides, creating a tunnel-like effect. The road appears to be a highway or a major road, given the presence of a guardrail and the absence of any buildings or structures close to the road. The trees are lush and green, indicating that the photo was likely taken during a time of year when trees are in full leaf. The sky is clear and blue, suggesting a bright and sunny day.


### Grid Placement
#### Input
Finally, given e.g. nine input images, in this case `img{5-13}.jpg`:

<p align="center" style="display: flex; flex-wrap: nowrap; justify-content: center; align-items: center;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/e0bb537d-fb00-4a78-b60d-08ad9dbb412f" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/f908be2d-c57c-4894-acfa-49883575b9db" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/c87e66af-6b90-4225-91ec-e8646be80ec4" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/9ffd6d1f-e1e4-4f2a-8c9c-f7a47c4ccd99" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/8775d67d-9de6-4012-a121-34410d884f09" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/4763442f-57df-4ea8-86d4-ea3c4e1c140a" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/3e394f62-4827-4069-912e-39d5b84d05e1" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/e70c11db-41c1-4cc3-bc99-27b155490f29" height="100" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/6eb64fb0-8d28-4c11-afb1-2529ea20002b" height="100">
</p>

#### Image Concatenation
```
python3 llava-multi-images.py --load-4bit --save-image --images test-images/img{5-13}.jpg --dist-images 150 --concat-strategy grid --temperature 0.7
```

LLaVA receives this new image with a size of `--grid-resolution`, which defaults to 2560x1440, consisting of a grid of images:
<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/53a0de86-f777-431b-ae9d-f8b0830daf7d" width="500">
</p>

#### LLaVA Response
Again, you can now pose questions to LLaVA, e.g., whether it can tell us a short rhyme about the different cars:

> USER: Write a rhyme about how many different cars you can see.

> ASSISTANT: In this collection, you'll see quite a few,<br>
> A variety of wheels, with styles to pursue.<br>
> Nine in total, each unique and grand,<br>
> Ranging from sports to luxury and beyond.<br>
>
> Some are sleek, some are chunky, some are fast,<br>
> All with designs that'll leave you amazed.<br>
> Brand names that shine like stars at night,<br>
> Mercedes, Audi, and BMW, they all take flight.<br>
>
> They're parked and zooming, on dirt and on tar,<br>
> Each model a masterpiece, a work of art. <br>
> From city streets to mountain paths, <br>
> These cars are ready to conquer every stretch. <br>
>
> So here's to the beauty and the power, <br>
> To the speed and style, and every hour. <br>
> For each model, a story, each one's a dream, <br>
> In this collage, cars are the stars of the scene.

## Disclaimer
This project is a prototype and serves as a basic example of using LLaVA CLI inference with multiple images at once. <b>I have not tested this extensively</b> - I've tried both LLaVA v1.5 and v1.6 13b with 4-bit quantization. Results may and probably will vary depending on the model and quantization you choose. Feel free to create a PR.
