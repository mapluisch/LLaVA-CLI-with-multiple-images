# LLaVA-CLI-with-multiple-images
LLaVA inference combining multiple images into one for streamlined processing and cross-image analysis.

## Examples

### Vertical Placement

For these two input images 

<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/66bda775-156b-404b-9171-41fa21e3c0c6" height="250" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/51719905-0265-4333-8e5a-798598f7f79b" height="250">   
</p>

calling my script with these arguments
```
python3 llava-multi-images.py --load-4bit --save-image --images test-images/img1.jpg test-images/img2.jpg --dist-images 250 --concat-strategy vertical
```

LLaVA receives this concatenated image
<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/cdd3573f-b3e1-4e87-af8c-4266583691a4" width="500">
</p>

and can answer questions about it, understanding that they are separate images:


> USER: Analyze and explain which image is the most visually intriguing.

> ASSISTANT: Both images are visually striking, but they present different types of scenes and evoke different emotions.
> 
> The top image captures a dramatic and serene landscape. It features a snow-covered mountain under a sky with the Aurora Borealis, also known as the Northern Lights. The aurora creates a vibrant, almost otherworldly glow that contrasts with the cool, white snow on the mountain. This image evokes a sense of tranquility and awe at the natural beauty of the scene.
> 
> The bottom image shows a solitary tree in a field with dry grass. The tree stands out against the flat landscape, and the dry grass suggests a season like autumn or early spring. This image conveys a sense of solitude and the quiet beauty of nature.
> 
> Both images are visually intriguing, but they appeal to different aspects of the viewer's experience. The top image is more dramatic and otherworldly, while the bottom image is more grounded and evokes a sense of solitude. The choice of which image is more visually intriguing would depend on personal preference and the emotions one wishes to evoke.


### Horizontal Placement

Given these two input images

<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/04b32564-78f1-4cbe-85cd-9eb113968e72" height="250" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/26a852e2-583c-47ca-9d97-7ccac7e2596e" height="250">
</p>

and these arguments
```
python3 llava-multi-images.py --load-4bit --save-image --images test-images/img3.jpg test-images/img4.jpg --dist-images 250 --concat-strategy horizontal
```

LLaVA receives this image
<p align="center">!
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/01873626-9b5a-40b0-90b8-2eb96160f5d6" width="500">
</p>

and can once again answer promptly:

> USER: Describe the main parts of each image.

> ASSISTANT: The image on the left shows a building with a sign that reads "HOTEL" prominently displayed. The building has a distinctive architectural style with a tower-like structure and a series of windows. The sky is overcast, and the lighting suggests it could be early morning or late afternoon.
> 
> The image on the right depicts a road lined with trees on both sides, creating a tunnel-like effect. The road appears to be a highway or a major road, given the presence of a guardrail and the absence of any buildings or structures close to the road. The trees are lush and green, indicating that the photo was likely taken during a time of year when trees are in full leaf. The sky is clear and blue, suggesting a bright and sunny day.


### Grid Placement

Finally, given four input images
<p align="center">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/6d53c532-6f57-4bea-b421-a31e9f8095b2" height="150" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/443bfb72-b230-4d19-82ef-e74b2c4bd84a" height="150" style="margin-right: 10px;">
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/2b3902cb-684f-4f67-b196-718223de85c2" height="150" style="margin-right: 10px;">  
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/eb9afab7-8b09-4192-83f4-4c784438c843" height="150">
</p>

and these arguments
```
python3 llava-multi-images.py --load-4bit --save-image --images test-images/img5.jpg test-images/img6.jpg test-images/img7.jpg test-images/img8.jpg --dist-images 100 --concat-strategy grid
```

LLaVA receives this concatenated image

<p align="center">!
  <img src="https://github.com/mapluisch/LLaVA-CLI-with-multiple-images/assets/31780571/69ee07eb-722a-46d1-a598-10a7fabe9cfb" width="500">
</p>

and can tell us which cars are present:

> USER: Describe the cars that you can see.

> ASSISTANT: The image is a collage of four photographs featuring different cars. In the top left photo, there is a silver Mercedes-Benz car with a distinctive grille and headlights. The car is parked on a road with greenery in the background. In the top right photo, there is a blue BMW car with a sporty design, also parked on a road. The car has a prominent grille and alloy wheels.
> 
> The bottom left photo shows a red sports car in motion, with smoke coming from its rear tires, indicating high speed. The car has a sleek design with a low and wide stance. In the bottom right photo, there is a silver Audi car with a distinctive front grille and LED headlights. The car is parked indoors, possibly in a showroom, with other cars visible in the background.
> 
> Each car is captured in different settings, highlighting their design and features. The collage provides a visual comparison of the cars, showcasing their brand identity and design language.

## Setup
You should follow the LLaVA tutorial, so that you have the pretrained model / checkpoint shards ready. Then, put my script into your LLaVA directory and start it while in the LLaVA conda-environment (`conda activate llava`).

## Usage 
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
--concate-strategy, default="vertical", choices=["vertical", "horizontal", "grid"]
--dist-images, default=20
```

Using `--images /some/img1.jpg /some/img2.jpg /some/img_n.jpg`, you can specify as many images as you want for inference. These input images will get concatenated using PIL.

Using `--save-image`, the resulting concatenated image gets stored in the LLaVA directory as `concat-image.jpg`.

Using `--concat-strategy`, you can specify the arrangement of the concatenated images (see [Examples](##Examples))

Using `--dist-images`, you can specify the spacing between images.

## Disclaimer
This project is a prototype and serves as a basic example of using LLaVA CLI inference with multiple images at once. I have not tested this extensively - I've tried both LLaVA v1.5 and v1.6 13b with 4-bit quantization. Feel free to create a PR.
