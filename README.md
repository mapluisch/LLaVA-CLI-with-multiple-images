# LLaVA-CLI-with-multiple-images
LLaVA inference combining multiple images into one for streamlined processing and analysis.

The images are not truncated or re-scaled, but just concatenated vertically with 20px spacing between them. You can adapt the `concatenate_images` func so images are resized, scaled, etc.; this repo should serve as a very basic example.  

You can specify as many images as you want.

## Setup
You should follow the LLaVA tutorial, so that you have the pretrained model / checkpoint shards ready. Then, put my script into your LLaVA directory and start it while in the LLaVA conda-environment (`conda activate llava`).

## Usage 
```
python llava-multi-images.py [ARGS]
```

### Arguments

Given that this project is based on LLaVA's `cli.py`, the following base arguments can be specified:
```
--model-path, default="liuhaotian/llava-v1.5-13b"
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
```

Using `--images /some/img1.jpg /some/img2.jpg /some/img_n.jpg`, you can specify as many images as you want for inference. These input images will get concatenated using PIL.

Using `--save-image`, the resulting concatenated image gets stored in the LLaVA directory as `concat-image.jpg`.


## Disclaimer
This project is a prototype and serves as a basic example of using LLaVA CLI inference with multiple images at once. I have not tested this extensively. Feel free to create a PR.
