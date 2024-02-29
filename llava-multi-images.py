import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

import requests
from PIL import Image
from io import BytesIO
from math import ceil, sqrt
from transformers import TextStreamer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def expand_image_range_paths(paths):
    expanded_paths = []
    # check if specified --images is range of imgs
    for path in paths:
        if "{" in path and "}" in path:
            pre, post = path.split("{", 1)
            range_part, post = post.split("}", 1)
            start, end = map(int, range_part.split("-"))

            for i in range(start, end + 1):
                expanded_paths.append(f"{pre}{i}{post}")
        else:
            expanded_paths.append(path)

    return expanded_paths


def parse_resolution(resolution_str):
    # try to parse a string into a resolution tuple for the grid output
    try:
        width, height = map(int, resolution_str.split(','))
        return width, height
    except Exception as e:
        raise argparse.ArgumentTypeError("Resolution must be w,h.") from e


def concatenate_images_vertical(images, dist_images):
    # calc max width from imgs
    width = max(img.width for img in images)
    # calc total height of imgs + dist between them
    total_height = sum(img.height for img in images) + dist_images * (len(images) - 1)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    # init var to track current height pos
    current_height = 0
    for img in images:
        # paste img in new_img at current height
        new_img.paste(img, (0, current_height))
        # update current height for next img
        current_height += img.height + dist_images

    return new_img


def concatenate_images_horizontal(images, dist_images):
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images

    return new_img


def concatenate_images_grid(images, dist_images, output_size):
    num_images = len(images)
    # calc grid size based on amount of input imgs
    grid_size = max(2, ceil(sqrt(num_images)))

    cell_width = (output_size[0] - dist_images * (grid_size - 1)) // grid_size
    cell_height = (output_size[1] - dist_images * (grid_size - 1)) // grid_size

    # create new img with output_size, black bg
    new_img = Image.new('RGB', output_size, (0, 0, 0))

    for index, img in enumerate(images):
        # calc img aspect ratio
        img_ratio = img.width / img.height
        # calc target aspect ratio per cell
        target_ratio = cell_width / cell_height

        # resize img to fit in cell
        if img_ratio > target_ratio:
            new_width = cell_width
            new_height = int(cell_width / img_ratio)
        else:
            new_width = int(cell_height * img_ratio)
            new_height = cell_height

        # resize img using lanczos filter
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        row = index // grid_size
        col = index % grid_size

        # calc x, y offsets for img positioning
        x_offset = col * (cell_width + dist_images) + (cell_width - new_width) // 2
        y_offset = row * (cell_height + dist_images) + (cell_height - new_height) // 2

        # paste resized img in calc pos
        new_img.paste(resized_img, (x_offset, y_offset))

    return new_img


def concatenate_images(images, strategy, dist_images, grid_resolution):
    if strategy == 'vertical':
        return concatenate_images_vertical(images, dist_images)
    elif strategy == 'horizontal':
        return concatenate_images_horizontal(images, dist_images)
    elif strategy == 'grid':
        return concatenate_images_grid(images, dist_images, grid_resolution)
    else:
        raise ValueError("Invalid concatenation strategy specified")


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    args.images = expand_image_range_paths(args.images)
    images = [load_image(img_file) for img_file in args.images]
    image = concatenate_images(images, args.concat_strategy, args.dist_images, args.grid_resolution) if len(images) > 1 else images[0]
    image_size = image.size

    if args.save_image:
        image.save("concat-image.jpg")

    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--images", type=str, nargs='+', required=True,
                    help="Specify the paths for images to be concatenated. Accepts multiple paths, or range of images in the same location, e.g. img{1-4}.jpg.")

    parser.add_argument("--save-image", action="store_true",
                    help="If used, stores the resulting concatenated image in the LLaVA directory as 'concat-image.jpg'.")

    parser.add_argument("--concat-strategy", type=str, default="vertical", choices=["vertical", "horizontal", "grid"],
                    help="Determines the arrangement strategy for image concatenation. Options: 'vertical', 'horizontal', 'grid'.")
    
    parser.add_argument("--dist-images", type=int, default=20,
                    help="Sets the spacing (in pixels) between concatenated images.")
    
    parser.add_argument("--grid-resolution", type=parse_resolution, default='2560,1440', 
                        help="Fixed resolution of the resulting grid image. Specify as width, height. Default is 2560,1440.")

    args = parser.parse_args()
    main(args)
