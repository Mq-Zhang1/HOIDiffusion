import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import random
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import csv
from tqdm import tqdm


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

templates = [
    "Fill in the blanks: a hand is grasping a [color] {} [background]",
    'Simply describe: a hand is grasping a what {} with what environement',
    'Fill in the blanks: there is a hand in the image holding a [color] {} [background]',
    'Simply describe: there is a hand holding a what {} with what environement',
    "Fill in the blanks: a hand is grasping a [texture] {} [background]",
    'Simply describe: a hand is grasping a what texture {} with what environement',
    'Simply describe: a hand is grasping a what texture what color {} with what environement',
]

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode #True:'llava-v1' 

    img_paths = []
    with open(args.image_file, 'r', encoding='utf-8') as fp:
        data = csv.DictReader(fp)
        for file in data:
            if file["image"] == "image": continue
            image_path = file["image"]
            img_paths.append({'rgb': image_path, \
                              "shape":{"top":int(file["top"]),"bottom":int(file["bottom"]),"left":int(file["left"]),"right":int(file["right"])}, "sentence":file["sentence"]})
    
            
    # start read images
    print(f"Processing {len(img_paths)} images")
    gen_prompt = []
    for i in tqdm(range(0,len(img_paths))):
        batch_image = []
        batch_text = []
        for j in range(args.batch_size):
            entry = img_paths[i+j]
            print(f"Processing image {entry['rgb']}")
            image = load_image(entry["rgb"]) # PIL image
            image = image.crop((entry["shape"]["left"],entry["shape"]["top"],entry["shape"]["right"],entry["shape"]["bottom"]))
            batch_image.append(image)

            object = entry["sentence"].split(" grasping a ")[-1]
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles
            # Input prompt are the same for all the images
            inp = random.choice(templates).format(object)
            text = DEFAULT_IMAGE_TOKEN + '\n' + inp #'<image>\nDescribe the image'
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_txt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() #[bs,47]
            batch_text.append(input_txt_ids)

        # Similar operation in model_worker.py
        image_tensor = process_images(batch_image, image_processor, args) # (len([]),3,336,336)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16) #True
        
        # Start generating, the input prompt are the same for all the images
        input_ids = torch.concat(batch_text, dim=0)
        assert input_ids.shape[0] == image_tensor.shape[0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2 #'</s>'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = None
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(outputs)
        gen_prompt.append([entry["rgb"], outputs])
        
        if (i)%500 == 0:
            with open(args.output_path, "a", newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(gen_prompt)
                outfile.close()
            gen_prompt = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
