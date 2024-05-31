# Prompt Generation

We use LLaVA to refine the prompts. However, the codebase is updated since our last usage, some changes may be required. A script is provided here but please refer to the original repo for details.

1. Download the [code](https://github.com/haotian-liu/LLaVA) and prepare the environment.
2. Copy the `prompt.py` into `llava/serve`.
3. Run following cmd to process the prompts:
    ```bash
    python -m llava.serve.prompt 
           --model-path liuhaotian/llava-v1.5-7b \
           --image-file <input_image_csv_file> \
           --load-4bit --batch-size 1 \
           --output-path <output_txt_path>
    ```
