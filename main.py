import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import MllamaForConditionalGeneration, AutoProcessor

from PIL import Image
from pprint import pprint


def print_hi(name):
    print("GPU available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    DEVICE = "cuda:0"

    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open("example_screenshot.png")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract all text from that image"}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    print(processor.decode(output[0]))
    # myprompts = ["Describe the image : <image>"]
    # myimages  = [Image.open("Baby.jpeg") ]
    #
    # processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base", do_image_splitting=False, )
    # model = AutoModelForVision2Seq.from_pretrained(
    #     "HuggingFaceM4/idefics2-8b-base",
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    #
    # inputs = processor(text=myprompts, images=myimages, padding=True, return_tensors="pt")
    # inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    # # Generate
    # generated_ids = model.generate(**inputs, max_new_tokens=100)
    # generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # pprint(generated_texts)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
