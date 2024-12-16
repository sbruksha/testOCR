import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from pprint import pprint


def print_hi(name):
    print("GPU available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    DEVICE = "cuda:0"

    myprompts = ["Describe the image : <image>"]
    myimages  = [Image.open("Baby.jpeg") ]

    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base", do_image_splitting=False, )
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b-base",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    inputs = processor(text=myprompts, images=myimages, padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    pprint(generated_texts)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
