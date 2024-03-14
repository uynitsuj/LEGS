import base64
import requests
import os 
import json
from PIL import Image
import io
import argparse 
import json 
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import csv 
from tqdm import tqdm 

# with tactile 
# Images: The first image is from a camera observing the tactile sensor (shiny, near the top of the image) and the surface. The second image is a cropped version of the first image that focuses on the contact patch. The third image is from the tactile sensor showing the tactile feedback.\n

# only vision
# prompt = """
# Surface Type: [Specify the surface type, e.g., "metal," "fabric"]
# Images: The first image is from a camera observing the tactile sensor (shiny, near the top of the image) and the surface. The second image is a cropped version of the first image that focuses on the contact patch. 
# Example: For a smooth and cold surface, the description might be "slick, chilly, hard, unyielding, glossy."
# Task: Based on these images, describe the possible tactile feelings of the contact patch using sensory adjectives. Limit your response up to five adjectives, separated by commas.""" # prompt 2

prompt = """
Image: The image is from a camera observing a room sized scene from a vertical perspective.
Example: For a room with a kitchen counter, an object to find might be "a coffee mug".
Task: Based on the image, describe an object that you might find in the scene. Limit your response to a single object. # prompt 3
"""

# center crop
# def crop_image(img, crop_size):
#     # Calculate the left, upper, right, and lower pixel coordinate for cropping
#     left = (img.width - crop_size) / 2
#     top = (img.height - crop_size) / 2
#     right = (img.width + crop_size) / 2
#     bottom = (img.height + crop_size) / 2

#     # Crop the image
#     img_cropped = img.crop((left, top, right, bottom))

#     # Resize the image to 512x512
#     img_resized = img_cropped.resize((512, 512))
#     return img_resized

def crop_image(img, crop_size, rgb_size : list = [224, 224], im_scale_range : list = [.12, .1], top_idx=0):
    rgb = TO_TENSOR(img)
    max_scale = rgb_size[0] / (
            im_scale_range[0] * min(rgb.shape[1], rgb.shape[2]))
    scaled_size = (int(max_scale * rgb.shape[1]), int(max_scale * rgb.shape[2]))
    rgb = TF.resize(rgb, scaled_size)
    
    # Calculate the size for cropping
    crop_height = int(np.ceil(np.sqrt(2) * im_scale_range[1] * max(rgb.shape[1], rgb.shape[2])))
    crop_width = crop_height  # Width remains the same

    # Calculate top left corner of the crop
    top = top_idx
    # top = 0 # Crop for Max's Dataset
    # top = 200  # Crop from the top (for will's dataset)
    left = (rgb.shape[2] - crop_width) // 2  # Center horizontally
    rgb = TF.crop(rgb, top, left, crop_height, crop_width)
    rgb = to_pil(rgb)
    img_resized = rgb.resize((512, 512))
    return img_resized

def encode_image(image_path, crop=False):
  # Open the image
    if crop:
        with Image.open(image_path) as img:
            # Determine the size for cropping (the minimum of width and height)
            crop_size = min(img.size) // 3
            top_idx = 200 if "will" in image_path else 0
            img_resized = crop_image(img, crop_size, top_idx=top_idx)

            # Save the processed image to a bytes object
            img_byte_arr = io.BytesIO()
            img_resized.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Encode the image to base64
            return base64.b64encode(img_byte_arr).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(vision_fp, tactile_fp, prompt):
    vision_image = encode_image(vision_fp)
    # cropped_vision_image = encode_image(vision_fp, crop=True)
    # tactile_image = encode_image(tactile_fp)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

    payload = {
        "model": "gpt-4-vision-preview",
        "seed": 42,
        "temperature": 0.8, 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{vision_image}",
                    }
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{cropped_vision_image}",
                    #     }
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{tactile_image}",
                    #     }
                    # },
                ]
            }, 
        ],
        "max_tokens": 300
    }
    success = False
    num_attempts = 0
    gpt_response = None
    while not success and num_attempts < 3:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            gpt_response = response.json()['choices'][0]['message']['content']
        except:
            num_attempts += 1
            gpt_response = "error"
            print("Error in generating response, generating again...")
            continue
        print(gpt_response)
        if len(gpt_response) > 0 and len(gpt_response) < 100 and "sorry" not in gpt_response and "cannot" not in gpt_response and "request" not in gpt_response and "error" not in gpt_response:
            success = True
        else:
            gpt_response = None
            num_attempts += 1
    return gpt_response

if __name__ == "__main__":
  
    which_bag = "full_kitchen_blur_pose_bag_colmap"
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=f"/home/lifelong/L3GS/l3gs/sampled_images/{which_bag}/")
    parser.add_argument("--api_key", type=str, default="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")
    parser.add_argument("--out_dir", type=str, default=f"/home/lifelong/L3GS/l3gs/gpt_outputs/{which_bag}/")
    parser.add_argument("--output_csv", type=str, default="outputs.csv")
    parser.add_argument("--failure_csv", type=str, default="failures.csv")
    parser.add_argument("--vis_dir", default="sample", type=str)
    parser.add_argument("--contact_json", type=str, default="contact.json") # can also be contact_before.json if generated by Raven
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # set numpy seed 
    np.random.seed(args.seed)

    api_key = args.api_key

    ratings = []
    images = []
    labels = []
    tactiles = []

    # read_json = os.path.join(args.folder, args.contact_json)
    # with open(read_json, "r") as f:
    #     data = json.load(f)
    
    os.makedirs(args.out_dir, exist_ok=True)
    output_csv = os.path.join(args.out_dir, args.output_csv)
    failure_csv = os.path.join(args.out_dir, args.failure_csv)

    existing_images = []
    exists_csv = os.path.exists(output_csv)
    if exists_csv:
        with open(output_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                existing_images.append(row[0])
    
    exists_failure_csv = os.path.exists(failure_csv)
    if exists_failure_csv:
        with open(failure_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                existing_images.append(row[0])

    # tactiles, images, backgrounds = data["tactile"], data["vision"], data["background"]
    # images = [plt.imread(os.path.join(args.folder, image)) for image in images]
    images = [(args.folder + name) for name in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, name))]

    print("length of images: {}".format(len(images)))

    # find out the indices that has not been labeled 
    # unlabelled_indices = []
    # for idx in range(len(images)):
    #     image_fp = os.path.join(args.folder, images[idx])
    #     if image_fp not in existing_images:
    #         unlabelled_indices.append(idx)
    # print("number of unlabelled indices: {}".format(len(unlabelled_indices)))

    vis_dir = os.path.join(args.out_dir, args.vis_dir)
    os.makedirs(vis_dir, exist_ok=True)

    # index = 0 
    rows = []
    # if args.random_sample:
    #     print(f"randomly sample {args.num_samples} indices...")
    #     random_idx = np.random.choice(unlabelled_indices, min(args.num_samples, len(unlabelled_indices)), replace=False)
    # else:
    #     random_idx = unlabelled_indices
    # print("number of images to label: {}".format(len(random_idx)))
    write_append = 'w' if not exists_csv else 'a'

    f = open(output_csv, write_append, newline='')
    ff = open(failure_csv, write_append, newline='')
    
    writer = csv.writer(f)
    # Write the header
    if not exists_csv:
        writer.writerow(["url", "tactile", "tactile_background", "caption"])

    writer_ff = csv.writer(ff)
    # Write the header
    if not exists_failure_csv:
        writer_ff.writerow(["url"])

    total_errors = 0

    for i in tqdm(range(len(images))):
        image_fp = images[i]
        print('generating response for {}'.format(image_fp))
        # print("generating response for {} and {}".format(image_fp, tactile_fp))
        # image_fp = os.path.join(args.folder, image_fp)
        # if image_fp in existing_images:
        #     print("image {} already exists, skipping...".format(image_fp))
        #     continue
        # tactile_fp = os.path.join(args.folder, tactile_fp)
        # background_fp = os.path.join(args.folder, background_fp)
        assistant_response = generate_response(image_fp, None, prompt)
        # if assistant_response is None:
        #     writer_ff.writerow([image_fp])
        #     ff.flush()
        #     print("cannot generate response, skipping...")
        #     continue
        # if assistant_response == "error":
        #     print("GPT error, will regenerate next time...")
        #     total_errors += 1
        #     # if total_errors > 3:
        #     #     print("too many errors, exiting...")
        #     #     break
        #     continue
        assistant_response = assistant_response.lower()
        print("assistant response: {}".format(assistant_response))
        if args.vis or np.random.random() < 0.01: 
            image = Image.open(image_fp)
            # crop_size = min(Image.open(image_fp).size) // 3
            # cropped_image = crop_image(Image.open(image_fp), crop_size)
            # tactile_image = Image.open(tactile_fp)
            # # we create three side by side plot, with an overall title as the generated response 
            fig, ax = plt.subplots(1, 3)
            # fig.suptitle(assistant_response)
            ax[0].imshow(image)
            # ax[1].imshow(cropped_image)
            # ax[2].imshow(tactile_image)
            plt.savefig(os.path.join(vis_dir, os.path.basename(image_fp).replace(".jpg", ".png")))
            plt.close()
        # index += 1 
        # if args.vis and index > 10:
        #     break
        rows.append([image_fp, assistant_response])
        writer.writerow(rows[-1])
        f.flush()
    f.close()
    # ff.close()
