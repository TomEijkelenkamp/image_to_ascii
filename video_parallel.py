import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import string
import os

# Settings
ascii_chars = string.printable.strip()
font_path = "C:/Windows/Fonts/consolab.ttf"  # Adjust for your OS
font_size = 50
char_width, char_height = font_size, font_size
grid_cols = 30
output_fps = 60
output_video = "arial_no_background_00006_ascii_bigger.mp4"

font = ImageFont.truetype(font_path, font_size)

# === ASCII RENDERING ===

def render_char(char):
    img = Image.new('L', (char_width, char_height), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((char_width - w) / 2, (char_height - h) / 2), char, font=font, fill=255)
    return np.array(img)

def create_char_library_tensor():
    chars = list(ascii_chars)
    tensors = [torch.tensor(render_char(c), dtype=torch.float32) / 255.0 for c in chars]
    return torch.stack(tensors).unsqueeze(1).cuda(), chars  # (C, 1, H, W)

def extract_patches(img_resized):
    patches = []
    for y in range(0, img_resized.shape[0], char_height):
        for x in range(0, img_resized.shape[1], char_width):
            patch = img_resized[y:y+char_height, x:x+char_width]
            patches.append(patch)
    return torch.from_numpy(np.array(patches)).float().unsqueeze(1).cuda() / 255.0

def best_match_gpu(patches, char_tensors):
    diff = patches.unsqueeze(1) - char_tensors.unsqueeze(0)  # (N, C, 1, H, W)
    mse = torch.mean(diff ** 2, dim=[2, 3, 4])  # (N, C)
    return torch.argmin(mse, dim=1)  # (N,)

def render_ascii_frame_from_matches(matches, chars, grid_w, grid_h):
    ascii_img = Image.new("RGB", (grid_w * char_width, grid_h * char_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(ascii_img)

    i = 0
    for y in range(grid_h):
        for x in range(grid_w):
            ch = chars[matches[i]]
            draw.text((x * char_width, y * char_height), ch, font=font, fill=(255, 255, 255))
            i += 1
    return np.array(ascii_img)

def frame_to_ascii_image(frame, char_tensor, chars, grid_w, grid_h):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (grid_w * char_width, grid_h * char_height))
    patches = extract_patches(img_resized)
    match_indices = best_match_gpu(patches, char_tensor).cpu().numpy()
    return render_ascii_frame_from_matches(match_indices, chars, grid_w, grid_h)

# === MAIN LOOP ===

def video_to_ascii(video_path, output_path=output_video):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or output_fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output size based on character grid
    aspect_ratio = height / width
    grid_w = grid_cols
    grid_h = int(grid_w * aspect_ratio * (char_width / char_height))
    out_width = grid_w * char_width
    out_height = grid_h * char_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # GPU char library
    char_tensor, chars = create_char_library_tensor()

    print(f"🎞 Converting {frame_count} frames...")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ascii_frame = frame_to_ascii_image(frame, char_tensor, chars, grid_w, grid_h)
        out.write(cv2.cvtColor(ascii_frame, cv2.COLOR_RGB2BGR))
        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ ASCII video saved to: {output_path}")

if __name__ == "__main__":
    import sys
    input_video = sys.argv[1] if len(sys.argv) > 1 else "arial_no_background_00006.mp4"
    video_to_ascii(input_video)
