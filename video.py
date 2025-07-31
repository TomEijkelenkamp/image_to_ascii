import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import os

# Settings
ascii_chars = string.printable.strip()
font_path = "C:/Windows/Fonts/consola.ttf"
font_size = 50
char_width, char_height = font_size, font_size
grid_cols = 24
output_fps = 60
output_video = "ascii_output_24_60.mp4"

font = ImageFont.truetype(font_path, font_size)

def render_char(char):
    img = Image.new('L', (char_width, char_height), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((char_width - w) / 2, (char_height - h) / 2), char, font=font, fill=255)
    return np.array(img)

def create_char_library():
    return {c: render_char(c) for c in ascii_chars}

def mse(patch, char_img):
    return np.mean((patch.astype(np.float32) - char_img.astype(np.float32)) ** 2)

def best_match(patch, char_lib):
    min_err = float('inf')
    best_char = ' '
    for char, img in char_lib.items():
        err = mse(patch, img)
        if err < min_err:
            min_err = err
            best_char = char
    return best_char

def frame_to_ascii_image(frame, char_lib):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    aspect_ratio = h / w
    grid_w = grid_cols
    grid_h = int(grid_w * aspect_ratio * (char_width / char_height))
    img_resized = cv2.resize(img_gray, (grid_w * char_width, grid_h * char_height))

    # Output image
    ascii_img = Image.new("RGB", (grid_w * char_width, grid_h * char_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(ascii_img)

    for y in range(0, img_resized.shape[0], char_height):
        for x in range(0, img_resized.shape[1], char_width):
            patch = img_resized[y:y+char_height, x:x+char_width]
            ch = best_match(patch, char_lib)
            draw.text((x, y), ch, font=font, fill=(255, 255, 255))

    return np.array(ascii_img)

def video_to_ascii(video_path, output_path=output_video):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or output_fps

    # Estimate new height for ascii video
    aspect_ratio = height / width
    grid_w = grid_cols
    grid_h = int(grid_w * aspect_ratio * (char_width / char_height))
    out_width = grid_w * char_width
    out_height = grid_h * char_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    char_lib = create_char_library()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 Converting {frame_count} frames...")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ascii_frame = frame_to_ascii_image(frame, char_lib)
        out.write(cv2.cvtColor(ascii_frame, cv2.COLOR_RGB2BGR))
        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ ASCII video saved to: {output_path}")

if __name__ == "__main__":
    import sys
    input_video = sys.argv[1] if len(sys.argv) > 1 else "arial.mp4"
    video_to_ascii(input_video)
