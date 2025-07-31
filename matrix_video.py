import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

# === Instellingen ===
matrix_chars = list("アィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモヤャユュヨョラリルレロワヲンヴヵヶーァィゥェォッャュョヮヽヾㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ012345678901234567890123456789")
font_path = "NotoSansJP-Bold.ttf"  # Adjust for your OS
# font_path = "NotoSansJP-VariableFont_wght.ttf"  # Adjust for your OS

font_size = 45
# char_width, char_height = font_size, font_size
grid_cols = 36
output_fps = 60
output_video = "matrix_silhouette_output_6.mp4"

font = ImageFont.truetype(font_path, font_size)

def render_char(char, char_width, char_height):
    img = Image.new('L', (char_width, char_height), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((char_width - w) / 2, (char_height - h) / 2), char, font=font, fill=255)
    return np.array(img)

def generate_matrix_frame(frame, char_width=32, char_height=32, column_offsets=None, chars_np=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    grid_w = grid_cols
    grid_h = int(h / w * grid_w * (char_width / char_height))

    img_out = Image.new("RGB", (grid_w * char_width, grid_h * char_height), (0, 0, 0))

    draw = ImageDraw.Draw(img_out)

    for col in range(grid_w):
        for row in range(grid_h):
            if (row * char_height + column_offsets[col]) % h + char_height >= h:
                continue

            # patch = gray[(row * char_height + column_offsets[col]) % h:((row + 1) * char_height + column_offsets[col]) % h, col * char_width:(col + 1) * char_width]
            # char = matrix_chars[np.argmin([np.mean(np.abs(chars_np[i] - patch)) for i in range(len(chars_np))])]

            patch_average = np.mean(gray[(row * char_height + column_offsets[col]) % h:((row + 1) * char_height + column_offsets[col]) % h, col * char_width:(col + 1) * char_width])

            random.seed(hash((row, col)))
            char = random.choice(matrix_chars)

            draw.text((col * char_width, (row * char_height + column_offsets[col]) % h), char, font=font, fill=(0, int(patch_average), 0))

    # Threshold to get a binary image (just in case)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours (edges of white regions)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to OpenCV format (NumPy array, BGR color order)
    img_cv = cv2.cvtColor(np.array(img_out), cv2.COLOR_RGB2BGR)

    # Draw the contours on the original frame
    # Green = (0, 150, 0), thickness = 2
    cv2.drawContours(img_cv, contours, -1, (0, 150, 0), 2)

    img_out = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    return np.array(img_out)          


def matrix_silhouette_video(input_path, output_path=output_video):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or output_fps

    # Bereken karaktergrootte op basis van gewenste kolommen
    char_width = width // grid_cols
    char_height = char_width  # Vierkant blok

    grid_w = grid_cols
    grid_h = height // char_height

    out_width = grid_w * char_width
    out_height = grid_h * char_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Per column offset voor naar beneder vallende karakters effect
    column_offsets = [random.randint(0, char_height) for _ in range(grid_w)]
    column_speeds = [random.randint(5, 10) for _ in range(grid_w)]

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    chars_np = [np.array(render_char(c, char_width, char_height)) for c in matrix_chars]

    print(f"🎞 Converting {frame_count} frames to Matrix silhouette effect...")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        matrix_frame = generate_matrix_frame(frame, char_width, char_height, column_offsets, chars_np)
        out.write(cv2.cvtColor(matrix_frame, cv2.COLOR_RGB2BGR))

        column_offsets = [offset + speed for offset, speed in zip(column_offsets, column_speeds)]

        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Matrix silhouette video saved to: {output_path}")

# === Uitvoeren ===
if __name__ == "__main__":
    import sys
    input_video = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    matrix_silhouette_video(input_video)
