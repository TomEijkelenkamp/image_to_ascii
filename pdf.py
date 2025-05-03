from PIL import Image, ImageFont, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black, white
import numpy as np
import string


Image.MAX_IMAGE_PIXELS = None


# Settings
# ascii_chars = "@%#*+=-:. "
ascii_chars = string.printable.strip()
font_path = "C:/Windows/Fonts/consola.ttf"  # Monospaced font path (Windows)
font_size = 8
char_width, char_height = font_size, font_size
grid_cols = 100

def mse(img1, img2):
    return np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)

def render_char(char, size):
    font = ImageFont.truetype(font_path, size)
    img = Image.new('L', (char_width, char_height), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((char_width - w) / 2, (char_height - h) / 2), char, font=font, fill=255)
    return img

def create_char_library():
    return {c: render_char(c, font_size) for c in ascii_chars}

def best_match(patch, char_lib):
    return min(char_lib, key=lambda c: mse(patch, char_lib[c]))

def image_to_ascii_pdf(image_path, pdf_path="ascii_output.pdf"):
    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    aspect_ratio = orig_h / orig_w
    grid_w = grid_cols
    grid_h = int(grid_w * aspect_ratio * (char_width / char_height))
    img = img.resize((grid_w * char_width, grid_h * char_height))

    char_lib = create_char_library()

    # Set up PDF canvas
    c = canvas.Canvas(pdf_path, pagesize=(grid_w * char_width, grid_h * char_height))
    c.setFillColor(black)
    c.rect(0, 0, grid_w * char_width, grid_h * char_height, stroke=0, fill=1)

    c.setFont("Courier", font_size)
    c.setFillColor(white)

    for y in range(0, img.height, char_height):
        for x in range(0, img.width, char_width):
            patch = img.crop((x, y, x + char_width, y + char_height))
            ch = best_match(patch, char_lib)

            # PDF origin is bottom-left
            pos_x = x
            pos_y = img.height - y - char_height

            c.drawString(pos_x, pos_y, ch)

    c.showPage()
    c.save()
    print(f"✅ PDF saved as '{pdf_path}'")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "van_gogh.jpg"
    output_path = "ascii_output.pdf"
    image_to_ascii_pdf(image_path, output_path)
