from PIL import Image, ImageFont, ImageDraw
import numpy as np
import svgwrite
import cairosvg

Image.MAX_IMAGE_PIXELS = None

# Settings
ascii_chars = "@%#*+=-:. "
font_path = "C:/Windows/Fonts/consola.ttf"
font_size = 12
char_width, char_height = font_size, font_size
grid_cols = 100

def mse(img1, img2):
    return np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

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
    min_err = float('inf')
    best_char = ' '
    for char, img in char_lib.items():
        err = mse(patch, img)
        if err < min_err:
            min_err = err
            best_char = char
    return best_char

def image_to_ascii_svg(image_path, output_path="ascii_output.svg"):
    img = Image.open(image_path).convert('L')
    orig_w, orig_h = img.size
    aspect_ratio = orig_h / orig_w
    grid_w = grid_cols
    grid_h = int(grid_w * aspect_ratio * (char_width / char_height))

    # Resize image to grid size in pixels
    img = img.resize((grid_w * char_width, grid_h * char_height))
    char_lib = create_char_library()

    # Create SVG document
    dwg = svgwrite.Drawing(output_path, profile='tiny',
                           size=(grid_w * char_width, grid_h * char_height),
                           viewBox=f"0 0 {grid_w * char_width} {grid_h * char_height}")
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='black'))

    for y in range(0, img.height, char_height):
        for x in range(0, img.width, char_width):
            patch = img.crop((x, y, x + char_width, y + char_height))
            ch = best_match(patch, char_lib)
            dwg.add(dwg.text(
                ch,
                insert=(x, y + char_height),  # Adjust baseline
                fill='white',
                font_size=font_size,
                font_family='Consolas'
            ))

    dwg.save()
    print(f"✅ SVG saved to {output_path}")

    cairosvg.svg2pdf(url="ascii_output.svg", write_to="ascii_output.pdf")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "van_gogh.jpg"
    image_to_ascii_svg(image_path)
