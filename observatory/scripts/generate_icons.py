"""Build installable PWA icons from the repository's vector mark (requires cairosvg)."""
from pathlib import Path
import cairosvg
root=Path(__file__).resolve().parents[1]/'web/public'
for size in [192,512]:
    cairosvg.svg2png(url=str(root/'icon.svg'),write_to=str(root/f'icon-{size}.png'),output_width=size,output_height=size)
