"""Generate the repository banner + social preview image (1280x640, GitHub 2:1).

Outputs:
    docs/banner.png       committed, embedded at the top of the README
    social_preview.png    local-only (gitignored): upload manually via
                          GitHub -> Settings -> Social preview (no upload API)

Deterministic: no randomness, no network. Requires Pillow.

Usage:
    python scripts/make_banner.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640

# Dark slate background with a subtle two-stop gradient (deterministic).
BG_TOP = (13, 17, 23)      # near GitHub dark canvas
BG_BOTTOM = (22, 30, 43)
ACCENT = (46, 211, 183)    # teal
ACCENT2 = (97, 142, 245)   # blue
TEXT = (234, 238, 242)
MUTED = (139, 148, 158)

TITLE = "LLM Cost Autopilot"
SUBTITLE = "Classify complexity. Route to the cheapest capable model."
CHIPS = [
    "93.6% routing accuracy",
    "53.2% simulated cost savings",
    "FastAPI + Streamlit",
    "109 tests",
]

FONT_CANDIDATES = {
    "bold": ["seguiemf.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
    "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
}


def _font(kind: str, size: int):
    import PIL
    for name in FONT_CANDIDATES[kind]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size) if hasattr(ImageFont, "load_default") else ImageFont.load_default()


def _gradient() -> Image.Image:
    img = Image.new("RGB", (W, H), BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        # ease the blend slightly toward the bottom color
        c = tuple(int(a + (b - a) * (t ** 1.2)) for a, b in zip(BG_TOP, BG_BOTTOM))
        d.line([(0, y), (W, y)], fill=c)
    return img


def _draw_route_arrows(d: ImageDraw.ImageDraw) -> None:
    """Three dots left (tiers) to three model boxes right, with connector lines."""
    left_x, right_x = 120, W - 120
    ys = [250, 320, 390]
    tier_labels = ["tier 1", "tier 2", "tier 3"]
    model_labels = ["local / cheap", "mid-tier", "premium"]
    dot_r = 9
    for y, tl, ml in zip(ys, tier_labels, model_labels):
        d.ellipse([left_x - dot_r, y - dot_r, left_x + dot_r, y + dot_r], fill=ACCENT)
        d.line([(left_x + dot_r + 4, y), (right_x - 90, y)], fill=(70, 82, 99), width=3)
        d.polygon([(right_x - 90, y - 7), (right_x - 90, y + 7), (right_x - 80, y)], fill=(70, 82, 99))
        box = [(right_x - 76, y - 26), (right_x + 60, y + 26)]
        d.rounded_rectangle(box, radius=14, fill=(31, 40, 55), outline=(58, 70, 86), width=2)
        f = _font("regular", 22)
        tw = d.textlength(ml, font=f)
        d.text((right_x - 8 - tw / 2, y - 14), ml, font=f, fill=TEXT)
        fs = _font("regular", 18)
        d.text((left_x - 70, y - 11), tl, font=fs, fill=MUTED)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    docs = root / "docs"
    docs.mkdir(exist_ok=True)

    img = _gradient()
    d = ImageDraw.Draw(img)

    # Decorative accent bar on the left edge.
    d.rectangle([0, 0, 10, H], fill=ACCENT)

    # Title + subtitle.
    f_title = _font("bold", 84)
    f_sub = _font("regular", 34)
    d.text((80, 84), TITLE, font=f_title, fill=TEXT)
    sw = d.textlength(SUBTITLE, font=f_sub)
    d.text(((W - sw) / 2, 196), SUBTITLE, font=f_sub, fill=MUTED)

    _draw_route_arrows(d)

    # Metric chips row.
    f_chip = _font("regular", 24)
    x, y, chip_h = 80, 470, 52
    gap = 18
    for text in CHIPS:
        tw = d.textlength(text, font=f_chip)
        pad = 20
        box = [x, y, x + tw + 2 * pad, y + chip_h]
        d.rounded_rectangle(box, radius=26, outline=ACCENT2, width=2, fill=(28, 36, 50))
        d.text((x + pad, y + 11), text, font=f_chip, fill=TEXT)
        x = box[2] + gap
        if x > W - 220:  # wrap conservatively (defensive; chips fit at current sizes)
            x, y = 80, y + chip_h + gap

    # Footer strip.
    f_foot = _font("regular", 20)
    foot = "FastAPI  ·  OpenAI / Anthropic / Ollama  ·  scikit-learn  ·  quality verification & auto-escalation"
    fw = d.textlength(foot, font=f_foot)
    d.text(((W - fw) / 2, H - 58), foot, font=f_foot, fill=MUTED)

    out_banner = docs / "banner.png"
    img.save(out_banner, "PNG", optimize=True)
    print(f"wrote {out_banner} ({out_banner.stat().st_size/1024:.0f} KB)")

    # Social preview: same art, saved at repo root, gitignored, for manual upload.
    out_social = root / "social_preview.png"
    img.save(out_social, "PNG", optimize=True)
    print(f"wrote {out_social} ({out_social.stat().st_size/1024:.0f} KB)")
    print("upload manually: GitHub repo -> Settings -> Social preview -> Edit -> Upload")


if __name__ == "__main__":
    main()
