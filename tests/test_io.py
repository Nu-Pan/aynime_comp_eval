from pathlib import Path

import pytest
from PIL import Image

from discord_webp_tuner.io import composite_alpha_to_rgb, parse_bg_color, resize_to_long_edge


def test_parse_bg_color():
    assert parse_bg_color("808080") == (128, 128, 128)
    assert parse_bg_color("#ffffff") == (255, 255, 255)
    with pytest.raises(ValueError):
        parse_bg_color("fff")


def test_composite_alpha_to_rgb():
    img = Image.new("RGBA", (2, 2), color=(0, 0, 0, 0))
    img.putpixel((0, 0), (255, 0, 0, 128))
    out = composite_alpha_to_rgb(img, (255, 255, 255))
    assert out.mode == "RGB"
    assert out.size == (2, 2)
    r, g, b = out.getpixel((0, 0))
    assert r > g and r > b


def test_resize_to_long_edge():
    img = Image.new("RGB", (400, 200), color=(1, 2, 3))
    out = resize_to_long_edge(img, 100)
    assert out.size == (100, 50)

