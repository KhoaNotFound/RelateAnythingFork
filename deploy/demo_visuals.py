"""Render instance masks and directed relation arrows for the Python demo."""
from __future__ import annotations

import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


def _font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    # Kaggle may not have DejaVu installed. Keep the requested size when
    # falling back; load_default() without size silently uses a tiny font.
    return ImageFont.load_default(size=size)


def _color(index):
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb((index * 0.618) % 1, 0.7, 1))


def _badge(draw, xy, text, font, color, width):
    bounds = draw.textbbox((0, 0), text, font=font)
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = max(2, min(xy[0], width - tw - 10))
    y = max(2, xy[1])
    draw.rounded_rectangle((x, y, x + tw + 8, y + th + 8), radius=4, fill=(15, 20, 30))
    draw.text((x + 4, y + 4 - bounds[1]), text, font=font, fill=color)


def _wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def render_regions(image, boxes, masks, labels, triplets, title):
    """Return an RGB preview; arrow numbers map to a readable relation legend.

    Mask overlays are independent, so overlapping instances retain boundaries.
    Input arrays, source image, and relation predictions are never modified.
    """
    overlay = image.convert("RGBA")
    anchors = []
    for idx, (box, mask) in enumerate(zip(boxes, masks)):
        color = _color(idx)
        binary = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8))
        layer = Image.new("RGBA", image.size, color + (0,))
        layer.putalpha(binary.point(lambda p: round(p * 0.32)))
        overlay = Image.alpha_composite(overlay, layer)
        boundary = ImageChops.subtract(binary, binary.filter(ImageFilter.MinFilter(3)))
        edge = Image.new("RGBA", image.size, color + (0,))
        edge.putalpha(boundary)
        overlay = Image.alpha_composite(overlay, edge)
        ys, xs = np.nonzero(mask)
        if len(xs):
            # Pick an actual foreground pixel nearest the centroid.
            closest = np.argmin((xs - xs.mean())**2 + (ys - ys.mean())**2)
            anchors.append((float(xs[closest]), float(ys[closest])))
        else:
            anchors.append(((float(box[0]) + float(box[2])) / 2,
                            (float(box[1]) + float(box[3])) / 2))

    scale = min(1600 / image.width, max(1.0, 900 / image.width))
    width, height = round(image.width * scale), round(image.height * scale)
    preview = overlay.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    anchors = [(x * scale, y * scale) for x, y in anchors]
    font = _font(18)
    relation_font = _font(18)
    line_height = 26
    draw = ImageDraw.Draw(preview)
    for idx, box in enumerate(boxes):
        _badge(draw, (float(box[0]) * scale, float(box[1]) * scale),
               f"#{idx} {labels[idx]}", font, _color(idx), width)

    legend = []
    for number, triplet in enumerate(triplets, 1):
        si, oi = triplet.subject_idx, triplet.object_idx
        start, end = np.array(anchors[si]), np.array(anchors[oi])
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 1:
            # Coincident endpoints: a visible loop still indicates direction.
            control = start + np.array([50.0, -70.0])
            end = end + np.array([8.0, 0.0])
        else:
            normal = np.array([-delta[1], delta[0]]) / distance
            control = (start + end) / 2 + normal * min(80, distance * 0.2)
        points = [tuple((1-t)**2*start + 2*(1-t)*t*control + t*t*end)
                  for t in np.linspace(0, 1, 35)]
        color = _color(si)
        draw.line(points, fill=(15, 20, 30), width=7)
        draw.line(points, fill=color, width=3)
        tip = np.array(points[-1])
        angle = math.atan2(tip[1]-points[-3][1], tip[0]-points[-3][0])
        wings = [tuple(tip - 15*np.array([math.cos(angle+a), math.sin(angle+a)]))
                 for a in (-0.45, 0.45)]
        draw.polygon([tuple(tip), *wings], fill=color)
        _badge(draw, points[len(points)//2], str(number), relation_font, color, width)
        text = (f"{number}. #{si} {labels[si]} -> {triplet.predicate} "
                f"[{triplet.score:.3f}] -> #{oi} {labels[oi]}")
        legend.extend((line, color) for line in _wrap(draw, text, relation_font, width-32))
    if not legend:
        legend = [("No relations returned." if len(boxes) >= 2 else
                   "Fewer than two objects: no relations to predict.", (220, 225, 235))]
    header = 40
    canvas = Image.new("RGB", (width, height + header + 18 + line_height*len(legend)), (15, 20, 30))
    canvas.paste(preview, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font, fill="white")
    for row, (line, color) in enumerate(legend):
        draw.text((12, header+height+8+row*line_height), line, font=relation_font, fill=color)
    return canvas


def save_visuals(image, boxes, masks, labels, predictions, output_base):
    paths = {}
    for name, triplets in {"masks": [], **predictions}.items():
        path = Path(f"{output_base}.{name}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        title = "Instance masks" if name == "masks" else f"Relations: {name} (subject -> object)"
        render_regions(image, boxes, masks, labels, triplets, title).save(path)
        paths[name] = str(path)
    return paths
