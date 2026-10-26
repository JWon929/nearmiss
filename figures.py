"""
Diagrams for the Method and Purpose pages, drawn as SVG so they are sharp at any size, need no image files and no
internet. Each function returns an <img> tag with the SVG inlined as a data URI.

Colours follow the app's page colours. Text is a mid grey that reads on light and dark backgrounds.
"""
from __future__ import annotations

import base64

INK = "#64748b"          # labels: readable on white and on Streamlit's dark background
BLUE, VIOLET, TEAL, ORANGE, GREEN, PINK, INDIGO = ("#2563eb", "#7c3aed", "#0d9488", "#ea580c", "#16a34a", "#db2777",
                                                   "#4f46e5")
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def _img(svg: str, alt: str) -> str:
    data = base64.b64encode(svg.encode()).decode()
    return f"<img alt='{alt}' src='data:image/svg+xml;base64,{data}'/>"


def _arrow_defs(*colors: str) -> str:
    return "<defs>" + "".join(
        f"<marker id='a{c[1:]}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
        f"<path d='M0,0 L10,5 L0,10 z' fill='{c}'/></marker>" for c in colors) + "</defs>"


def pipeline() -> str:
    """Video -> detect & track -> calibrate -> trajectories -> conflict check -> review -> report."""
    steps = [("Video", "#94a3b8"), ("Detect &amp; track", BLUE), ("Calibrate", VIOLET), ("Trajectories", INDIGO),
             ("Conflict check", TEAL), ("Your review", ORANGE), ("Report", GREEN)]
    w, h, bw, gap = 980, 120, 118, 22
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' {FONT}>", _arrow_defs(INK)]
    for i, (label, c) in enumerate(steps):
        x = 10 + i * (bw + gap)
        parts.append(f"<rect x='{x}' y='28' width='{bw}' height='56' rx='14' fill='{c}' opacity='.92'/>")
        parts.append(f"<text x='{x + bw / 2}' y='61' text-anchor='middle' font-size='15' font-weight='600' "
                     f"fill='#fff'>{label}</text>")
        if i < len(steps) - 1:
            parts.append(f"<line x1='{x + bw + 2}' y1='56' x2='{x + bw + gap - 3}' y2='56' stroke='{INK}' "
                         f"stroke-width='2.5' marker-end='url(#a{INK[1:]})'/>")
    parts.append(f"<text x='{w / 2}' y='112' text-anchor='middle' font-size='13' fill='{INK}'>Everything after the "
                 f"video runs on this computer. Only your review decides what was real.</text></svg>")
    return _img("".join(parts), "NearMiss pipeline")


def calibration() -> str:
    """A rectangle seen at an angle (camera view) mapped to its true shape (top-down)."""
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 640 230' {FONT}>{_arrow_defs(VIOLET)}"
         # camera view: road and a trapezoid crosswalk
         f"<rect x='10' y='10' width='250' height='190' rx='12' fill='{INK}' opacity='.10'/>"
         f"<polygon points='95,40 165,40 245,195 25,195' fill='{INK}' opacity='.25'/>"
         f"<polygon points='88,110 172,110 196,160 64,160' fill='none' stroke='{VIOLET}' stroke-width='3'/>"
         + "".join(f"<circle cx='{x}' cy='{y}' r='6' fill='{VIOLET}'/>" for x, y in ((88, 110), (172, 110), (196, 160),
                                                                                     (64, 160)))
         + f"<text x='135' y='222' text-anchor='middle' font-size='14' fill='{INK}'>Camera view: 4 clicks</text>"
         # arrow
         f"<line x1='275' y1='110' x2='370' y2='110' stroke='{VIOLET}' stroke-width='3' marker-end='url(#a{VIOLET[1:]})'/>"
         f"<text x='322' y='97' text-anchor='middle' font-size='13' fill='{INK}'>homography</text>"
         # top-down
         f"<rect x='385' y='10' width='245' height='190' rx='12' fill='{INK}' opacity='.10'/>"
         f"<rect x='450' y='60' width='120' height='90' fill='none' stroke='{VIOLET}' stroke-width='3'/>"
         f"<text x='510' y='52' text-anchor='middle' font-size='13' fill='{VIOLET}' font-weight='600'>8.0 m</text>"
         f"<text x='586' y='110' font-size='13' fill='{VIOLET}' font-weight='600'>3.5 m</text>"
         f"<text x='507' y='222' text-anchor='middle' font-size='14' fill='{INK}'>Top-down, in metres</text></svg>")
    return _img(s, "Calibration: from camera view to metres")


def ttc() -> str:
    """Two road users on a collision course; the predicted meeting point and the time until it."""
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 260' {FONT}>{_arrow_defs(BLUE, PINK, TEAL)}"
         f"<rect x='0' y='0' width='520' height='260' rx='14' fill='{INK}' opacity='.07'/>"
         # car moving right
         f"<rect x='40' y='150' width='54' height='28' rx='7' fill='{BLUE}'/>"
         f"<line x1='98' y1='164' x2='250' y2='164' stroke='{BLUE}' stroke-width='3' stroke-dasharray='7 6' "
         f"marker-end='url(#a{BLUE[1:]})'/>"
         f"<text x='67' y='200' text-anchor='middle' font-size='13' fill='{INK}'>car, 10 m/s</text>"
         # pedestrian moving down
         f"<circle cx='300' cy='40' r='11' fill='{PINK}'/>"
         f"<line x1='300' y1='56' x2='300' y2='138' stroke='{PINK}' stroke-width='3' stroke-dasharray='7 6' "
         f"marker-end='url(#a{PINK[1:]})'/>"
         f"<text x='318' y='44' font-size='13' fill='{INK}'>pedestrian, 1.4 m/s</text>"
         # collision zone
         f"<circle cx='300' cy='164' r='26' fill='{TEAL}' opacity='.18' stroke='{TEAL}' stroke-width='2.5'/>"
         f"<text x='340' y='170' font-size='14' fill='{TEAL}' font-weight='700'>they would touch here</text>"
         f"<text x='260' y='238' text-anchor='middle' font-size='14' fill='{INK}'>TTC = seconds until they meet if "
         f"neither slows down or turns</text></svg>")
    return _img(s, "Time to collision")


def pet() -> str:
    """Crossing paths and the time gap at the shared point."""
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 260' {FONT}>{_arrow_defs(BLUE, PINK)}"
         f"<rect x='0' y='0' width='520' height='260' rx='14' fill='{INK}' opacity='.07'/>"
         f"<line x1='20' y1='110' x2='480' y2='110' stroke='{BLUE}' stroke-width='3' marker-end='url(#a{BLUE[1:]})'/>"
         f"<line x1='250' y1='20' x2='250' y2='190' stroke='{PINK}' stroke-width='3' marker-end='url(#a{PINK[1:]})'/>"
         f"<circle cx='250' cy='110' r='14' fill='{ORANGE}' opacity='.85'/>"
         f"<text x='268' y='96' font-size='13' fill='{ORANGE}' font-weight='700'>crossing point</text>"
         f"<rect x='330' y='97' width='54' height='26' rx='7' fill='{BLUE}'/>"
         f"<text x='357' y='140' text-anchor='middle' font-size='12' fill='{INK}'>car has left</text>"
         f"<circle cx='250' cy='45' r='10' fill='{PINK}'/>"
         f"<text x='265' y='50' font-size='12' fill='{INK}'>pedestrian arriving</text>"
         # timeline
         f"<line x1='60' y1='222' x2='460' y2='222' stroke='{INK}' stroke-width='2'/>"
         f"<rect x='120' y='214' width='120' height='16' rx='4' fill='{BLUE}' opacity='.8'/>"
         f"<rect x='300' y='214' width='120' height='16' rx='4' fill='{PINK}' opacity='.8'/>"
         f"<text x='270' y='207' text-anchor='middle' font-size='13' font-weight='700' fill='{ORANGE}'>PET</text>"
         f"<line x1='240' y1='238' x2='300' y2='238' stroke='{ORANGE}' stroke-width='3'/>"
         f"<text x='260' y='255' text-anchor='middle' font-size='12' fill='{INK}'>time the point was empty between them"
         f"</text></svg>")
    return _img(s, "Post-encroachment time")


def drac() -> str:
    """Braking needed: closing speed and the distance left."""
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 200' {FONT}>{_arrow_defs(ORANGE)}"
         f"<rect x='0' y='0' width='520' height='200' rx='14' fill='{INK}' opacity='.07'/>"
         f"<rect x='40' y='80' width='60' height='30' rx='7' fill='{BLUE}'/>"
         f"<rect x='380' y='80' width='60' height='30' rx='7' fill='{INK}' opacity='.6'/>"
         f"<text x='410' y='130' text-anchor='middle' font-size='12' fill='{INK}'>stopped car</text>"
         f"<line x1='106' y1='95' x2='372' y2='95' stroke='{ORANGE}' stroke-width='3' marker-end='url(#a{ORANGE[1:]})'/>"
         f"<text x='240' y='84' text-anchor='middle' font-size='13' fill='{ORANGE}' font-weight='700'>closing at 12 m/s, "
         f"20 m left</text>"
         f"<text x='260' y='160' text-anchor='middle' font-size='15' fill='{INK}'>DRAC = speed² / (2 × distance) = "
         f"3.6 m/s²</text>"
         f"<text x='260' y='184' text-anchor='middle' font-size='12' fill='{INK}'>above about 3.4 m/s² is harder than "
         f"comfortable braking</text></svg>")
    return _img(s, "Deceleration rate to avoid the crash")


def closest() -> str:
    """Distance between two road users over time; the lowest point is the closest approach."""
    pts = [(40, 40), (90, 52), (140, 75), (190, 110), (240, 150), (270, 164), (300, 158), (350, 120), (400, 80),
           (450, 55), (480, 45)]
    path = "M" + " L".join(f"{x},{y}" for x, y in pts)
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 210' {FONT}>"
         f"<rect x='0' y='0' width='520' height='210' rx='14' fill='{INK}' opacity='.07'/>"
         f"<line x1='30' y1='180' x2='495' y2='180' stroke='{INK}' stroke-width='1.5'/>"
         f"<line x1='30' y1='20' x2='30' y2='180' stroke='{INK}' stroke-width='1.5'/>"
         f"<path d='{path}' fill='none' stroke='{TEAL}' stroke-width='3.5' stroke-linejoin='round'/>"
         f"<circle cx='270' cy='164' r='7' fill='{ORANGE}'/>"
         f"<text x='282' y='150' font-size='13' fill='{ORANGE}' font-weight='700'>closest: 1.4 m</text>"
         f"<line x1='30' y1='140' x2='495' y2='140' stroke='{PINK}' stroke-width='1.5' stroke-dasharray='6 5'/>"
         f"<text x='490' y='134' text-anchor='end' font-size='12' fill='{PINK}'>flag threshold</text>"
         f"<text x='262' y='200' text-anchor='middle' font-size='12' fill='{INK}'>time</text>"
         f"<text x='22' y='100' text-anchor='middle' font-size='12' fill='{INK}' transform='rotate(-90 22 100)'>distance"
         f"</text></svg>")
    return _img(s, "Closest approach")


def privacy() -> str:
    """Street-level: head and plate regions pixelated; drone: whole road user."""
    def pix(x, y, w, h, c):
        return "".join(f"<rect x='{x + i * w / 4}' y='{y + j * h / 3}' width='{w / 4 + .5}' height='{h / 3 + .5}' "
                       f"fill='{c}' opacity='{.45 + .12 * ((i + j) % 3)}'/>" for i in range(4) for j in range(3))
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 520 190' {FONT}>"
         f"<rect x='0' y='0' width='520' height='190' rx='14' fill='{INK}' opacity='.07'/>"
         f"<rect x='40' y='30' width='44' height='110' rx='8' fill='none' stroke='{PINK}' stroke-width='2.5'/>"
         + pix(40, 30, 44, 34, PINK) +
         f"<rect x='120' y='70' width='130' height='70' rx='10' fill='none' stroke='{BLUE}' stroke-width='2.5'/>"
         + pix(120, 70, 130, 26, BLUE) + pix(120, 112, 130, 28, BLUE) +
         f"<text x='145' y='170' text-anchor='middle' font-size='13' fill='{INK}'>street-level: head and plate areas"
         f"</text>"
         f"<rect x='330' y='45' width='60' height='100' rx='10' fill='none' stroke='{BLUE}' stroke-width='2.5'/>"
         + pix(330, 45, 60, 100, BLUE) +
         f"<rect x='410' y='75' width='34' height='34' rx='6' fill='none' stroke='{PINK}' stroke-width='2.5'/>"
         + pix(410, 75, 34, 34, PINK) +
         f"<text x='400' y='170' text-anchor='middle' font-size='13' fill='{INK}'>drone: the whole road user</text></svg>")
    return _img(s, "Privacy blur")


def figure(img_tag: str, caption: str = "") -> str:
    return f"<figure class='nm-fig'>{img_tag}" + (f"<figcaption>{caption}</figcaption>" if caption else "") + "</figure>"
