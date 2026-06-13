"""
================================================================================
  Lightroom Preset Generator — Analyse d'image vers fichier .XMP
================================================================================
  Auteur  : MVP · Analyse visuelle → Paramètres Lightroom
  Version : 1.0.0
  Dépendances : Pillow, NumPy (pip install Pillow numpy)

  Usage :
      python lightroom_preset_generator.py <chemin_image> [nom_preset]

  Exemple :
      python lightroom_preset_generator.py photo.jpg "MonPreset"
================================================================================
"""

import sys
import os
import colorsys
import random
from datetime import datetime

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — ANALYSE DE L'IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def load_image_as_array(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge une image et retourne :
    - rgb_array  : tableau (H, W, 3) en float32 normalisé [0.0, 1.0]
    - gray_array : tableau (H, W)   luminosité perceptuelle en float32 [0.0, 1.0]
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32) / 255.0          # [0, 1]

    # Luminosité perceptuelle : coefficients ITU-R BT.601
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return rgb, gray


def analyze_exposure(gray: np.ndarray) -> float:
    """
    Exposition estimée à partir de la luminosité moyenne.
    Plage Lightroom : [-5.0, +5.0], 0 = neutre (luminosité ~0.5).
    Formule : (mean - 0.5) × 10 → clampé sur [-5, 5]
    """
    mean_lum = float(np.mean(gray))
    exposure = (mean_lum - 0.5) * 10.0
    return float(np.clip(exposure, -5.0, 5.0))


def analyze_contrast(gray: np.ndarray) -> int:
    """
    Contraste estimé via l'écart-type de la luminosité.
    std faible  (< 0.12) → image plate     → contraste positif
    std élevé   (> 0.25) → très contrasté  → contraste négatif
    Plage Lightroom : [-100, +100]
    """
    std = float(np.std(gray))
    contrast = int(np.clip((0.18 - std) * 300, -100, 100))
    return contrast


def analyze_highlights(gray: np.ndarray) -> int:
    """
    Hautes lumières : percentile 95 de la luminosité.
    Pixels très clairs → récupérer (valeur négative).
    Plage Lightroom : [-100, +100]
    """
    p95 = float(np.percentile(gray, 95))
    highlights = int(np.clip((0.85 - p95) * 400, -100, 100))
    return highlights


def analyze_shadows(gray: np.ndarray) -> int:
    """
    Ombres : percentile 5 de la luminosité.
    Pixels très sombres → déboucher (valeur positive).
    Plage Lightroom : [-100, +100]
    """
    p05 = float(np.percentile(gray, 5))
    shadows = int(np.clip((0.10 - p05) * 500, -100, 100))
    return shadows


def analyze_whites(gray: np.ndarray) -> int:
    """
    Blancs : percentile 99 (pixels les plus lumineux).
    Plage Lightroom : [-100, +100]
    """
    p99 = float(np.percentile(gray, 99))
    whites = int(np.clip((0.92 - p99) * 300, -100, 100))
    return whites


def analyze_blacks(gray: np.ndarray) -> int:
    """
    Noirs : percentile 1 (pixels les plus sombres).
    Plage Lightroom : [-100, +100]
    """
    p01 = float(np.percentile(gray, 1))
    blacks = int(np.clip((0.04 - p01) * 600, -100, 100))
    return blacks


def analyze_white_balance(rgb: np.ndarray) -> tuple[int, int]:
    """
    Température et Teinte estimées à partir des moyennes des canaux R, G, B.

    Température (unité interne LR ~[-100, +100]) :
        ratio R/B > 1 → image chaude → compenser avec valeur négative
        ratio R/B < 1 → image froide → compenser avec valeur positive

    Teinte [-150, +150] :
        surplus de vert (G > (R+B)/2) → teinte positive (magenta pour compenser)
    """
    r_mean = float(np.mean(rgb[:, :, 0]))
    g_mean = float(np.mean(rgb[:, :, 1]))
    b_mean = float(np.mean(rgb[:, :, 2]))

    rb_ratio = r_mean / (b_mean + 1e-6)
    temperature = int(np.clip((1.0 - rb_ratio) * 100, -100, 100))

    green_bias = g_mean - (r_mean + b_mean) / 2.0
    tint = int(np.clip(-green_bias * 200, -150, 150))

    return temperature, tint


def analyze_vibrance_saturation(rgb: np.ndarray) -> tuple[int, int]:
    """
    Vibrance et Saturation estimées via la saturation moyenne en espace HSV.
    Plages LR : Vibrance [-100, +100], Saturation [-100, +100]
    """
    flat_rgb = rgb.reshape(-1, 3)[::10]

    saturations = np.array([
        colorsys.rgb_to_hsv(float(r), float(g), float(b))[1]
        for r, g, b in flat_rgb
    ], dtype=np.float32)

    mean_sat = float(np.mean(saturations))
    vibrance  = int(np.clip((0.40 - mean_sat) * 150, -100, 100))
    saturation = int(np.clip((0.45 - mean_sat) * 100, -80, 80))

    return vibrance, saturation


def analyze_color_grading(rgb: np.ndarray, gray: np.ndarray) -> dict:
    """
    Color Grading — teinte dominante par zone tonale.
    Zones : Shadows (<0.25), Midtones (0.25–0.75), Highlights (>0.75)
    """
    result = {}
    zones = {
        "Shadow":    gray < 0.25,
        "Midtone":   (gray >= 0.25) & (gray < 0.75),
        "Highlight": gray >= 0.75,
    }

    for zone_name, mask in zones.items():
        pixels = rgb[mask]
        if len(pixels) < 50:
            result[zone_name] = {"Hue": 0, "Sat": 0, "Lum": 0}
            continue

        sample = pixels[::max(1, len(pixels) // 1000)]

        hues, sats = [], []
        for r, g, b in sample:
            h, s, _ = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            if s > 0.05:
                hues.append(h * 360)
                sats.append(s)

        if hues:
            mean_hue = float(np.mean(hues))
            mean_sat = float(np.mean(sats))
            cg_sat   = int(np.clip(mean_sat * 30, 0, 30))
        else:
            mean_hue = 0.0
            cg_sat   = 0

        result[zone_name] = {"Hue": round(mean_hue, 1), "Sat": cg_sat, "Lum": 0}

    return result


def run_full_analysis(image_path: str) -> dict:
    """
    Analyse complète d'une image.
    Retourne un dictionnaire de tous les paramètres Lightroom estimés.
    """
    print(f"\n📷  Analyse de : {os.path.basename(image_path)}")
    print("─" * 50)

    rgb, gray = load_image_as_array(image_path)

    exposure             = analyze_exposure(gray)
    contrast             = analyze_contrast(gray)
    highlights           = analyze_highlights(gray)
    shadows              = analyze_shadows(gray)
    whites               = analyze_whites(gray)
    blacks               = analyze_blacks(gray)
    temperature, tint    = analyze_white_balance(rgb)
    vibrance, saturation = analyze_vibrance_saturation(rgb)
    color_grading        = analyze_color_grading(rgb, gray)

    params = {
        "Exposure2012":   round(exposure, 2),
        "Contrast2012":   contrast,
        "Highlights2012": highlights,
        "Shadows2012":    shadows,
        "Whites2012":     whites,
        "Blacks2012":     blacks,
        "Temperature":    temperature,
        "Tint":           tint,
        "Vibrance":       vibrance,
        "Saturation":     saturation,
        "ColorGrading":   color_grading,
    }

    for k, v in params.items():
        if k != "ColorGrading":
            print(f"  {k:<22} {v:>8}")
    cg = color_grading
    print(f"\n  Color Grading — Shadows   : Hue={cg['Shadow']['Hue']:.0f}°  Sat={cg['Shadow']['Sat']}")
    print(f"  Color Grading — Midtones  : Hue={cg['Midtone']['Hue']:.0f}°  Sat={cg['Midtone']['Sat']}")
    print(f"  Color Grading — Highlights: Hue={cg['Highlight']['Hue']:.0f}°  Sat={cg['Highlight']['Sat']}")

    return params


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — GÉNÉRATION DU FICHIER .XMP
# ─────────────────────────────────────────────────────────────────────────────

def _make_uuid() -> str:
    chars = "0123456789abcdef"
    uid = "".join(random.choices(chars, k=32))
    return f"{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:]}"


def generate_xmp_file(params: dict, preset_name: str, output_path: str) -> str:
    """
    Génère et écrit le fichier .xmp sur le disque via template de chaîne.
    Structure conforme au schéma CRS (Camera Raw Settings) Lightroom Classic.
    Retourne le chemin absolu du fichier créé.
    """
    cg = params["ColorGrading"]
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    uid = _make_uuid()

    xmp_content = f"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about=""
            xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
            xmlns:xmp="http://ns.adobe.com/xap/1.0/"

            xmp:CreatorTool="LightroomPresetGenerator-MVP"
            xmp:CreateDate="{now}"

            crs:PresetType="Normal"
            crs:PresetName="{preset_name}"
            crs:UUID="{uid}"
            crs:ProcessVersion="15.4"

            crs:Exposure2012="{params['Exposure2012']}"
            crs:Contrast2012="{params['Contrast2012']}"
            crs:Highlights2012="{params['Highlights2012']}"
            crs:Shadows2012="{params['Shadows2012']}"
            crs:Whites2012="{params['Whites2012']}"
            crs:Blacks2012="{params['Blacks2012']}"

            crs:Temperature="{params['Temperature']}"
            crs:Tint="{params['Tint']}"

            crs:Vibrance="{params['Vibrance']}"
            crs:Saturation="{params['Saturation']}"

            crs:ColorGradeShadowHue="{int(cg['Shadow']['Hue'])}"
            crs:ColorGradeShadowSat="{cg['Shadow']['Sat']}"
            crs:ColorGradeShadowLum="{cg['Shadow']['Lum']}"
            crs:ColorGradeMidtoneHue="{int(cg['Midtone']['Hue'])}"
            crs:ColorGradeMidtoneSat="{cg['Midtone']['Sat']}"
            crs:ColorGradeMidtoneLum="{cg['Midtone']['Lum']}"
            crs:ColorGradeHighlightHue="{int(cg['Highlight']['Hue'])}"
            crs:ColorGradeHighlightSat="{cg['Highlight']['Sat']}"
            crs:ColorGradeHighlightLum="{cg['Highlight']['Lum']}"
            crs:ColorGradeGlobalHue="0"
            crs:ColorGradeGlobalSat="0"
            crs:ColorGradeGlobalLum="0"
            crs:ColorGradeBlending="50"
            crs:ColorGradeBalance="0"

            crs:ConvertToGrayscale="False"
            crs:EnableColorAdjustments="False"
            crs:EnableDetail="False"
            crs:EnableEffects="False"
            crs:EnableGraduatedFilters="False"
            crs:EnableLensCorrections="False"
            crs:EnableRadialFilters="False"
            crs:EnableRedEye="False"
            crs:EnableRetouch="False"
            crs:EnableSplitToning="False"/>
   </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xmp_content)

    return os.path.abspath(output_path)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — VALEURS DE DÉMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PARAMS = {
    "Exposure2012":   -0.35,
    "Contrast2012":    25,
    "Highlights2012": -60,
    "Shadows2012":     40,
    "Whites2012":     -15,
    "Blacks2012":     -20,
    "Temperature":    -12,
    "Tint":             8,
    "Vibrance":        30,
    "Saturation":      10,
    "ColorGrading": {
        "Shadow":    {"Hue": 220, "Sat": 12, "Lum": -5},
        "Midtone":   {"Hue":  35, "Sat":  8, "Lum":  0},
        "Highlight": {"Hue":  50, "Sat": 10, "Lum":  5},
    }
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("\n⚠️  Aucune image fournie — utilisation des valeurs de démonstration.\n")
        params      = DEMO_PARAMS
        preset_name = "DemoPreset_MVP"
        output_xmp  = "preset_demo.xmp"
    else:
        image_path  = sys.argv[1]
        preset_name = sys.argv[2] if len(sys.argv) > 2 else "GeneratedPreset"
        output_xmp  = preset_name.replace(" ", "_") + ".xmp"

        if not os.path.isfile(image_path):
            print(f"❌  Fichier introuvable : {image_path}")
            sys.exit(1)

        params = run_full_analysis(image_path)

    print(f"\n🛠   Génération du preset : {preset_name}")
    xmp_path = generate_xmp_file(params, preset_name, output_xmp)
    print(f"✅  Fichier XMP créé     : {xmp_path}")
    print(f"\n💡  Importez dans Lightroom Classic :\n"
          f"    Développement → Presets → icône (+) → Importer les préréglages\n")


if __name__ == "__main__":
    main()
