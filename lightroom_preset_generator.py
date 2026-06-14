"""
================================================================================
  Lightroom Preset Generator — Analyse d'image vers fichier .XMP
================================================================================
  Version : 2.0.0  (Session 2 — Courbe des tons, HSL, Netteté, Bruit)
  Dépendances : Pillow, NumPy (pip install Pillow numpy)

  Usage :
      python lightroom_preset_generator.py <chemin_image> [nom_preset]
================================================================================
"""

import sys
import os
import colorsys
import random
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — CHARGEMENT DE L'IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def load_image_as_array(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge une image et retourne :
    - rgb_array  : (H, W, 3) float32 normalisé [0.0, 1.0]
    - gray_array : (H, W)    luminosité perceptuelle ITU-R BT.601
    """
    img = Image.open(image_path).convert("RGB")
    rgb  = np.array(img, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return rgb, gray


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — PARAMÈTRES DE BASE (inchangés depuis v1)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_exposure(gray):
    mean_lum = float(np.mean(gray))
    return float(np.clip((mean_lum - 0.5) * 10.0, -5.0, 5.0))

def analyze_contrast(gray):
    return int(np.clip((0.18 - float(np.std(gray))) * 300, -100, 100))

def analyze_highlights(gray):
    return int(np.clip((0.85 - float(np.percentile(gray, 95))) * 400, -100, 100))

def analyze_shadows(gray):
    return int(np.clip((0.10 - float(np.percentile(gray, 5))) * 500, -100, 100))

def analyze_whites(gray):
    return int(np.clip((0.92 - float(np.percentile(gray, 99))) * 300, -100, 100))

def analyze_blacks(gray):
    return int(np.clip((0.04 - float(np.percentile(gray, 1))) * 600, -100, 100))

def analyze_white_balance(rgb):
    r = float(np.mean(rgb[:, :, 0]))
    g = float(np.mean(rgb[:, :, 1]))
    b = float(np.mean(rgb[:, :, 2]))
    temperature = int(np.clip((1.0 - r / (b + 1e-6)) * 100, -100, 100))
    tint        = int(np.clip(-(g - (r + b) / 2.0) * 200, -150, 150))
    return temperature, tint

def analyze_vibrance_saturation(rgb):
    flat = rgb.reshape(-1, 3)[::10]
    sats = np.array([colorsys.rgb_to_hsv(float(r), float(g), float(b))[1]
                     for r, g, b in flat], dtype=np.float32)
    m = float(np.mean(sats))
    return int(np.clip((0.40 - m) * 150, -100, 100)), int(np.clip((0.45 - m) * 100, -80, 80))

def analyze_color_grading(rgb, gray):
    result = {}
    zones = {
        "Shadow":    gray < 0.25,
        "Midtone":   (gray >= 0.25) & (gray < 0.75),
        "Highlight": gray >= 0.75,
    }
    for name, mask in zones.items():
        pixels = rgb[mask]
        if len(pixels) < 50:
            result[name] = {"Hue": 0, "Sat": 0, "Lum": 0}
            continue
        sample = pixels[::max(1, len(pixels) // 1000)]
        hues, sats = [], []
        for r, g, b in sample:
            h, s, _ = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            if s > 0.05:
                hues.append(h * 360)
                sats.append(s)
        if hues:
            result[name] = {
                "Hue": round(float(np.mean(hues)), 1),
                "Sat": int(np.clip(float(np.mean(sats)) * 30, 0, 30)),
                "Lum": 0
            }
        else:
            result[name] = {"Hue": 0, "Sat": 0, "Lum": 0}
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — NOUVEAUTÉS SESSION 2
# ─────────────────────────────────────────────────────────────────────────────

# ── 3A. COURBE DES TONS (Point Curve) ────────────────────────────────────────

def analyze_tone_curve(gray: np.ndarray) -> dict:
    """
    Génère une courbe des tons en 5 points d'ancrage à partir de l'histogramme.

    Principe :
    - On calcule la luminosité médiane dans 5 zones tonales (noirs, ombres,
      milieu, hautes lumières, blancs).
    - On compare chaque médiane à une valeur de référence "neutre" (droite 1:1).
    - L'écart donne le déplacement du point de courbe (output ajusté).

    Chaque point = (input, output) en valeurs 0–255.
    """
    # Zones en percentiles de luminosité
    zones = {
        "blacks":     (0,  10),
        "shadows":    (10, 35),
        "midtones":   (35, 65),
        "highlights": (65, 90),
        "whites":     (90, 100),
    }

    # Points d'ancrage input fixes (positions sur l'axe X de la courbe)
    anchors_input = {
        "blacks":     0,
        "shadows":    64,
        "midtones":   128,
        "highlights": 192,
        "whites":     255,
    }

    curve_points = {}
    flat_gray = gray.flatten()

    for zone, (p_low, p_high) in zones.items():
        # Pixels appartenant à cette zone tonale
        low  = np.percentile(flat_gray, p_low)
        high = np.percentile(flat_gray, p_high)
        mask = (flat_gray >= low) & (flat_gray <= high)
        zone_pixels = flat_gray[mask]

        if len(zone_pixels) == 0:
            # Pas de pixels dans cette zone → point neutre
            inp = anchors_input[zone]
            curve_points[zone] = (inp, inp)
            continue

        median_val = float(np.median(zone_pixels))

        # Input fixe, output ajusté selon l'écart à la référence neutre
        inp = anchors_input[zone]
        ref = inp / 255.0  # valeur neutre attendue

        # Correction : si la zone est plus sombre que la ref → relever l'output
        delta = (median_val - ref) * 255.0
        output = int(np.clip(inp - delta * 0.6, 0, 255))

        curve_points[zone] = (inp, output)

    return curve_points


def format_tone_curve_xmp(curve_points: dict) -> str:
    """
    Formate les points de courbe au format XMP attendu par Lightroom :
    une liste de paires "input, output" séparées par des virgules.
    Ex : "0, 0, 64, 58, 128, 132, 192, 198, 255, 255"
    """
    ordered_zones = ["blacks", "shadows", "midtones", "highlights", "whites"]
    pairs = []
    for zone in ordered_zones:
        inp, out = curve_points[zone]
        pairs.append(f"{inp}, {out}")
    return ", ".join(pairs)


# ── 3B. HSL (Teinte / Saturation / Luminance par couleur) ────────────────────

# Définition des 8 couleurs Lightroom avec leurs plages de teinte en degrés HSV
LR_COLOR_RANGES = {
    "Red":     [(345, 360), (0, 15)],   # Rouge (plage circulaire)
    "Orange":  [(15, 45)],
    "Yellow":  [(45, 75)],
    "Green":   [(75, 150)],
    "Aqua":    [(150, 195)],
    "Blue":    [(195, 255)],
    "Purple":  [(255, 300)],
    "Magenta": [(300, 345)],
}


def _hue_in_range(hue_deg: float, ranges: list) -> bool:
    """Vérifie si une teinte (0–360°) appartient à une plage donnée."""
    for (lo, hi) in ranges:
        if lo <= hue_deg < hi:
            return True
    return False


def analyze_hsl(rgb: np.ndarray) -> dict:
    """
    Estime les corrections HSL par couleur pour Lightroom.

    Pour chaque couleur LR (Rouge, Orange, …) :
    - On isole les pixels dont la teinte HSV appartient à la plage de cette couleur
      et dont la saturation est > 0.10 (pixels colorés, pas gris)
    - On calcule la saturation moyenne et la luminosité moyenne de ces pixels
    - On en déduit une correction de Saturation et de Luminance

    Teinte (Hue)       : toujours 0 en v2 (correction fine, nécessite calibration)
    Saturation (Sat)   : si sous-saturé → boost ; si sur-saturé → réduction
    Luminance (Lum)    : si sombre → relever ; si trop clair → baisser légèrement
    """
    # Échantillonnage (1 pixel sur 5 pour équilibre précision/performance)
    flat_rgb = rgb.reshape(-1, 3)[::5]

    # Conversion HSV de tous les pixels échantillonnés
    hsv_pixels = np.array([
        colorsys.rgb_to_hsv(float(r), float(g), float(b))
        for r, g, b in flat_rgb
    ])  # shape (N, 3) : [hue 0-1, sat 0-1, val 0-1]

    hsl_params = {}

    for color_name, ranges in LR_COLOR_RANGES.items():
        # Masque : pixels dont la teinte correspond à cette couleur ET assez saturés
        color_mask = np.array([
            _hue_in_range(h * 360, ranges) and s > 0.10
            for h, s, v in hsv_pixels
        ])

        matching = hsv_pixels[color_mask]

        if len(matching) < 20:
            # Pas assez de pixels de cette couleur → corrections neutres
            hsl_params[color_name] = {"Hue": 0, "Sat": 0, "Lum": 0}
            continue

        mean_sat = float(np.mean(matching[:, 1]))   # saturation moyenne [0, 1]
        mean_val = float(np.mean(matching[:, 2]))   # valeur (luminosité) moyenne

        # Correction saturation : référence 0.50 (saturation "normale")
        sat_correction = int(np.clip((0.50 - mean_sat) * 120, -80, 80))

        # Correction luminance : référence 0.55
        lum_correction = int(np.clip((0.55 - mean_val) * 100, -70, 70))

        hsl_params[color_name] = {
            "Hue": 0,               # Correction teinte neutre (v2)
            "Sat": sat_correction,
            "Lum": lum_correction,
        }

    return hsl_params


# ── 3C. NETTETÉ ET RÉDUCTION DU BRUIT ────────────────────────────────────────

def analyze_sharpness(image_path: str) -> dict:
    """
    Estime la netteté via le filtre Laplacien appliqué à l'image en niveaux de gris.

    Principe :
    Le filtre Laplacien détecte les contours (transitions brusques de luminosité).
    La variance du résultat mesure la "quantité de détails fins" :
    - Variance élevée → image nette → peu de correction nécessaire
    - Variance faible → image floue → sharpening plus fort

    Paramètres LR générés :
    - Sharpness      : force du masque de netteté [0, 150]
    - SharpenRadius  : rayon du masque [0.5, 3.0]
    - SharpenDetail  : protection des détails [0, 100]
    - SharpenEdgeMasking : masque des bords [0, 100]
    """
    img_gray = Image.open(image_path).convert("L")  # niveaux de gris

    # Redimensionnement pour la performance (max 800px)
    w, h = img_gray.size
    if max(w, h) > 800:
        scale = 800 / max(w, h)
        img_gray = img_gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Application du filtre Laplacien
    laplacian = img_gray.filter(ImageFilter.Kernel(
        size=3,
        kernel=[-1, -1, -1,
                -1,  8, -1,
                -1, -1, -1],
        scale=1,
        offset=128
    ))

    lap_array = np.array(laplacian, dtype=np.float32)
    variance  = float(np.var(lap_array))

    # Mapping variance → paramètres de netteté
    # Variance typique : floue < 200, normale ~500, très nette > 1000
    if variance < 150:
        # Image très floue
        sharpness = 100
        radius    = 1.0
        detail    = 25
        masking   = 20
    elif variance < 400:
        # Image légèrement floue
        sharpness = 70
        radius    = 0.8
        detail    = 40
        masking   = 30
    elif variance < 800:
        # Image bien nette
        sharpness = 40
        radius    = 0.5
        detail    = 60
        masking   = 50
    else:
        # Image très nette / déjà sur-sharpened
        sharpness = 25
        radius    = 0.5
        detail    = 75
        masking   = 70

    return {
        "Sharpness":          sharpness,
        "SharpenRadius":      radius,
        "SharpenDetail":      detail,
        "SharpenEdgeMasking": masking,
        "_variance":          round(variance, 1),   # debug info
    }


def analyze_noise_reduction(gray: np.ndarray) -> dict:
    """
    Estime le bruit dans les zones sombres de l'image.

    Principe :
    Le bruit numérique apparaît principalement dans les ombres.
    On isole les pixels sombres (< 30% de luminosité) et on mesure
    l'écart-type local → forte variation dans les zones sombres = bruit.

    Paramètres LR générés :
    - LuminanceSmoothing    : réduction du bruit de luminance [0, 100]
    - LuminanceNoiseReductionDetail    : préservation des détails [0, 100]
    - ColorNoiseReduction   : réduction du bruit de couleur [0, 100]
    - ColorNoiseReductionDetail : détail bruit couleur [0, 100]
    """
    # Isoler les pixels sombres (zones où le bruit est le plus visible)
    dark_mask   = gray < 0.30
    dark_pixels = gray[dark_mask]

    if len(dark_pixels) < 100:
        # Pas assez de zones sombres → réduction minimale
        return {
            "LuminanceSmoothing":            10,
            "LuminanceNoiseReductionDetail": 50,
            "ColorNoiseReduction":           25,
            "ColorNoiseReductionDetail":     50,
        }

    # Bruit estimé via l'écart-type des pixels sombres
    noise_std = float(np.std(dark_pixels))

    # Mapping : std élevé → beaucoup de bruit → réduction forte
    # std typique : faible bruit < 0.03, moyen ~0.06, fort > 0.10
    luminance_nr = int(np.clip(noise_std * 600, 5, 80))
    color_nr     = int(np.clip(noise_std * 400, 10, 60))

    # Plus on applique de réduction, moins on préserve les détails fins
    lum_detail   = int(np.clip(80 - luminance_nr * 0.5, 20, 80))
    color_detail = 50

    return {
        "LuminanceSmoothing":            luminance_nr,
        "LuminanceNoiseReductionDetail": lum_detail,
        "ColorNoiseReduction":           color_nr,
        "ColorNoiseReductionDetail":     color_detail,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — ANALYSE COMPLÈTE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(image_path: str) -> dict:
    """Point d'entrée : analyse complète → dictionnaire de tous les paramètres LR."""
    print(f"\n📷  Analyse de : {os.path.basename(image_path)}")
    print("─" * 55)

    rgb, gray = load_image_as_array(image_path)

    # Paramètres de base
    exposure             = analyze_exposure(gray)
    contrast             = analyze_contrast(gray)
    highlights           = analyze_highlights(gray)
    shadows              = analyze_shadows(gray)
    whites               = analyze_whites(gray)
    blacks               = analyze_blacks(gray)
    temperature, tint    = analyze_white_balance(rgb)
    vibrance, saturation = analyze_vibrance_saturation(rgb)
    color_grading        = analyze_color_grading(rgb, gray)

    # Nouveautés Session 2
    tone_curve           = analyze_tone_curve(gray)
    hsl                  = analyze_hsl(rgb)
    sharpness            = analyze_sharpness(image_path)
    noise                = analyze_noise_reduction(gray)

    params = {
        # Base
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
        # Session 2
        "ToneCurve":      tone_curve,
        "HSL":            hsl,
        "Sharpness":      sharpness,
        "NoiseReduction": noise,
    }

    # Affichage console
    base_keys = ["Exposure2012","Contrast2012","Highlights2012","Shadows2012",
                 "Whites2012","Blacks2012","Temperature","Tint","Vibrance","Saturation"]
    print("  ── Tonalité & Balance ──")
    for k in base_keys:
        print(f"  {k:<28} {params[k]:>8}")

    print("\n  ── Courbe des tons ──")
    for zone, (inp, out) in tone_curve.items():
        print(f"  {zone:<20} input={inp:>3}  output={out:>3}")

    print("\n  ── HSL ──")
    for color, vals in hsl.items():
        print(f"  {color:<10} Hue={vals['Hue']:>4}  Sat={vals['Sat']:>4}  Lum={vals['Lum']:>4}")

    print("\n  ── Netteté ──")
    for k, v in sharpness.items():
        if not k.startswith("_"):
            print(f"  {k:<30} {v:>8}")

    print("\n  ── Réduction du bruit ──")
    for k, v in noise.items():
        print(f"  {k:<30} {v:>8}")

    return params


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — GÉNÉRATION DU FICHIER .XMP
# ─────────────────────────────────────────────────────────────────────────────

def _make_uuid() -> str:
    chars = "0123456789abcdef"
    uid = "".join(random.choices(chars, k=32))
    return f"{uid[:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:]}"


def _build_hsl_xmp(hsl: dict) -> str:
    """Génère les attributs XMP pour la section HSL."""
    colors = ["Red","Orange","Yellow","Green","Aqua","Blue","Purple","Magenta"]
    lines = []
    for c in colors:
        v = hsl.get(c, {"Hue": 0, "Sat": 0, "Lum": 0})
        lines.append(f'            crs:HueAdjustment{c}="{v["Hue"]}"')
        lines.append(f'            crs:SaturationAdjustment{c}="{v["Sat"]}"')
        lines.append(f'            crs:LuminanceAdjustment{c}="{v["Lum"]}"')
    return "\n".join(lines)


def generate_xmp_file(params: dict, preset_name: str, output_path: str) -> str:
    """Génère et écrit le fichier .xmp complet (v2) sur le disque."""
    cg    = params["ColorGrading"]
    tc    = params["ToneCurve"]
    sh    = params["Sharpness"]
    nr    = params["NoiseReduction"]
    now   = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    uid   = _make_uuid()

    # Courbe des tons formatée
    curve_str = format_tone_curve_xmp(tc)

    # Bloc HSL
    hsl_block = _build_hsl_xmp(params["HSL"])

    xmp_content = f"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about=""
            xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
            xmlns:xmp="http://ns.adobe.com/xap/1.0/"

            xmp:CreatorTool="LightroomPresetGenerator-MVP-v2"
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

            crs:ToneCurveName2012="Custom"
            crs:ToneCurvePV2012="{curve_str}"
            crs:ToneCurvePV2012Red="{curve_str}"
            crs:ToneCurvePV2012Green="{curve_str}"
            crs:ToneCurvePV2012Blue="{curve_str}"

{hsl_block}

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

            crs:Sharpness="{sh['Sharpness']}"
            crs:SharpenRadius="{sh['SharpenRadius']}"
            crs:SharpenDetail="{sh['SharpenDetail']}"
            crs:SharpenEdgeMasking="{sh['SharpenEdgeMasking']}"

            crs:LuminanceSmoothing="{nr['LuminanceSmoothing']}"
            crs:LuminanceNoiseReductionDetail="{nr['LuminanceNoiseReductionDetail']}"
            crs:ColorNoiseReduction="{nr['ColorNoiseReduction']}"
            crs:ColorNoiseReductionDetail="{nr['ColorNoiseReductionDetail']}"

            crs:ConvertToGrayscale="False"
            crs:EnableColorAdjustments="True"
            crs:EnableDetail="True"
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
#  SECTION 6 — VALEURS DE DÉMONSTRATION (v2)
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
    },
    # Nouveautés v2
    "ToneCurve": {
        "blacks":     (0,   5),
        "shadows":    (64,  58),
        "midtones":   (128, 132),
        "highlights": (192, 198),
        "whites":     (255, 250),
    },
    "HSL": {
        "Red":     {"Hue": 0, "Sat":  10, "Lum":  -5},
        "Orange":  {"Hue": 0, "Sat":  15, "Lum":   5},
        "Yellow":  {"Hue": 0, "Sat":   8, "Lum":  10},
        "Green":   {"Hue": 0, "Sat": -10, "Lum":   0},
        "Aqua":    {"Hue": 0, "Sat":   0, "Lum":   0},
        "Blue":    {"Hue": 0, "Sat":  20, "Lum": -10},
        "Purple":  {"Hue": 0, "Sat":   0, "Lum":   0},
        "Magenta": {"Hue": 0, "Sat":   5, "Lum":   0},
    },
    "Sharpness": {
        "Sharpness":          60,
        "SharpenRadius":      0.8,
        "SharpenDetail":      40,
        "SharpenEdgeMasking": 30,
    },
    "NoiseReduction": {
        "LuminanceSmoothing":            20,
        "LuminanceNoiseReductionDetail": 50,
        "ColorNoiseReduction":           25,
        "ColorNoiseReductionDetail":     50,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("\n⚠️  Aucune image fournie — utilisation des valeurs de démonstration.\n")
        params      = DEMO_PARAMS
        preset_name = "DemoPreset_MVP_v2"
        output_xmp  = "preset_demo_v2.xmp"
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
