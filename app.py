"""
================================================================================
  Interface Streamlit — Lightroom Preset Generator
================================================================================
  Lancement : streamlit run app.py
  Dépendances : streamlit, Pillow, numpy (pip install streamlit Pillow numpy)
================================================================================
"""

import io
import os
import tempfile
import streamlit as st
from PIL import Image

# Import des fonctions d'analyse depuis le script principal
from lightroom_preset_generator import (
    run_full_analysis,
    generate_xmp_file,
    DEMO_PARAMS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lightroom Preset Generator",
    page_icon="📷",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
#  EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────

st.title("📷 Lightroom Preset Generator")
st.caption(
    "Télécharge une photo d'inspiration → l'application analyse ses caractéristiques "
    "visuelles → génère un preset `.xmp` prêt à importer dans Lightroom Classic."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  ZONE D'UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("1️⃣  Importer une image")

    preset_name = st.text_input(
        "Nom du preset",
        value="MonPreset",
        placeholder="Ex : Golden Hour, Cinematic Blue...",
        help="Ce nom apparaîtra dans Lightroom après import.",
    )

    uploaded_file = st.file_uploader(
        "Choisir une photo (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="L'image n'est pas envoyée sur un serveur — tout se passe localement.",
    )

    use_demo = st.checkbox(
        "Utiliser les valeurs de démonstration (sans image)",
        value=False,
    )

with col_preview:
    st.subheader("Aperçu de l'image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=uploaded_file.name)
    else:
        st.info("Aucune image importée pour l'instant.")


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  BOUTON D'ANALYSE
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("2️⃣  Analyser et générer le preset")

if st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True):

    # ── Vérifications ────────────────────────────────────────────────────────
    if uploaded_file is None and not use_demo:
        st.warning("⚠️ Importe une image ou coche l'option démonstration.")
        st.stop()

    if not preset_name.strip():
        st.warning("⚠️ Donne un nom à ton preset.")
        st.stop()

    # ── Analyse ──────────────────────────────────────────────────────────────
    with st.spinner("Analyse en cours…"):

        if use_demo:
            params = DEMO_PARAMS
            st.info("Mode démonstration — valeurs d'exemple utilisées.")
        else:
            # Sauvegarde temporaire de l'image uploadée sur le disque
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                params = run_full_analysis(tmp_path)
            finally:
                os.unlink(tmp_path)  # Nettoyage du fichier temporaire

    st.success("✅ Analyse terminée !")


    # ─────────────────────────────────────────────────────────────────────────
    #  AFFICHAGE DES RÉSULTATS
    # ─────────────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("3️⃣  Paramètres estimés")

    # ── Ligne 1 : Tonalité de base ───────────────────────────────────────────
    st.markdown("**Tonalité de base**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Exposition",    params["Exposure2012"])
    c2.metric("Contraste",     params["Contrast2012"])
    c3.metric("Hautes lumières", params["Highlights2012"])
    c4.metric("Ombres",        params["Shadows2012"])
    c5.metric("Blancs",        params["Whites2012"])
    c6.metric("Noirs",         params["Blacks2012"])

    st.markdown("")

    # ── Ligne 2 : Balance des blancs + Présence ──────────────────────────────
    col_wb, col_pres = st.columns(2, gap="large")

    with col_wb:
        st.markdown("**Balance des blancs**")
        w1, w2 = st.columns(2)
        w1.metric("Température", params["Temperature"])
        w2.metric("Teinte",      params["Tint"])

    with col_pres:
        st.markdown("**Présence**")
        p1, p2 = st.columns(2)
        p1.metric("Vibrance",    params["Vibrance"])
        p2.metric("Saturation",  params["Saturation"])

    st.markdown("")

    # ── Ligne 3 : Color Grading ──────────────────────────────────────────────
    st.markdown("**Color Grading**")
    cg = params["ColorGrading"]
    cg1, cg2, cg3 = st.columns(3)

    with cg1:
        st.markdown("🌑 **Ombres (Shadows)**")
        st.write(f"Teinte : `{int(cg['Shadow']['Hue'])}°`")
        st.write(f"Saturation : `{cg['Shadow']['Sat']}`")
        st.write(f"Luminance : `{cg['Shadow']['Lum']}`")

    with cg2:
        st.markdown("🌗 **Tons moyens (Midtones)**")
        st.write(f"Teinte : `{int(cg['Midtone']['Hue'])}°`")
        st.write(f"Saturation : `{cg['Midtone']['Sat']}`")
        st.write(f"Luminance : `{cg['Midtone']['Lum']}`")

    with cg3:
        st.markdown("☀️ **Hautes lumières (Highlights)**")
        st.write(f"Teinte : `{int(cg['Highlight']['Hue'])}°`")
        st.write(f"Saturation : `{cg['Highlight']['Sat']}`")
        st.write(f"Luminance : `{cg['Highlight']['Lum']}`")


    # ─────────────────────────────────────────────────────────────────────────
    #  GÉNÉRATION ET TÉLÉCHARGEMENT DU FICHIER XMP
    # ─────────────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("4️⃣  Télécharger le preset")

    xmp_filename = preset_name.strip().replace(" ", "_") + ".xmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xmp", mode="w") as tmp_xmp:
        tmp_xmp_path = tmp_xmp.name

    generate_xmp_file(params, preset_name.strip(), tmp_xmp_path)

    with open(tmp_xmp_path, "r", encoding="utf-8") as f:
        xmp_bytes = f.read().encode("utf-8")

    os.unlink(tmp_xmp_path)

    st.download_button(
        label=f"⬇️  Télécharger {xmp_filename}",
        data=xmp_bytes,
        file_name=xmp_filename,
        mime="application/octet-stream",
        use_container_width=True,
        type="primary",
    )

    st.caption(
        "**Comment importer dans Lightroom Classic** : "
        "Panneau Développement → Presets (colonne gauche) → icône **+** → "
        "**Importer les préréglages** → sélectionne le fichier `.xmp`."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PIED DE PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption("MVP · Lightroom Preset Generator — traitement 100% local, aucune donnée envoyée.")
