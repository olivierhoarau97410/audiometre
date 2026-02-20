# -*- coding: utf-8 -*-
"""
Audiomètre V3 — Streamlit (port de audiometreV2.py)
----------------------------------------------------
Test d'audition simple (non médical).

Dépendances :
    pip install streamlit>=1.28 numpy matplotlib

Lancer :
    streamlit run audiometreV3.py

Déploiement :
    Streamlit Community Cloud — https://streamlit.io/cloud
"""

import io
import wave
import random
import re
import unicodedata

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (obligatoire côté serveur)
import matplotlib.pyplot as plt
import streamlit as st

# ── Constantes (identiques à V2) ────────────────────────────────────────────
SAMPLE_RATE        = 44100
DURATION_S         = 1.0
FREQUENCIES        = [125, 250, 500, 1000, 2000, 4000, 8000]
LEVELS_DB          = [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
MAX_DB_FOR_SCALING = 80
PEAK_AT_MAX_DB     = 0.9
FADE_SEC           = 0.01  # fondu 10 ms anti-clic

MODE_OPTIONS = {
    "Deux oreilles (L+R)": "LR",
    "Oreille gauche (L)":  "L",
    "Oreille droite (R)":  "R",
}
MODE_LABEL = {v: k for k, v in MODE_OPTIONS.items()}


# ── Fonctions audio (réutilisées verbatim depuis V2) ─────────────────────────

def level_to_amplitude(db: int) -> float:
    """Niveau dB relatif → amplitude linéaire (0..1)."""
    return PEAK_AT_MAX_DB * (10 ** ((db - MAX_DB_FOR_SCALING) / 20.0))


def make_tone(freq: int, db: int, ear: str = "B",
              duration_s: float = DURATION_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Génère un bip sinusoïdal stéréo float32 avec enveloppe.
    ear : 'L' (gauche), 'R' (droite), 'B' (bilatéral)
    """
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    amp = level_to_amplitude(db)
    y = amp * np.sin(2 * np.pi * freq * t)

    fade_len = max(1, int(FADE_SEC * sr))
    if fade_len * 2 < y.size:
        fade = np.linspace(0, 1, fade_len)
        env = np.ones_like(y)
        env[:fade_len] = fade
        env[-fade_len:] = fade[::-1]
        y = y * env

    left  = y if ear in ("L", "B") else np.zeros_like(y)
    right = y if ear in ("R", "B") else np.zeros_like(y)
    return np.column_stack([left, right]).astype(np.float32)


def _sanitize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.strip("_")


# ── Encodage WAV (stdlib wave, pas de scipy) ─────────────────────────────────

def make_wav_bytes(freq: int, db: int, ear: str = "B") -> bytes:
    """Génère un fichier WAV stéréo 16-bit en mémoire (bytes)."""
    stereo_f32 = make_tone(freq, db, ear)
    stereo_i16 = np.clip(stereo_f32, -1.0, 1.0)
    stereo_i16 = (stereo_i16 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)          # 16 bits
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo_i16.tobytes())
    buf.seek(0)
    return buf.read()


# ── Audiogramme ───────────────────────────────────────────────────────────────

def draw_audiogram(results: dict, user_name: str, mode_code: str) -> plt.Figure:
    """Retourne une Figure matplotlib (axe Y inversé, convention audiologique)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Seuil auditif (dB)")
    ax.set_xticks(FREQUENCIES)
    ax.set_xticklabels(["125", "250", "500", "1k", "2k", "4k", "8k"])
    ax.set_yticks(LEVELS_DB)
    ax.set_ylim(85, -15)
    ax.set_xlim(100, 9500)
    ax.grid(True, linestyle=":", linewidth=0.8)

    who = user_name.strip()
    mode_lbl = MODE_LABEL.get(mode_code, mode_code)
    title = f"Audiogramme – {who} – {mode_lbl}" if who else f"Audiogramme – {mode_lbl}"
    ax.set_title(title)

    style_map = {
        "L": ("Gauche", "o", "tab:blue"),
        "R": ("Droite", "s", "tab:orange"),
    }
    legend_handles = []

    for ear in results:
        label, marker, color = style_map[ear]
        xs, ys = [], []
        for f in FREQUENCIES:
            v = results[ear][f]
            if v is not None:
                xs.append(f)
                ys.append(v)
        if xs:
            line, = ax.plot(xs, ys, marker=marker, linestyle="-",
                            color=color, label=label, linewidth=2, markersize=8)
            legend_handles.append(line)
        for f in FREQUENCIES:
            if results[ear][f] is None:
                ax.plot([f], [80], marker="x", markersize=12,
                        linestyle="none", color="red", markeredgewidth=2)

    if legend_handles:
        ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


# ── Gestion de l'état ─────────────────────────────────────────────────────────

def _empty_results(ears):
    return {ear: {f: None for f in FREQUENCIES} for ear in ears}


def init_state(user_name: str, mode_code: str):
    ears = list(mode_code)  # "LR"→["L","R"], "L"→["L"], "R"→["R"]
    sequence = [
        (ear, freq, db)
        for ear in ears
        for freq in FREQUENCIES
        for db in LEVELS_DB
    ]
    random.shuffle(sequence)
    st.session_state.update({
        "phase":         "testing",
        "user_name":     user_name,
        "mode_code":     mode_code,
        "ears_used":     ears,
        "sequence":      sequence,
        "idx":           0,
        "results":       _empty_results(ears),
        "current_trial": sequence[0],
        "audio_key":     0,
    })


def record_answer(heard: bool):
    ear, freq, db = st.session_state.current_trial
    if heard:
        prev = st.session_state.results[ear][freq]
        if prev is None or db < prev:
            st.session_state.results[ear][freq] = db
    st.session_state.idx += 1
    st.session_state.audio_key += 1
    if st.session_state.idx >= len(st.session_state.sequence):
        st.session_state.phase = "done"
    else:
        st.session_state.current_trial = st.session_state.sequence[
            st.session_state.idx
        ]


def reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ── Pages ─────────────────────────────────────────────────────────────────────

def render_home():
    st.title("🎧 Audiomètre")

    st.info(
        "**Avant de commencer :**\n\n"
        "- Munissez-vous d'un casque ou d'écouteurs\n"
        "- Placez-vous dans une pièce calme\n"
        "- Réglez le volume de votre ordinateur à **70 %** et n'y touchez plus pendant le test"
    )

    st.warning(
        "⚠️ Ce test **n'a pas de valeur médicale** mais permet d'évaluer l'acuité auditive. "
        "S'il suggère une gêne ou une perte, consultez un spécialiste."
    )

    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("Votre prénom :", placeholder="facultatif")
    with col2:
        mode_label = st.selectbox("Mode de test :", list(MODE_OPTIONS.keys()))

    st.markdown("---")
    if st.button("▶ Démarrer le test", use_container_width=True, type="primary"):
        init_state(user_name.strip(), MODE_OPTIONS[mode_label])
        st.rerun()


def render_testing():
    ear, freq, db = st.session_state.current_trial
    idx   = st.session_state.idx
    total = len(st.session_state.sequence)

    ear_label = {"L": "Oreille gauche 🔵", "R": "Oreille droite 🟠"}[ear]

    st.title("🎧 Audiomètre")
    st.progress(idx / total, text=f"Essai {idx + 1} / {total}")

    st.markdown(f"### {ear_label} — {freq} Hz — {db} dB")
    st.caption("Le son va être joué automatiquement. Écoutez attentivement.")

    # Audio widget — on encode le WAV en base64 et on injecte un <audio autoplay>
    # via st.components pour forcer la lecture à chaque essai sans dépendre du
    # paramètre key= (non supporté par toutes les versions de Streamlit)
    import base64
    import streamlit.components.v1 as components
    wav_bytes = make_wav_bytes(freq, db, ear)
    b64 = base64.b64encode(wav_bytes).decode()
    components.html(
        f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        height=0,
    )
    st.audio(wav_bytes, format="audio/wav")

    st.markdown("---")
    st.markdown("### Avez-vous entendu ce son ?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ J'ai entendu",
                     key=f"heard_{idx}",
                     use_container_width=True,
                     type="primary"):
            record_answer(True)
            st.rerun()
    with col2:
        if st.button("❌ Je n'ai pas entendu",
                     key=f"nothrd_{idx}",
                     use_container_width=True):
            record_answer(False)
            st.rerun()


def render_done():
    st.title("🎧 Audiomètre — Résultats")
    st.success("Mesure terminée ! Voici votre audiogramme.")

    results   = st.session_state.results
    user_name = st.session_state.user_name
    mode_code = st.session_state.mode_code

    # Audiogramme
    fig = draw_audiogram(results, user_name, mode_code)
    st.pyplot(fig)

    # Export PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close(fig)

    who_safe = _sanitize_name(user_name) if user_name else "audiometrie"
    st.download_button(
        label="📥 Télécharger l'audiogramme (PNG)",
        data=buf,
        file_name=f"audiogramme_{who_safe}.png",
        mime="image/png",
        use_container_width=True,
    )

    st.markdown("---")

    # Tableau récapitulatif des seuils
    st.markdown("#### Seuils auditifs détectés")
    freq_labels = ["125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz", "8 kHz"]
    ear_labels  = {"L": "Gauche (dB)", "R": "Droite (dB)"}

    header = "| Fréquence | " + " | ".join(
        ear_labels[e] for e in results
    ) + " |"
    sep = "|---|" + "---|" * len(results)
    rows = []
    for i, f in enumerate(FREQUENCIES):
        vals = []
        for ear in results:
            v = results[ear][f]
            vals.append(str(v) if v is not None else "non entendu")
        rows.append("| " + freq_labels[i] + " | " + " | ".join(vals) + " |")
    st.markdown("\n".join([header, sep] + rows))

    st.markdown("---")
    if st.button("🔄 Recommencer un nouveau test", use_container_width=True):
        reset_state()
        st.rerun()


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Audiomètre",
        page_icon="🎧",
        layout="centered",
    )
    phase = st.session_state.get("phase", "home")
    if phase == "home":
        render_home()
    elif phase == "testing":
        render_testing()
    elif phase == "done":
        render_done()


if __name__ == "__main__":
    main()
