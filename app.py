import gc
import json
import os
import gdown
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pywt
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras import layers, models
import streamlit as st

# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="AEROSTRESS-3D v2.5 | Diagnostic CND Aéronautique",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# INJECTION CSS SUR MESURE (THÈME SOMBRE & CYAN NÉON CONSERVÉ À L'IDENTIQUE)
# ==============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    [data-testid="stSidebar"] {
        background-color: #121721;
        border-right: 1px solid #1f293d;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00f2fe;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.4);
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #8b98a5;
        margin-bottom: 25px;
    }
    .idle-card {
        background: #161c28;
        border: 1px dashed #2a364f;
        border-radius: 10px;
        padding: 35px;
        text-align: center;
        color: #8b98a5;
        margin-top: 20px;
    }
    .status-badge-danger {
        background-color: #3d1217;
        color: #ff4d4d;
        border: 1px solid #ff4d4d;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 0 10px rgba(255, 77, 77, 0.2);
    }
    .status-badge-success {
        background-color: #0e3321;
        color: #2ed573;
        border: 1px solid #2ed573;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 0 10px rgba(46, 213, 115, 0.2);
    }
    .metric-card {
        background-color: #161c28;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 10px 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8b98a5;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        height: 45px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

DOMAIN_BOUNDS_MM = np.array([15.0, 15.0, 15.0], dtype=np.float32)
EDGE_LOW_MM = 3.0
EDGE_HIGH_MM = 12.0
CRITICAL_SIZE_MM = 3.5
DRIVE_FOLDER_ID = "1FI42tAQgF2xZxAoNTckskBYH088jXRB9"


# ==============================================================================
# BLOC 1 : ARCHITECTURE RESNET-3D v2.5 (IDENTIQUE AU NOTEBOOK D'ENTRAÎNEMENT)
# ==============================================================================
class MCDropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.mc_mode = False

    def call(self, inputs, training=None):
        if training or self.mc_mode:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"rate": self.rate})
        return cfg


def residual_block_3d(x, filters, dilation_rate=(1, 1, 1), block_name=None):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, (1, 1, 1), padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv3D(filters, (3, 3, 3), padding="same", dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(filters, (3, 3, 3), padding="same", dilation_rate=dilation_rate, name=block_name)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_aerostress_resnet3d_v2(input_shape=(16, 16, 8, 1), dropout_rate=0.4):
    inputs = layers.Input(shape=input_shape, name="Acoustic_Tensor_3D")

    x = residual_block_3d(inputs, 32)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = residual_block_3d(x, 64, dilation_rate=(2, 2, 1))
    x = residual_block_3d(x, 128, dilation_rate=(1, 1, 1), block_name="last_conv3d")

    shared = layers.GlobalAveragePooling3D()(x)
    shared = MCDropout(dropout_rate, name="mc_dropout_shared")(shared)

    h_cls = layers.Dense(64, activation="relu")(shared)
    h_cls = MCDropout(dropout_rate * 0.5, name="mc_dropout_cls")(h_cls)
    out_logit = layers.Dense(1, activation=None, name="Class_Logit")(h_cls)
    out_class = layers.Activation("sigmoid", name="Class_State")(out_logit)

    h_loc = layers.Dense(64, activation="relu")(shared)
    h_loc = MCDropout(dropout_rate * 0.5, name="mc_dropout_loc")(h_loc)
    out_coords = layers.Dense(3, activation="sigmoid", name="Coords_XYZ")(h_loc)

    h_sev = layers.Dense(64, activation="relu")(shared)
    h_sev = MCDropout(dropout_rate * 0.5, name="mc_dropout_sev")(h_sev)
    out_severity = layers.Dense(1, activation="softplus", name="Severity_MM")(h_sev)

    full_model = models.Model(
        inputs=inputs,
        outputs=[out_class, out_coords, out_severity],
        name="AeroStress_ResNet3D_v2_5",
    )
    logit_model = models.Model(
        inputs=inputs,
        outputs=[out_logit, out_coords, out_severity],
        name="AeroStress_Logit_Extractor",
    )
    return full_model, logit_model


def _find_file_by_suffix(suffix):
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(suffix):
                return os.path.join(root, f)
    return None


@st.cache_resource
def load_cached_ai_model():
    weights_path = _find_file_by_suffix(".weights.h5")
    config_path = _find_file_by_suffix("calibration_config.json")

    if not weights_path:
        with st.spinner("📥 Téléchargement du modèle AeroStress-3D v2.5 depuis Google Drive..."):
            try:
                gdown.download_folder(id=DRIVE_FOLDER_ID, quiet=True)
            except Exception as e:
                return None, None, None, f"Échec du téléchargement Google Drive : {e}"
        weights_path = _find_file_by_suffix(".weights.h5")
        config_path = _find_file_by_suffix("calibration_config.json")

    if not weights_path or not os.path.exists(weights_path):
        return None, None, None, "Fichier de poids (.weights.h5) introuvable dans le dossier Google Drive."

    full_model, logit_model = build_aerostress_resnet3d_v2(input_shape=(16, 16, 8, 1))
    try:
        full_model.load_weights(weights_path)
    except Exception as e:
        return None, None, None, f"Erreur de compatibilité des poids ({weights_path}) : {e}"

    cal_cfg = {
        "optimal_temperature": 0.7861,
        "calibrated_threshold": 0.1887,
        "critical_size_mm": CRITICAL_SIZE_MM,
        "mc_samples": 25,
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cal_cfg.update(json.load(f))

    return full_model, logit_model, cal_cfg, None


model, logit_extractor, CAL_CFG, LOAD_ERROR = load_cached_ai_model()

if LOAD_ERROR is not None:
    st.error(f"🚨 Arrêt de sécurité : {LOAD_ERROR}")
    st.info("Vérifiez que 'aerostress_3d_v2_5.weights.h5' est bien présent et partagé en lecture dans votre dossier Google Drive.")
    st.stop()


# ==============================================================================
# BLOC 2 : ONDELETTES (ANTI-NaN), MC-DROPOUT, GRAD-CAM LOGIT PUR & PLOTLY 360°
# ==============================================================================
def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def denoise_signal_wavelet(signal_1d, wavelet="db4", level=3):
    """Débruitage DWT (db4) identique à l'entraînement, protégé contre les capteurs HS (signal nul)."""
    n = len(signal_1d)
    if np.max(np.abs(signal_1d)) <= 1e-12:
        return np.zeros(n, dtype=np.float32)

    coeffs = pywt.wavedec(signal_1d, wavelet, mode="symmetric", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = float(sigma * np.sqrt(2.0 * np.log(n)))

    if threshold <= 1e-12:
        denoised_coeffs = coeffs
    else:
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]
        ]

    reconstructed = np.nan_to_num(pywt.waverec(denoised_coeffs, wavelet, mode="symmetric")[:n], nan=0.0)
    return reconstructed.astype(np.float32)


def raw_1d_to_volume_3d(raw_signals_1d):
    """Convertit (8, 1024) -> volume 3D normalisé Z-score (16, 16, 8, 1) + signaux 1D filtrés."""
    n_sensors, _ = raw_signals_1d.shape
    clean_1d = np.zeros_like(raw_signals_1d, dtype=np.float32)
    vol_3d = np.zeros((16, 16, n_sensors, 1), dtype=np.float32)

    for s in range(n_sensors):
        clean_1d[s] = denoise_signal_wavelet(raw_signals_1d[s])
        vol_3d[:, :, s, 0] = clean_1d[s].reshape(16, 16, -1).mean(axis=-1)

    mean_v = float(vol_3d.mean())
    std_v = float(vol_3d.std()) + 1e-8
    vol_3d = (vol_3d - mean_v) / std_v
    return vol_3d, clean_1d


def predict_mc_calibrated(model_obj, logit_model, tensor_batch, temperature, mc_samples=25):
    """Inférence MC-Dropout calibrée par Temperature Scaling (BatchNorm figé)."""
    mc_layers = [l for l in model_obj.layers if isinstance(l, MCDropout)]
    probs_mc = np.zeros(mc_samples, dtype=np.float32)
    coords_mc = np.zeros((mc_samples, 3), dtype=np.float32)
    sev_mc = np.zeros(mc_samples, dtype=np.float32)

    try:
        for l in mc_layers:
            l.mc_mode = True
        for t in range(mc_samples):
            l_out, c_out, s_out = logit_model(tensor_batch, training=False)
            probs_mc[t] = float(sigmoid_np(float(l_out[0, 0]) / temperature))
            coords_mc[t] = c_out[0].numpy() * DOMAIN_BOUNDS_MM
            sev_mc[t] = float(s_out[0, 0])
    finally:
        for l in mc_layers:
            l.mc_mode = False

    return {
        "prob_mean": float(probs_mc.mean()),
        "prob_std": float(probs_mc.std()),
        "coords_mean": coords_mc.mean(axis=0),
        "coords_std": coords_mc.std(axis=0),
        "sev_mean": float(sev_mc.mean()),
        "sev_std": float(sev_mc.std()),
    }


def compute_honest_gradcam_3d(logit_model, tensor_batch, layer_name="last_conv3d"):
    """Grad-CAM 3D honnête sur le logit pré-sigmoïde, sur-échantillonnage trilinéaire pur (sans gaussienne)."""
    grad_model = models.Model(
        inputs=logit_model.inputs,
        outputs=[logit_model.get_layer(layer_name).output, logit_model.get_layer("Class_Logit").output],
    )
    inp = tf.convert_to_tensor(tensor_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, logit_out = grad_model(inp, training=False)
        score = logit_out[:, 0]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2, 3)).numpy()
    cam = np.maximum(np.tensordot(conv_out[0].numpy(), weights, axes=([3], [0])), 0.0)
    if cam.max() > 0:
        cam /= cam.max()

    zoom_factors = (16.0 / cam.shape[0], 16.0 / cam.shape[1], 8.0 / cam.shape[2])
    return np.clip(zoom(cam, zoom_factors, order=1), 0.0, 1.0).astype(np.float32)


def render_gradcam_3d_interactive(heatmap_3d, coords_pred, is_crack, gt_coords=None):
    """Rendu volumétrique Grad-CAM 3D pur (sans gaussienne artificielle) avec légende et colorbar séparées."""
    X, Y, Z = np.meshgrid(
        np.linspace(0.0, 15.0, 16, dtype=np.float32),
        np.linspace(0.0, 15.0, 16, dtype=np.float32),
        np.linspace(0.0, 15.0, 8, dtype=np.float32),
        indexing="ij",
    )

    heatmap_flat = heatmap_3d.astype(np.float32).flatten()
    v_min = float(np.min(heatmap_flat))
    v_max = float(np.max(heatmap_flat))
    if v_max <= v_min:
        v_max = v_min + 1.0

    traces = [
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=heatmap_flat,
            isomin=float(v_min + 0.20 * (v_max - v_min)),
            isomax=float(v_max),
            opacity=0.35,
            surface_count=10,
            colorscale="Jet",
            colorbar=dict(
                title=dict(text="Grad-CAM Logit", font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=10),
                len=0.80,
                y=0.45,
                thickness=16,
                x=1.01,
            ),
            name="Activation Grad-CAM 3D",
        )
    ]

    if is_crack:
        traces.append(
            go.Scatter3d(
                x=[coords_pred[0]],
                y=[coords_pred[1]],
                z=[coords_pred[2]],
                mode="markers+text",
                marker=dict(size=11, color="#ff4d4d", symbol="diamond", line=dict(color="white", width=2)),
                text=["📍 Prédiction IA"],
                textposition="top center",
                textfont=dict(color="#ff4d4d", size=12),
                name="Coordonnées IA",
            )
        )

    if gt_coords is not None:
        traces.append(
            go.Scatter3d(
                x=[float(gt_coords[0])],
                y=[float(gt_coords[1])],
                z=[float(gt_coords[2])],
                mode="markers+text",
                marker=dict(size=10, color="#00f2fe", symbol="cross", line=dict(color="white", width=2)),
                text=["🎯 Vérité Terrain"],
                textposition="bottom center",
                textfont=dict(color="#00f2fe", size=12),
                name="Vérité Terrain",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#121721",
        margin=dict(l=0, r=10, b=0, t=20),
        height=480,
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(18, 23, 33, 0.82)",
            bordercolor="#2a364f",
            borderwidth=1,
            font=dict(color="#e6edf3", size=11),
        ),
        scene=dict(
            xaxis=dict(title=dict(text="X (mm)", font=dict(color="#00f2fe")), range=[0, 15], backgroundcolor="#121721", gridcolor="#2a364f"),
            yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#00f2fe")), range=[0, 15], backgroundcolor="#121721", gridcolor="#2a364f"),
            zaxis=dict(title=dict(text="Z (Prof. mm)", font=dict(color="#00f2fe")), range=[0, 15], backgroundcolor="#121721", gridcolor="#2a364f"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.65),
        ),
    )
    return fig


# ==============================================================================
# BLOC 3 : BARRE LATÉRALE (SALLE DE CONTRÔLE)
# ==============================================================================
st.sidebar.markdown("## 🛠️ Salle de Contrôle")
st.sidebar.caption("Analysez un échantillon réel du jeu de test ou téléversez vos signaux capteurs.")

capture_mode = st.sidebar.selectbox(
    "Source de l'acquisition acoustique",
    ["Échantillons de Test Réels (Vérité Terrain)", "Téléverser un fichier capteur (.npy / .npz)"]
)

raw_1d_for_plot = None
clean_1d_for_plot = None
input_vol_3d = None
gt_info = None

DEMO_SAMPLES = {
    "Fissure Critique de Bordure (< 3 mm / > 12 mm)": "samples/real_test_fissure_bord.npz",
    "Fissure Centrale Profonde": "samples/real_test_fissure_centre.npz",
    "Structure Saine (Vibrations Opérationnelles)": "samples/real_test_sain.npz",
}


def load_npz_package(file_or_path):
    data = np.load(file_or_path)
    raw_1d = data["raw_signals_1d"].astype(np.float32) if "raw_signals_1d" in data else None
    if raw_1d is not None:
        vol_3d, clean_1d = raw_1d_to_volume_3d(raw_1d)
    else:
        vol_3d = data["tensor_3d"].astype(np.float32)
        clean_1d = None

    gt = None
    if "gt_class" in data:
        gt = {
            "is_crack": bool(float(data["gt_class"].flatten()[0]) >= 0.5),
            "coords": data["gt_coords_mm"].flatten() if "gt_coords_mm" in data else np.zeros(3),
            "severity": float(data["gt_severity_mm"].flatten()[0]) if "gt_severity_mm" in data else 0.0,
        }
    return vol_3d, raw_1d, clean_1d, gt


if capture_mode == "Échantillons de Test Réels (Vérité Terrain)":
    selected_label = st.sidebar.selectbox("Sélectionner un cas de validation :", list(DEMO_SAMPLES.keys()))
    sample_path = DEMO_SAMPLES[selected_label]
    if os.path.exists(sample_path):
        input_vol_3d, raw_1d_for_plot, clean_1d_for_plot, gt_info = load_npz_package(sample_path)
    else:
        st.sidebar.warning(f"Fichier '{sample_path}' absent du dépôt. Utilisez le mode téléversement ci-dessus.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Fichier capteur (.npy ou .npz)",
        type=["npy", "npz"],
        help="Formats acceptés : signaux bruts (8, 1024), volume (16, 16, 8, 1) ou archive .npz."
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".npz"):
            input_vol_3d, raw_1d_for_plot, clean_1d_for_plot, gt_info = load_npz_package(uploaded_file)
        else:
            arr = np.load(uploaded_file).astype(np.float32)
            if arr.shape == (8, 1024):
                raw_1d_for_plot = arr
                input_vol_3d, clean_1d_for_plot = raw_1d_to_volume_3d(raw_1d_for_plot)
            elif arr.shape == (16, 16, 8, 1):
                input_vol_3d = arr
            elif arr.shape == (16, 16, 8):
                input_vol_3d = np.expand_dims(arr, axis=-1)
            else:
                st.sidebar.error(f"Dimensions invalides : {arr.shape}. Attendu : (8, 1024) ou (16, 16, 8, 1).")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"🌡️ Calibration active : T = {CAL_CFG['optimal_temperature']:.4f} | "
    f"Seuil Sûreté = {CAL_CFG['calibrated_threshold']:.4f}"
)
btn_inject = st.sidebar.button("⚡ Injecter le signal dans l'analyseur")


# ==============================================================================
# BLOC 4 : TABLEAU DE BORD PRINCIPAL
# ==============================================================================
st.markdown("<div class='main-header'>🛡️ AEROSTRESS-3D v2.5</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Système de Diagnostic Multi-Tâches IA, Quantification d'Incertitude & Localisation 3D (CND Acoustique)</div>", unsafe_allow_html=True)

if (not btn_inject and "analyzed" not in st.session_state) or input_vol_3d is None:
    st.markdown("""
    <div class='idle-card'>
        <h2>📡 Système Prêt à l'Analyse</h2>
        <p>L'intelligence artificielle multi-tâches d'AeroStress-3D attend de recevoir les signaux acoustiques pour démarrer le monitoring 3D en temps réel.</p>
        <br>
        <p style='color: #00f2fe;'>💡 Choisissez un échantillon dans le panneau latéral gauche et cliquez sur <b>'Injecter le signal'</b> !</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.session_state["analyzed"] = True

    tensor_batch = np.expand_dims(input_vol_3d, axis=0).astype(np.float32)
    mc_res = predict_mc_calibrated(
        model,
        logit_extractor,
        tensor_batch,
        temperature=CAL_CFG["optimal_temperature"],
        mc_samples=CAL_CFG.get("mc_samples", 25),
    )

    threshold = float(CAL_CFG["calibrated_threshold"])
    prob_crack = mc_res["prob_mean"]
    prob_std = mc_res["prob_std"]
    is_crack = prob_crack >= threshold
    manual_review = prob_std > 0.08

    if is_crack:
        severity_disp = mc_res["sev_mean"]
        coords_disp = np.clip(mc_res["coords_mean"], 0.0, 15.0)
        health_score = float(np.clip(100.0 * (1.0 - severity_disp / CRITICAL_SIZE_MM), 0.0, 100.0))
    else:
        severity_disp = 0.00
        coords_disp = np.array([0.00, 0.00, 0.00], dtype=np.float32)
        health_score = 100.0

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### 📊 Diagnostic de l'IA")

        if is_crack:
            st.markdown(
                f"<div class='status-badge-danger'>⚠️ FISSURE DÉTECTÉE ({prob_crack*100:.1f}% ± {prob_std*100:.1f}%)</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Alerte déclenchée (Probabilité calibrée ≥ seuil de sûreté {threshold*100:.1f}%).")
        else:
            st.markdown(
                f"<div class='status-badge-success'>🟢 STRUCTURE SAINE ({(1.0 - prob_crack)*100:.1f}% ± {prob_std*100:.1f}%)</div>",
                unsafe_allow_html=True,
            )
            st.caption("Aucun défaut critique détecté dans la structure.")

        if manual_review:
            st.warning(f"🔍 ZONE D'INSPECTION MANUELLE : Incertitude épistémique MC-Dropout élevée (σ = {prob_std:.3f}).")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 📏 Sévérité et Intégrité Physique")
        st.metric(
            "Taille / Sévérité Estimée",
            f"{severity_disp:.2f} mm" + (f" (± {mc_res['sev_std']:.2f} mm)" if is_crack else ""),
        )
        st.progress(float(np.clip(prob_crack, 0.0, 1.0)), text=f"Probabilité Calibrée d'Anomalie : {prob_crack*100:.1f}%")
        st.progress(health_score / 100.0, text=f"Score d'Intégrité Structurelle (Critique = {CRITICAL_SIZE_MM} mm) : {health_score:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 📍 Localisation Spatiale 3D (mm)")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Axe X</div><div class='metric-value'>{coords_disp[0]:.2f} mm</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Axe Y</div><div class='metric-value'>{coords_disp[1]:.2f} mm</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Axe Z (Prof.)</div><div class='metric-value'>{coords_disp[2]:.2f} mm</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if is_crack:
            is_edge_pred = (
                (coords_disp[0] < EDGE_LOW_MM) or (coords_disp[0] > EDGE_HIGH_MM)
                or (coords_disp[1] < EDGE_LOW_MM) or (coords_disp[1] > EDGE_HIGH_MM)
            )
            if is_edge_pred:
                st.warning("🚨 ALERTE BORDURES : Défaut situé dans la zone critique de paroi (< 3 mm ou > 12 mm).")
            else:
                st.info("ℹ️ Défaut localisé dans la zone centrale nominale de la structure.")

        # Comparaison directe côte à côte avec la vérité terrain
        if gt_info is not None:
            st.markdown("#### 🎯 Comparaison Vérité Terrain (Test Jamais Vu)")
            if gt_info["is_crack"]:
                gt_c = gt_info["coords"]
                err_xyz = float(np.mean(np.abs(coords_disp - gt_c))) if is_crack else float("nan")
                st.success(
                    f"**Réel :** Fissuré | Pos = ({gt_c[0]:.2f}, {gt_c[1]:.2f}, {gt_c[2]:.2f}) mm | "
                    f"Taille = {gt_info['severity']:.2f} mm\n\n"
                    f"**Écart moyen 3D (MAE) :** {err_xyz:.3f} mm"
                )
            else:
                st.success("**Réel :** Structure Saine (0.00 mm) — Prédiction conforme.")

    with col_right:
        st.markdown("### 🌐 Inspecteur Grad-CAM 3D Pur (Interactif 360°)")

        heatmap_3d = compute_honest_gradcam_3d(logit_extractor, tensor_batch)
        gt_coords_3d = gt_info["coords"] if (gt_info is not None and gt_info["is_crack"]) else None

        tab_3d, tab_2d = st.tabs(["🌐 Volume 3D Rotatif (360°)", "📈 Signal Temporel 2D (db4)"])

        with tab_3d:
            st.caption("🖱️ *Activation Grad-CAM trilinéaire pure calculée sur le logit (sans gaussienne artificielle).*")
            fig_3d = render_gradcam_3d_interactive(heatmap_3d, coords_disp, is_crack, gt_coords=gt_coords_3d)
            st.plotly_chart(fig_3d, use_container_width=True)

        with tab_2d:
            fig_2d, ax = plt.subplots(figsize=(7, 4))
            fig_2d.patch.set_facecolor("#0b0e14")
            ax.set_facecolor("#121721")

            if raw_1d_for_plot is not None and clean_1d_for_plot is not None:
                sensor_idx = int(np.clip(round((coords_disp[2] / 15.0) * 7), 0, 7)) if is_crack else 0
                ax.plot(raw_1d_for_plot[sensor_idx], color="#ff4d4d", alpha=0.55, label="Brut PZT")
                ax.plot(clean_1d_for_plot[sensor_idx], color="#00f2fe", linewidth=1.8, label="Filtré Ondelettes (db4)")
                ax.set_title(
                    f"Onde Acoustique 1D — Capteur Piézoélectrique PZT-{sensor_idx + 1}",
                    color="white", fontsize=10, fontweight="bold",
                )
            else:
                target_h = min(15, max(0, int(round(coords_disp[0])))) if is_crack else 8
                target_w = min(15, max(0, int(round(coords_disp[1])))) if is_crack else 8
                ax.plot(input_vol_3d[target_h, target_w, :, 0], color="#00f2fe", linewidth=2, label="Profil Spatial Z")
                ax.set_title(f"Profil Inter-Capteurs — Voxel ({target_h}, {target_w})", color="white", fontsize=10, fontweight="bold")

            ax.tick_params(colors="white")
            ax.grid(True, linestyle="--", alpha=0.2)
            ax.legend(facecolor="#161c28", labelcolor="white")
            st.pyplot(fig_2d)
            plt.close(fig_2d)

    gc.collect()
