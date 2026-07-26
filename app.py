# =====================================================================
#   🛡️ ANALYSE-CORE V2 - NASA PREDICTIVE MAINTENANCE DASHBOARD (app.py)
# =====================================================================

import os
import gdown
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# ---------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE & THÈME CYBERPUNK
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Analyse-Core v2 - NASA Diagnostic",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
        /* Fond global et typographie */
        .stApp {
            background: linear-gradient(135deg, #0a0b10 0%, #12131c 100%);
            color: #e2e8f0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Titre principal Style Néon */
        .neon-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
            margin-bottom: 0.5rem;
        }
        
        /* Cartes de diagnostic (Glassmorphism) */
        .diagnostic-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Badges d'état */
        .badge-normal {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.4) 100%);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
        }
        .badge-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.4) 100%);
            border: 1px solid #f59e0b;
            color: #fbbf24;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
        }
        .badge-danger {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
        }
        
        /* Barre latérale */
        section[data-testid="stSidebar"] {
            background-color: #0d0e15 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# 2. ARCHITECTURE DU MODÈLE RESNET-18 MULTI-TÂCHES
# ---------------------------------------------------------------------
class AnalyseCoreIMS(nn.Module):
    def __init__(self):
        super(AnalyseCoreIMS, self).__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        in_features = 512
        
        # Tête de Régression RUL
        self.rul_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Tête de Régression Santé globale
        self.health_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Tête de Classification des Défauts (4 classes)
        self.fault_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )
        
    def forward(self, img):
        features = self.backbone(img)
        features = torch.flatten(features, 1)
        return self.rul_head(features), self.health_head(features), self.fault_head(features)


# ---------------------------------------------------------------------
# 3. CHARGEMENT DU MODÈLE DEPUIS GOOGLE DRIVE
# ---------------------------------------------------------------------
GOOGLE_DRIVE_FILE_ID = "1BiF5lYgM7ChVaABQT3e3_stEjkugd95M"
MODEL_LOCAL_PATH = "analyse_core_ims_v2.pt"

@st.cache_resource
def load_predictive_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("📥 Connexion au stockage Cloud Google Drive..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            try:
                gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Erreur de téléchargement du modèle : {e}")
            
    model = AnalyseCoreIMS()
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


# ---------------------------------------------------------------------
# 4. GÉNÉRATEUR PHYSIQUE D'IMPACTS AMORTIS (CONFORME NASA IMS)
# ---------------------------------------------------------------------
def generate_realistic_signal(test_type, fs=20000, N=2048):
    t = np.linspace(0, N / fs, N, endpoint=False)
    noise = np.random.normal(0, 0.15, N)
    f_res = 3500.0  # Fréquence de résonance structurelle (3.5 kHz)
    decay = 900.0   # Taux d'amortissement de l'onde de choc

    if test_type == "Nominale (Machine Saine)":
        return noise

    # Fréquences physiques exactes du roulement Rexnord ZA-2115 (Dataset NASA IMS)
    if test_type == "Anomalie Bague Interne (Début d'usure)":
        f_fault = 236.0  # BPFI (Haute cadence d'impacts)
        amplitude = 2.2
    elif test_type == "Défaut de Bille (Dégradation moyenne)":
        f_fault = 278.0  # Cadence d'impacts billes
        amplitude = 1.8
    else:  # Bague Externe
        f_fault = 162.0  # BPFO (Cadence d'impacts plus basse)
        amplitude = 3.5

    sig = noise.copy()
    period_samples = int(fs / f_fault)
    
    # Injection des impulsions périodiques
    for idx in range(0, N, period_samples):
        dt = t[idx:] - t[idx]
        impact = amplitude * np.exp(-decay * dt) * np.sin(2 * np.pi * f_res * dt)
        sig[idx:] += impact

    return sig


# ---------------------------------------------------------------------
# 5. INTERFACE UTILISATEUR & DASHBOARD STREAMLIT
# ---------------------------------------------------------------------

# En-tête
st.markdown("<h1 class='neon-title'>🛡️ ANALYSE-CORE v2</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8f98; font-size: 1.1rem; margin-top:-10px;'>Système de Diagnostic Multi-Tâches IA & Maintenance Prédictive (Dataset NASA IMS)</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; height: 1px; background: linear-gradient(90deg, rgba(0,242,254,0.8), rgba(0,0,0,0)); margin-bottom: 30px;'>", unsafe_allow_html=True)

# Gestion de l'état de session pour la persistance des données
if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None

# Panneau latéral
st.sidebar.markdown("<h2 style='color: #00f2fe; font-size: 1.5rem; margin-bottom: 20px;'>🛠️ Salle de Contrôle</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #8a8f98; font-size: 0.9rem;'>Fournissez les vibrations de l'accéléromètre à analyser.</p>", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "Méthode de capture du signal",
    ["🔄 Générer un signal physique de test", "📤 Téléverser un fichier brut (.npy, .txt)"]
)

if option == "📤 Téléverser un fichier brut (.npy, .txt)":
    uploaded_file = st.sidebar.file_uploader("Fichier accélérométrique (min. 2048 points)", type=["npy", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".npy"):
            st.session_state.current_signal = np.load(uploaded_file)
        else:
            st.session_state.current_signal = np.loadtxt(uploaded_file)
else:
    test_type = st.sidebar.selectbox("Sélectionner une signature physique :", [
        "Nominale (Machine Saine)", 
        "Anomalie Bague Interne (Début d'usure)", 
        "Défaut de Bille (Dégradation moyenne)", 
        "Défaillance Bague Externe (État critique)"
    ])
    
    if st.sidebar.button("⚡ Injecter le signal dans l'analyseur"):
        st.session_state.current_signal = generate_realistic_signal(test_type)
        st.sidebar.success(f"Signal '{test_type}' injecté avec succès !")

sig = st.session_state.current_signal

# ---------------------------------------------------------------------
# 6. TRAITEMENT DU SIGNAL & PRÉDICTION IA
# ---------------------------------------------------------------------
if sig is not None:
    if len(sig) < 2048:
        st.error("⚠️ Erreur : Le signal doit contenir au moins 2048 points d'échantillonnage.")
    else:
        # Normalisation Z-score du segment
        segment = sig[:2048]
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-10)
        
        # 1. Spectrogramme STFT
        f, t_spec, Sxx = signal.spectrogram(segment, fs=20000, nperseg=256, noverlap=128)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-10)
        
        # 2. Preparation Tenseur PyTorch [1, 3, 128, 128]
        img_tensor = torch.tensor(Sxx_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = T.Resize((128, 128))(img_tensor).repeat(1, 3, 1, 1)
        
        # 3. Inférence IA
        model = load_predictive_model()
        with torch.no_grad():
            pred_rul, pred_health, pred_fault = model(img_tensor)
            
        rul_val = pred_rul.item() * 100.0
        health_val = pred_health.item() * 100.0
        
        # Probabilités Softmax sur les 4 classes
        probabilities = torch.softmax(pred_fault, dim=1).squeeze().numpy()
        fault_idx = np.argmax(probabilities)
        
        # 4. AFFICHAGE DES RÉSULTATS
        col_metrics, col_plots = st.columns([1, 1.3])
        
        with col_metrics:
            st.markdown("<h3 style='color: #00f2fe; margin-bottom: 20px;'>📊 Diagnostic de l'IA</h3>", unsafe_allow_html=True)
            
            classes = [
                ("🟢 MACHINE SAINE", "Aucune anomalie détectée. Fonctionnement nominal optimal.", "badge-normal"),
                ("⚠️ DEFAUT BAGUE INTERNE", "Usure naissante sur la bague interne du roulement.", "badge-warning"),
                ("⚠️ DEFAUT ELEMENTS ROULANTS", "Dégradation détectée sur les billes.", "badge-warning"),
                ("🚨 DEFAUT BAGUE EXTERNE (CRITIQUE)", "Défaillance structurelle majeure de la bague externe !", "badge-danger")
            ]
            
            title, desc, badge_style = classes[fault_idx]
            
            # Carte de classification principale
            st.markdown("<div class='diagnostic-card'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #8a8f98; margin-bottom: 5px; font-weight:600;'>CLASSIFICATION DU COMPORTEMENT</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='{badge_style}' style='font-size: 1.2rem;'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #d1d5db; margin-top: 10px; font-size: 0.9rem;'>{desc}</p>", unsafe_allow_html=True)
            
            # Probabilités détaillées
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #8a8f98; font-size:0.85rem; font-weight:bold;'>PROBABILITÉS PAR CLASSE :</p>", unsafe_allow_html=True)
            labels = ["Sain", "Bague Int.", "Bille", "Bague Ext."]
            for lbl, prob in zip(labels, probabilities):
                st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#a0aec0;'><span>{lbl}</span><span>{prob*100:.1f}%</span></div>", unsafe_allow_html=True)
                st.progress(float(prob))
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Jauges Santé & RUL
            st.markdown("<div class='diagnostic-card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='display:flex; justify-content:space-between;'><span style='font-weight:600;'>❤️ ÉTAT DE SANTÉ</span><span style='color:#00f2fe; font-weight:bold;'>{health_val:.1f}%</span></div>", unsafe_allow_html=True)
            st.progress(health_val / 100.0)
            
            st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='display:flex; justify-content:space-between;'><span style='font-weight:600;'>⏳ DURÉE DE VIE RESTANTE (RUL)</span><span style='color:#f59e0b; font-weight:bold;'>{rul_val:.1f}%</span></div>", unsafe_allow_html=True)
            st.progress(rul_val / 100.0)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_plots:
            st.markdown("<h3 style='color: #00f2fe; margin-bottom: 20px;'>📈 Signaux & Spectres de Fréquence</h3>", unsafe_allow_html=True)
            
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))
            fig.patch.set_facecolor('#0a0b10')
            
            # Signal temporel (Waveform)
            ax1.plot(segment[:600], color='#00f2fe', alpha=0.85, linewidth=1.2)
            ax1.set_facecolor('#11131e')
            ax1.set_title("VIBRATIONS BRUTES (ÉCHANTILLON TEMPOREL)", color='#8a8f98', fontsize=9, loc='left')
            ax1.grid(True, color='white', alpha=0.05)
            ax1.tick_params(colors='#8a8f98', labelsize=8)
            
            # Spectrogramme STFT
            im = ax2.pcolormesh(t_spec * 1000, f, Sxx_db, shading='gouraud', cmap='magma')
            ax2.set_facecolor('#11131e')
            ax2.set_title("SPECTROGRAMME STFT (RÉPARTITION SPECTRO-TEMPORELLE)", color='#8a8f98', fontsize=9, loc='left')
            ax2.set_ylabel("Fréquence (Hz)", color='#8a8f98', fontsize=8)
            ax2.set_xlabel("Temps (ms)", color='#8a8f98', fontsize=8)
            ax2.tick_params(colors='#8a8f98', labelsize=8)
            
            cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2, shrink=0.7)
            cbar.set_label("Densité de puissance (dB)", color='#8a8f98', fontsize=8)
            cbar.ax.tick_params(labelsize=7, colors='#8a8f98')
            
            plt.tight_layout()
            st.pyplot(fig)

else:
    # État initial au démarrage de l'app
    st.markdown("<div class='diagnostic-card' style='text-align: center; padding: 60px 20px; border-style: dashed; border-color: rgba(0, 242, 254, 0.4);'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#00f2fe; margin-bottom: 15px;'>📡 Système Prêt à l'Analyse</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8a8f98; font-size:1.1rem;'>Sélectionnez une signature dans le panneau latéral gauche et cliquez sur 'Injecter le signal'.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
