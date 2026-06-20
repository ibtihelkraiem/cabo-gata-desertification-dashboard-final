import time
import base64
import io
import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Earth Engine / web map packages for the Sentinel-2 Products Explorer
import ee
import folium
from streamlit_folium import st_folium



# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Cabo de Gata Desertification Dashboard",
    page_icon="🌵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent

# Expected structure:
#   app.py
#   data/
#   images/
#
# Fallback for GitHub web upload:
# if the CSV files or PNG images are uploaded directly in the repository root,
# the dashboard will still find them.
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR

IMAGES_DIR = BASE_DIR / "images"
if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("*.png")):
    IMAGES_DIR = BASE_DIR

YEARS = [
    "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"
]

YEAR_FILE = {
    "2015-2016": "2015_2016",
    "2016-2017": "2016_2017",
    "2017-2018": "2017_2018",
    "2018-2019": "2018_2019",
    "2019-2020": "2019_2020",
    "2020-2021": "2020_2021",
    "2021-2022": "2021_2022",
    "2022-2023": "2022_2023",
    "2023-2024": "2023_2024",
    "2024-2025": "2024_2025",
}

STATIC_LAYERS = ["Prototype 2024-2025"]

LEGEND_INFO = {
    "NDVI": [
        ("#D73027", "Highly Non-Vegetated"),
        ("#FC8D59", "Bare Soil"),
        ("#FEE08B", "Arid Land"),
        ("#A6D96A", "Moderate Vegetation"),
        ("#1A9850", "Dense Vegetation"),
        ("#F6F4F0", "Study area boundary"),
    ],
    "NDWI": [
        ("#B30000", "Extreme Aridity"),
        ("#FFB000", "High Water Stress"),
        ("#E8E85D", "Moderate Water Stress"),
        ("#6FA6E8", "Hydrated Vegetation"),
        ("#0A2E8A", "High Moisture"),
        ("#F6F4F0", "Study area boundary"),
    ],
    "NDDI": [
        ("#1557B0", "No Drought"),
        ("#A7F000", "Mild Stress"),
        ("#F0F06A", "Moderate Drought"),
        ("#F7A900", "Severe Drought"),
        ("#E60000", "Extreme Drought"),
        ("#F6F4F0", "Study area boundary"),
    ],
    "Risk Classification": [
        ("#4C7F00", "Very Low"),
        ("#8FD400", "Low"),
        ("#F4EA00", "Moderate"),
        ("#F7AE00", "High"),
        ("#F40000", "Very High"),
        ("#F6F4F0", "Study area boundary"),
    ],
    "Prototype 2024-2025": [
        ("#4C7F00", "Very Low"),
        ("#8FD400", "Low"),
        ("#F4EA00", "Moderate"),
        ("#F7AE00", "High"),
        ("#F40000", "Very High"),
        ("#F6F4F0", "Study area boundary"),
    ],
}




# ============================================================
# MULTILINGUAL TEXT
# ============================================================

LANG_OPTIONS = {
    "EN": "en",
    "FR": "fr",
    "ES": "es",
}

UI_TEXT = {
    "en": {
        "app_title": "AI-Based Desertification Monitoring Dashboard",
        "subtitle": "Cabo de Gata–Níjar Natural Park · Sentinel-2 indices · Random Forest desertification risk classification · Prototype 2024–2025",
        "controls": "Dashboard controls",
        "language": "Language",
        "layer_type": "Layer type",
        "hydro_year": "Hydrological year",
        "animation_speed": "Animation speed",
        "speed_slow": "Slow – 1 s/year",
        "speed_normal": "Normal – 0.5 s/year",
        "speed_fast": "Fast – 0.15 s/year",
        "auto": "▶ Auto visualization",
        "stop": "■ Stop",
        "selected_year": "Selected year",
        "fixed_proto": "Prototype 2024–2025 is a fixed zoomed layout for the current situation.",
        "auto_running": "Auto visualization running",
        "legend_title": "Legend – what each color means",
        "legend_note": "Legend based on the symbology used for the selected dashboard layer.",
        "map_ref": "Map reference information",
        "source": "Source",
        "resolution": "Spatial resolution",
        "coordinate_system": "Coordinate system",
        "epsg": "EPSG",
        "formula_method": "Formula / method",
        "layer_explanation": "Layer explanation",
        "mean_ndvi": "Mean NDVI",
        "mean_ndwi": "Mean NDWI",
        "drought_level": "Drought intensity level",
        "very_high": "Very High risk",
        "high_very_high": "High + Very High risk",
        "hydro_axis": "Hydrological year",
        "value": "Value",
        "risk_chart_title": "Evolution of High + Very High Desertification Risk",
        "area_affected": "Area affected (%)",
        "detailed": "Concise analysis",
        "status": "Status",
        "ten_year_rank": "10-year rank",
        "risk_context": "Risk context",
        "prediction_title": "Scenario-Based Desertification Risk Estimation for Next Years",
        "prediction_intro": "This exploratory tool estimates possible future risk from vegetation condition, surface moisture, drought intensity, precipitation and temperature anomaly.",
        "expected_ndvi": "Expected mean NDVI",
        "expected_ndwi": "Expected mean NDWI",
        "expected_drought": "Expected drought intensity score",
        "precipitation": "Precipitation compared with normal year (%)",
        "temperature": "Temperature anomaly (°C)",
        "est_vh": "Estimated Very High risk",
        "est_hvh": "Estimated High + Very High risk",
        "stress_score": "Scenario stress score",
        "est_vh_sub": "Expected share of the study area classified as Very High risk.",
        "est_hvh_sub": "Expected severe-risk area under the selected scenario.",
        "stress_sub": "Synthetic stress indicator based on vegetation, moisture and climate pressure.",
        "scenario_assessment": "Scenario assessment",
        "risk_explanation": "Risk explanation",
        "prediction_explain": "Higher stress values mean weaker vegetation, lower moisture, stronger drought, lower precipitation and warmer conditions. Together, these factors increase the probability that High and Very High risk zones expand.",
        "critical": "Critical concern",
        "high_concern": "High concern",
        "moderate_concern": "Moderate concern",
        "low_concern": "Low to moderate concern",
        "critical_text": "Very High risk is strongly concentrated. Priority monitoring is required in the most exposed areas.",
        "high_text": "Severe risk may expand where vegetation cover and moisture are low.",
        "moderate_text": "Severe-risk pressure is moderate; local hotspots remain sensitive to drought.",
        "low_text": "Severe-risk expansion is limited, but vulnerable hotspots should still be monitored.",
        "synthesis_title": "Dashboard synthesis",
        "synthesis_text": "The dashboard combines Sentinel-2 indicators, AI desertification risk classification and a 2024–2025 prototype zoom. The map gives the spatial pattern, the chart shows the 10-year evolution, and the analysis explains the selected year in a concise way.",
        "risk_distribution_title": "Risk Class Distribution",
        "risk_distribution_note": "The bar chart summarizes the proportion of the study area in each desertification risk class for the selected hydrological year.",
    },
    "fr": {
        "app_title": "Tableau de bord de suivi de la désertification par IA",
        "subtitle": "Parc naturel de Cabo de Gata–Níjar · Indices Sentinel-2 · Classification Random Forest · Prototype 2024–2025",
        "controls": "Contrôles du tableau de bord",
        "language": "Langue",
        "layer_type": "Type de couche",
        "hydro_year": "Année hydrologique",
        "animation_speed": "Vitesse d’animation",
        "speed_slow": "Lente – 1 s/an",
        "speed_normal": "Normale – 0,5 s/an",
        "speed_fast": "Rapide – 0,15 s/an",
        "auto": "▶ Visualisation automatique",
        "stop": "■ Arrêter",
        "selected_year": "Année sélectionnée",
        "fixed_proto": "Le prototype 2024–2025 est une vue zoomée fixe de la situation actuelle.",
        "auto_running": "Visualisation automatique en cours",
        "legend_title": "Légende – signification des couleurs",
        "legend_note": "Légende basée sur la symbologie utilisée pour la couche sélectionnée.",
        "map_ref": "Informations cartographiques",
        "source": "Source",
        "resolution": "Résolution spatiale",
        "coordinate_system": "Système de coordonnées",
        "epsg": "EPSG",
        "formula_method": "Formule / méthode",
        "layer_explanation": "Explication de la couche",
        "mean_ndvi": "NDVI moyen",
        "mean_ndwi": "NDWI moyen",
        "drought_level": "Niveau d’intensité de sécheresse",
        "very_high": "Risque très élevé",
        "high_very_high": "Risque élevé + très élevé",
        "hydro_axis": "Année hydrologique",
        "value": "Valeur",
        "risk_chart_title": "Évolution du risque élevé + très élevé",
        "area_affected": "Surface affectée (%)",
        "detailed": "Analyse concise",
        "status": "État",
        "ten_year_rank": "Rang sur 10 ans",
        "risk_context": "Contexte du risque",
        "prediction_title": "Estimation du risque de désertification pour les prochaines années",
        "prediction_intro": "Ce module exploratoire estime le risque futur possible à partir de l’état de la végétation, de l’humidité de surface, de l’intensité de sécheresse, des précipitations et de l’anomalie thermique.",
        "expected_ndvi": "NDVI moyen attendu",
        "expected_ndwi": "NDWI moyen attendu",
        "expected_drought": "Score d’intensité de sécheresse attendu",
        "precipitation": "Précipitations par rapport à une année normale (%)",
        "temperature": "Anomalie de température (°C)",
        "est_vh": "Risque très élevé estimé",
        "est_hvh": "Risque élevé + très élevé estimé",
        "stress_score": "Score de stress du scénario",
        "est_vh_sub": "Part attendue de la zone classée en risque très élevé.",
        "est_hvh_sub": "Surface à risque sévère attendue selon le scénario.",
        "stress_sub": "Indicateur synthétique basé sur végétation, humidité et pression climatique.",
        "scenario_assessment": "Évaluation du scénario",
        "risk_explanation": "Explication du risque",
        "prediction_explain": "Des valeurs de stress élevées indiquent une végétation plus faible, moins d’humidité, une sécheresse plus forte, moins de précipitations et des conditions plus chaudes. Ces facteurs augmentent la probabilité d’expansion des zones à risque élevé et très élevé.",
        "critical": "Préoccupation critique",
        "high_concern": "Préoccupation élevée",
        "moderate_concern": "Préoccupation modérée",
        "low_concern": "Préoccupation faible à modérée",
        "critical_text": "Le risque très élevé est fortement concentré. Un suivi prioritaire est nécessaire dans les zones les plus exposées.",
        "high_text": "Le risque sévère peut s’étendre lorsque la végétation et l’humidité sont faibles.",
        "moderate_text": "La pression du risque sévère reste modérée ; les hotspots locaux demeurent sensibles à la sécheresse.",
        "low_text": "L’expansion du risque sévère est limitée, mais les zones vulnérables doivent rester surveillées.",
        "synthesis_title": "Synthèse du tableau de bord",
        "synthesis_text": "Le tableau de bord combine les indicateurs Sentinel-2, la classification du risque par IA et un zoom prototype 2024–2025. La carte montre la distribution spatiale, le graphique montre l’évolution sur 10 ans et l’analyse explique l’année sélectionnée de manière concise.",
        "risk_distribution_title": "Distribution des classes de risque",
        "risk_distribution_note": "Le graphique résume la proportion de la zone d’étude dans chaque classe de risque pour l’année hydrologique sélectionnée.",
    },
    "es": {
        "app_title": "Panel de seguimiento de desertificación basado en IA",
        "subtitle": "Parque Natural Cabo de Gata–Níjar · Índices Sentinel-2 · Clasificación Random Forest · Prototipo 2024–2025",
        "controls": "Controles del panel",
        "language": "Idioma",
        "layer_type": "Tipo de capa",
        "hydro_year": "Año hidrológico",
        "animation_speed": "Velocidad de animación",
        "speed_slow": "Lenta – 1 s/año",
        "speed_normal": "Normal – 0,5 s/año",
        "speed_fast": "Rápida – 0,15 s/año",
        "auto": "▶ Visualización automática",
        "stop": "■ Detener",
        "selected_year": "Año seleccionado",
        "fixed_proto": "El prototipo 2024–2025 es una vista ampliada fija de la situación actual.",
        "auto_running": "Visualización automática en curso",
        "legend_title": "Leyenda – significado de los colores",
        "legend_note": "Leyenda basada en la simbología utilizada para la capa seleccionada.",
        "map_ref": "Información cartográfica",
        "source": "Fuente",
        "resolution": "Resolución espacial",
        "coordinate_system": "Sistema de coordenadas",
        "epsg": "EPSG",
        "formula_method": "Fórmula / método",
        "layer_explanation": "Explicación de la capa",
        "mean_ndvi": "NDVI medio",
        "mean_ndwi": "NDWI medio",
        "drought_level": "Nivel de intensidad de sequía",
        "very_high": "Riesgo muy alto",
        "high_very_high": "Riesgo alto + muy alto",
        "hydro_axis": "Año hidrológico",
        "value": "Valor",
        "risk_chart_title": "Evolución del riesgo alto + muy alto",
        "area_affected": "Superficie afectada (%)",
        "detailed": "Análisis conciso",
        "status": "Estado",
        "ten_year_rank": "Rango en 10 años",
        "risk_context": "Contexto del riesgo",
        "prediction_title": "Estimación del riesgo de desertificación para próximos años",
        "prediction_intro": "Este módulo exploratorio estima el riesgo futuro a partir del estado de la vegetación, la humedad superficial, la intensidad de sequía, la precipitación y la anomalía térmica.",
        "expected_ndvi": "NDVI medio esperado",
        "expected_ndwi": "NDWI medio esperado",
        "expected_drought": "Puntuación esperada de sequía",
        "precipitation": "Precipitación frente a un año normal (%)",
        "temperature": "Anomalía de temperatura (°C)",
        "est_vh": "Riesgo muy alto estimado",
        "est_hvh": "Riesgo alto + muy alto estimado",
        "stress_score": "Puntuación de estrés del escenario",
        "est_vh_sub": "Porcentaje esperado del área clasificado como riesgo muy alto.",
        "est_hvh_sub": "Área de riesgo severo esperada según el escenario.",
        "stress_sub": "Indicador sintético basado en vegetación, humedad y presión climática.",
        "scenario_assessment": "Evaluación del escenario",
        "risk_explanation": "Explicación del riesgo",
        "prediction_explain": "Valores altos de estrés indican vegetación más débil, menor humedad, sequía más intensa, menor precipitación y condiciones más cálidas. En conjunto, estos factores aumentan la probabilidad de expansión de zonas de riesgo alto y muy alto.",
        "critical": "Preocupación crítica",
        "high_concern": "Preocupación alta",
        "moderate_concern": "Preocupación moderada",
        "low_concern": "Preocupación baja a moderada",
        "critical_text": "El riesgo muy alto está fuertemente concentrado. Se requiere seguimiento prioritario en las zonas más expuestas.",
        "high_text": "El riesgo severo puede expandirse donde la cobertura vegetal y la humedad son bajas.",
        "moderate_text": "La presión de riesgo severo es moderada; los hotspots locales siguen siendo sensibles a la sequía.",
        "low_text": "La expansión del riesgo severo es limitada, aunque las zonas vulnerables deben mantenerse bajo seguimiento.",
        "synthesis_title": "Síntesis del panel",
        "synthesis_text": "El panel combina indicadores Sentinel-2, clasificación de riesgo por IA y un prototipo ampliado 2024–2025. El mapa muestra el patrón espacial, el gráfico muestra la evolución de 10 años y el análisis explica el año seleccionado de forma concisa.",
        "risk_distribution_title": "Distribución de clases de riesgo",
        "risk_distribution_note": "El gráfico resume la proporción del área de estudio en cada clase de riesgo para el año hidrológico seleccionado.",
    },
}

LAYER_TEXT = {
    "NDVI": {
        "en": {"title": "NDVI – Vegetation Condition", "description": "Vegetation vigor and density. Higher values generally indicate healthier vegetation.", "science": "NDVI is used to monitor vegetation productivity and drought response.", "chart_title": "Temporal Evolution of Mean NDVI", "y_title": "Mean NDVI"},
        "fr": {"title": "NDVI – État de la végétation", "description": "Vigueur et densité de la végétation. Des valeurs élevées indiquent généralement une végétation plus saine.", "science": "Le NDVI permet de suivre la productivité végétale et la réponse à la sécheresse.", "chart_title": "Évolution temporelle du NDVI moyen", "y_title": "NDVI moyen"},
        "es": {"title": "NDVI – Estado de la vegetación", "description": "Vigor y densidad de la vegetación. Valores altos indican generalmente vegetación más sana.", "science": "El NDVI permite analizar productividad vegetal y respuesta a la sequía.", "chart_title": "Evolución temporal del NDVI medio", "y_title": "NDVI medio"},
    },
    "NDWI": {
        "en": {"title": "NDWI – Surface Moisture Condition", "description": "Surface moisture condition. Lower values indicate drier vegetation or soil.", "science": "NDWI helps identify moisture dynamics in semi-arid environments.", "chart_title": "Temporal Evolution of Mean NDWI", "y_title": "Mean NDWI"},
        "fr": {"title": "NDWI – Humidité de surface", "description": "État de l’humidité de surface. Des valeurs faibles indiquent une végétation ou un sol plus secs.", "science": "Le NDWI aide à identifier la dynamique d’humidité en milieu semi-aride.", "chart_title": "Évolution temporelle du NDWI moyen", "y_title": "NDWI moyen"},
        "es": {"title": "NDWI – Humedad superficial", "description": "Condición de humedad superficial. Valores bajos indican vegetación o suelo más secos.", "science": "El NDWI ayuda a identificar dinámicas de humedad en ambientes semiáridos.", "chart_title": "Evolución temporal del NDWI medio", "y_title": "NDWI medio"},
    },
    "NDDI": {
        "en": {"title": "NDDI – Surface Drought Intensity", "description": "Surface dryness derived from vegetation and moisture information.", "science": "NDDI identifies relative drought intensity and spatial stress patterns.", "chart_title": "Temporal Evolution of NDDI Drought Score", "y_title": "Drought score (0–100)"},
        "fr": {"title": "NDDI – Intensité de sécheresse", "description": "Sécheresse de surface dérivée de la végétation et de l’humidité.", "science": "Le NDDI identifie l’intensité relative de sécheresse et les zones de stress spatial.", "chart_title": "Évolution du score de sécheresse NDDI", "y_title": "Score de sécheresse (0–100)"},
        "es": {"title": "NDDI – Intensidad de sequía", "description": "Sequedad superficial derivada de vegetación y humedad.", "science": "El NDDI identifica intensidad relativa de sequía y patrones espaciales de estrés.", "chart_title": "Evolución del score de sequía NDDI", "y_title": "Score de sequía (0–100)"},
    },
    "Risk Classification": {
        "en": {"title": "AI Desertification Risk Classification", "description": "Annual desertification risk derived from the Random Forest model.", "science": "The model combines spectral indicators, land conditions and climatic stress.", "chart_title": "Evolution of High + Very High Desertification Risk", "y_title": "Area affected (%)"},
        "fr": {"title": "Classification du risque de désertification par IA", "description": "Risque annuel de désertification dérivé du modèle Random Forest.", "science": "Le modèle combine les indicateurs spectraux, les conditions du territoire et le stress climatique.", "chart_title": "Évolution du risque élevé + très élevé", "y_title": "Surface affectée (%)"},
        "es": {"title": "Clasificación del riesgo de desertificación por IA", "description": "Riesgo anual de desertificación derivado del modelo Random Forest.", "science": "El modelo combina indicadores espectrales, condiciones del territorio y estrés climático.", "chart_title": "Evolución del riesgo alto + muy alto", "y_title": "Superficie afectada (%)"},
    },
    "Prototype 2024-2025": {
        "en": {"title": "Prototype 2024–2025 – Current Situation", "description": "Zoomed view of the selected hotspot affected by severe desertification risk.", "science": "The prototype highlights local spatial concentration of severe risk in the current situation.", "chart_title": "Evolution of High + Very High Desertification Risk", "y_title": "Area affected (%)"},
        "fr": {"title": "Prototype 2024–2025 – Situation actuelle", "description": "Vue zoomée du hotspot sélectionné affecté par un risque sévère de désertification.", "science": "Le prototype met en évidence la concentration locale du risque sévère dans la situation actuelle.", "chart_title": "Évolution du risque élevé + très élevé", "y_title": "Surface affectée (%)"},
        "es": {"title": "Prototipo 2024–2025 – Situación actual", "description": "Vista ampliada del hotspot seleccionado afectado por riesgo severo de desertificación.", "science": "El prototipo muestra la concentración local del riesgo severo en la situación actual.", "chart_title": "Evolución del riesgo alto + muy alto", "y_title": "Superficie afectada (%)"},
    },
}


# Short names used only in the Layer type selector.
# The internal layer names remain unchanged, so the dashboard logic is not affected.
LAYER_SELECTOR_LABELS = {
    "NDVI": "NDVI",
    "NDWI": "NDWI",
    "NDDI": "NDDI",
    "Risk Classification": "Risk map",
    "Prototype 2024-2025": "Prototype",
}


LEGEND_LABELS = {
    "NDVI": {
        "en": ["Highly Non-Vegetated", "Bare Soil", "Arid Land", "Moderate Vegetation", "Dense Vegetation", "Study area boundary"],
        "fr": ["Très faiblement végétalisé", "Sol nu", "Terrain aride", "Végétation modérée", "Végétation dense", "Limite de la zone d’étude"],
        "es": ["Muy baja vegetación", "Suelo desnudo", "Terreno árido", "Vegetación moderada", "Vegetación densa", "Límite del área de estudio"],
    },
    "NDWI": {
        "en": ["Extreme Aridity", "High Water Stress", "Moderate Water Stress", "Hydrated Vegetation", "High Moisture"],
        "fr": ["Aridité extrême", "Stress hydrique élevé", "Stress hydrique modéré", "Végétation hydratée", "Humidité élevée"],
        "es": ["Aridez extrema", "Estrés hídrico alto", "Estrés hídrico moderado", "Vegetación hidratada", "Alta humedad"],
    },
    "NDDI": {
        "en": ["No Drought", "Mild Stress", "Moderate Drought", "Severe Drought", "Extreme Drought"],
        "fr": ["Pas de sécheresse", "Stress léger", "Sécheresse modérée", "Sécheresse sévère", "Sécheresse extrême"],
        "es": ["Sin sequía", "Estrés leve", "Sequía moderada", "Sequía severa", "Sequía extrema"],
    },
    "Risk Classification": {
        "en": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "fr": ["Très faible", "Faible", "Modéré", "Élevé", "Très élevé"],
        "es": ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"],
    },
    "Prototype 2024-2025": {
        "en": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "fr": ["Très faible", "Faible", "Modéré", "Élevé", "Très élevé"],
        "es": ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"],
    },
}

def get_lang():
    return st.session_state.get("language_code", "en")

def tr(key):
    lang = get_lang()
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, UI_TEXT["en"].get(key, key))

def lt(layer_type, field):
    lang = get_lang()
    return LAYER_TEXT.get(layer_type, {}).get(lang, LAYER_TEXT.get(layer_type, {}).get("en", {})).get(field, LAYER_INFO[layer_type].get(field, field))


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}

    /* Keep Streamlit header available so the sidebar open/close arrow stays visible */
    header[data-testid="stHeader"] {
        background: rgba(7, 16, 23, 0.35) !important;
        height: 2.2rem !important;
        visibility: visible !important;
    }

    div[data-testid="collapsedControl"],
    button[kind="header"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    .stApp {
        color: #F6F4F0;
        background:
            radial-gradient(circle at 12% 18%, rgba(46, 92, 62, 0.26), transparent 26%),
            radial-gradient(circle at 92% 14%, rgba(188, 155, 106, 0.21), transparent 30%),
            radial-gradient(circle at 72% 82%, rgba(106, 67, 33, 0.34), transparent 34%),
            linear-gradient(135deg, #071017 0%, #0D1820 42%, #20160E 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.10;
        z-index: 0;
        background:
            repeating-radial-gradient(circle at 22% 28%, transparent 0px, transparent 16px, rgba(246,244,240,0.22) 17px, transparent 19px),
            repeating-radial-gradient(circle at 82% 72%, transparent 0px, transparent 22px, rgba(188,155,106,0.16) 24px, transparent 26px);
    }

    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 0.35rem;
        padding-left: 1.35rem;
        padding-right: 1.35rem;
        max-width: 1680px;
    }

    section[data-testid="stSidebar"] {
        min-width: 285px !important;
        width: 285px !important;
        background: linear-gradient(180deg, rgba(4,10,15,0.98) 0%, rgba(11,21,28,0.98) 100%);
        border-right: 1px solid rgba(242,201,139,0.18);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0.35rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #F6F4F0 !important;
    }

    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {
        color: #111111 !important;
    }

    h1 {
        color: #F6F4F0;
        font-size: 2.05rem !important;
        line-height: 1.12;
        font-weight: 900;
        margin-bottom: 0.25rem;
        margin-top: 0.1rem;
        text-shadow: 0 3px 12px rgba(0,0,0,0.65);
    }

    h2 {
        color: #F2C98B;
        font-size: 1.38rem !important;
        font-weight: 850;
        margin-top: 0.15rem;
        margin-bottom: 0.25rem;
    }

    h3 {
        color: #F2C98B;
        font-weight: 850;
    }

    .hero-card {
        background: rgba(8, 16, 24, 0.58);
        border: 1px solid rgba(242, 201, 139, 0.18);
        border-radius: 18px;
        padding: 10px 16px 9px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.24);
        margin-bottom: 0.45rem;
    }

    .subtitle {
        color: #E6D2A9;
        font-size: 0.96rem;
        line-height: 1.45;
        margin-top: 0.25rem;
        margin-bottom: 0.15rem;
    }

    .small-text {
        color: #E6D2A9;
        font-size: 0.93rem;
        line-height: 1.45;
    }

    .scientific-note {
        background: rgba(242, 201, 139, 0.07);
        border-left: 4px solid #BC9B6A;
        color: #F6F4F0;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 0.90rem;
        line-height: 1.45;
        margin-top: 0.55rem;
    }

    .main-map-card {
        background: transparent;
        border: none;
        border-radius: 18px;
        padding: 0;
        box-shadow: none;
        max-width: 100%;
        margin-left: auto;
        margin-right: auto;
    }

    .language-card {
        max-width: 360px;
        margin: 0.2rem 0 0.6rem auto;
    }

    div[data-testid="stHorizontalBlock"] {
        row-gap: 0.45rem !important;
    }

    .section-card, .deep-analysis-card, .legend-card, .map-meta-card {
        margin-bottom: 0.55rem !important;
    }

    .layout-map-card {
        max-width: 100% !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .layout-map-image {
        width: 100% !important;
        max-height: 520px !important;
        object-fit: contain !important;
        background: transparent !important;
        display: block !important;
        margin: 0 auto !important;
    }

    .dashboard-map {
        width: 100%;
        max-height: 520px;
        object-fit: contain;
        display: block;
        border-radius: 12px;
        background: transparent;
    }


    .prototype-map {
        max-height: 470px;
        width: 100%;
    }

    .legend-card {
        background: rgba(8, 16, 24, 0.52);
        border: 1px solid rgba(242, 201, 139, 0.18);
        border-radius: 16px;
        padding: 10px 13px;
        margin-top: 0.55rem;
        margin-left: 0;
        margin-right: 0;
        box-shadow: 0 10px 28px rgba(0,0,0,0.26);
        min-height: 185px;
    }

    .legend-title {
        color: #F2C98B;
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 0.55rem;
    }

    .legend-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem 0.75rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #F6F4F0;
        font-size: 0.88rem;
        line-height: 1.3;
    }

    .legend-swatch {
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1px solid rgba(255,255,255,0.42);
        flex: 0 0 18px;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
    }

    .legend-note {
        color: #E6D2A9;
        font-size: 0.78rem;
        margin-top: 0.6rem;
        line-height: 1.35;
    }

    .map-meta-card {
        background: rgba(8, 16, 24, 0.52);
        border: 1px solid rgba(242, 201, 139, 0.18);
        border-radius: 16px;
        padding: 10px 13px;
        margin-top: 0.55rem;
        margin-left: 0;
        margin-right: 0;
        box-shadow: 0 10px 28px rgba(0,0,0,0.26);
        min-height: 185px;
    }
    .map-meta-title {
        color: #F2C98B;
        font-size: 0.98rem;
        font-weight: 850;
        margin-bottom: 0.55rem;
    }
    .map-meta-row {
        color: #F6F4F0;
        font-size: 0.84rem;
        line-height: 1.45;
        margin-bottom: 0.28rem;
    }
    .map-meta-label {
        color: #E6D2A9;
        font-weight: 700;
    }
    .map-meta-note {
        color: #E6D2A9;
        font-size: 0.76rem;
        line-height: 1.35;
        margin-top: 0.55rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        height: 52px !important;
        min-height: 52px !important;
        padding: 0.35rem 0.45rem !important;
        font-size: 0.82rem !important;
        white-space: normal !important;
        line-height: 1.1 !important;
    }


    .control-panel {
        background: rgba(8,16,24,0.78);
        border: 1px solid rgba(242,201,139,0.22);
        border-radius: 18px;
        padding: 12px 16px;
        margin-bottom: 0.7rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    }

    .control-title {
        color: #F2C98B;
        font-size: 1.05rem;
        font-weight: 850;
        margin-bottom: 0.28rem;
    }


    /* Strong text visibility */
    label, .stRadio label, .stSlider label, .stSelectbox label,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span {
        color: #F6F4F0 !important;
        opacity: 1 !important;
    }

    .stRadio label p,
    div[role="radiogroup"] label p,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] span {
        color: #F6F4F0 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    .hero-card {
        text-align: center !important;
    }

    .hero-card h1 {
        text-align: center !important;
    }

    .hero-card .subtitle {
        text-align: center !important;
    }

    .control-panel label,
    .control-panel p,
    .control-panel span {
        color: #F6F4F0 !important;
        opacity: 1 !important;
    }

    .control-panel .stRadio div[role="radiogroup"] label {
        opacity: 1 !important;
    }



    .full-width-analysis {
        width: 100%;
        margin-top: 0.55rem;
        margin-bottom: 0.7rem;
    }

    .full-width-analysis .deep-analysis-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .full-width-analysis .deep-analysis-text {
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .deep-analysis-card {
        background: rgba(8, 16, 24, 0.82);
        border: 1px solid rgba(188, 155, 106, 0.34);
        border-radius: 16px;
        padding: 11px 14px;
        margin-top: 0.55rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.24);
    }

    .deep-analysis-title {
        color: #F2C98B;
        font-size: 1.05rem;
        font-weight: 900;
        margin-bottom: 0.55rem;
    }

    .deep-analysis-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 0.6rem;
    }

    .deep-analysis-item {
        background: rgba(246, 244, 240, 0.055);
        border: 1px solid rgba(246, 244, 240, 0.10);
        border-radius: 12px;
        padding: 9px 10px;
    }

    .deep-analysis-label {
        color: #D9C7A3;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.22rem;
    }

    .deep-analysis-value {
        color: #F6F4F0;
        font-size: 0.96rem;
        font-weight: 850;
        line-height: 1.28;
    }

    .deep-analysis-text {
        color: #F6F4F0;
        font-size: 0.90rem;
        line-height: 1.50;
    }

    .deep-analysis-text b {
        color: #F2C98B;
    }

    .side-panel {
        background: rgba(8, 16, 24, 0.78);
        border: 1px solid rgba(242, 201, 139, 0.22);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.30);
        margin-bottom: 0.55rem;
    }

    .selected-year {
        font-size: 1.25rem;
        font-weight: 900;
        color: #F6F4F0;
        margin-bottom: 0.45rem;
    }

    .selected-year-value {
        color: #69F083;
        font-family: Consolas, monospace;
        background: rgba(105, 240, 131, 0.10);
        border: 1px solid rgba(105, 240, 131, 0.25);
        border-radius: 9px;
        padding: 2px 8px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(246,244,240,0.055);
        border: 1px solid rgba(246,244,240,0.10);
        border-radius: 12px;
        padding: 9px 12px;
        margin-top: 8px;
    }

    .metric-label {
        color: #D9C7A3;
        font-size: 0.91rem;
    }

    .metric-value {
        color: #F6F4F0;
        font-size: 1.22rem;
        font-weight: 850;
    }

    .prediction-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 0.55rem;
        margin-bottom: 0.3rem;
    }

    .prediction-card {
        background: rgba(8,16,24,0.82);
        border: 1px solid rgba(242, 201, 139, 0.20);
        border-radius: 15px;
        padding: 14px 14px 12px 14px;
    }

    .prediction-title {
        color: #D9C7A3;
        font-size: 0.92rem;
        margin-bottom: 0.3rem;
    }

    .prediction-value {
        color: #F6F4F0;
        font-size: 2.05rem;
        line-height: 1.05;
        font-weight: 900;
    }

    .prediction-subtext {
        color: #E6D2A9;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .scenario-box {
        background: rgba(251, 140, 0, 0.12);
        border: 1px solid #FB8C00;
        color: #F6F4F0;
        border-radius: 15px;
        padding: 14px 16px;
        margin-top: 0.65rem;
    }

    .section-card {
        background: rgba(8, 16, 24, 0.52);
        border: 1px solid rgba(242, 201, 139, 0.18);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.26);
        margin-top: 0.45rem;
    }

    div[data-testid="stPlotlyChart"] {
        background: transparent;
        border-radius: 16px;
        padding: 0;
        border: none;
        box-shadow: none;
    }

    .stButton > button {
        background: linear-gradient(135deg, #F2C98B, #BC9B6A) !important;
        color: #081018 !important;
        border: 1px solid rgba(246,244,240,0.25) !important;
        border-radius: 12px !important;
        font-weight: 850 !important;
        padding: 0.58rem 0.85rem !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #FFE0A6, #D4B382) !important;
        color: #000000 !important;
    }

    .chart-caption {
        color: #D9C7A3;
        font-size: 0.88rem;
        margin-bottom: 0.15rem;
    }

    /* Smartphone layout fixes */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
            padding-top: 0.20rem !important;
        }

        .hero-card {
            padding: 12px 10px !important;
            margin-bottom: 0.55rem !important;
            border-radius: 14px !important;
        }

        .hero-card h1,
        h1 {
            font-size: 1.45rem !important;
            line-height: 1.18 !important;
        }

        .subtitle {
            font-size: 0.78rem !important;
            line-height: 1.35 !important;
        }

        .section-card {
            padding: 12px 10px !important;
            border-radius: 14px !important;
            margin-top: 0.35rem !important;
        }

        .section-card h2,
        h2 {
            font-size: 1.10rem !important;
            line-height: 1.25 !important;
        }

        .small-text {
            font-size: 0.82rem !important;
            line-height: 1.45 !important;
        }

        .prediction-grid {
            grid-template-columns: 1fr !important;
            gap: 8px !important;
        }

        .prediction-card {
            padding: 11px 12px !important;
            border-radius: 13px !important;
        }

        .prediction-title {
            font-size: 0.82rem !important;
            line-height: 1.25 !important;
        }

        .prediction-value {
            font-size: 1.65rem !important;
            line-height: 1.12 !important;
            overflow-wrap: normal !important;
            word-break: normal !important;
            white-space: normal !important;
        }

        .prediction-subtext {
            font-size: 0.75rem !important;
            line-height: 1.35 !important;
        }

        .control-panel {
            padding: 10px 11px !important;
            border-radius: 14px !important;
        }

        .control-title {
            font-size: 1.0rem !important;
        }

        .side-panel {
            padding: 11px 12px !important;
            border-radius: 14px !important;
        }

        .selected-year {
            font-size: 1.05rem !important;
        }

        .metric-row {
            padding: 8px 10px !important;
        }

        .metric-label {
            font-size: 0.78rem !important;
        }

        .metric-value {
            font-size: 1.0rem !important;
        }

        .dashboard-map,
        .layout-map-image {
            max-height: 430px !important;
            width: 100% !important;
            object-fit: contain !important;
        }

        div[data-testid="stPlotlyChart"] {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        .legend-grid {
            grid-template-columns: 1fr !important;
        }

        .legend-card,
        .map-meta-card,
        .deep-analysis-card {
            padding: 10px 11px !important;
            border-radius: 14px !important;
        }

        .deep-analysis-grid {
            grid-template-columns: 1fr !important;
        }

        /* Keep language selector usable on mobile */
        div[data-testid="stSelectbox"] {
            max-width: 155px !important;
        }
    }

    @media (max-width: 420px) {
        .hero-card h1,
        h1 {
            font-size: 1.25rem !important;
        }

        .prediction-value {
            font-size: 1.45rem !important;
        }

        .dashboard-map,
        .layout-map-image {
            max-height: 360px !important;
        }
    }

    /* Sentinel-2 summary cards: desktop = 3 cards, tablet/phone = vertical */
    .s2-summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 0.55rem;
        margin-bottom: 0.3rem;
    }

    .s2-summary-card {
        background: rgba(8,16,24,0.82);
        border: 1px solid rgba(242, 201, 139, 0.20);
        border-radius: 15px;
        padding: 14px 14px 12px 14px;
        overflow-wrap: normal;
        word-break: normal;
    }

    .s2-summary-title {
        color: #D9C7A3;
        font-size: 0.92rem;
        margin-bottom: 0.3rem;
        line-height: 1.25;
    }

    .s2-summary-value {
        color: #F6F4F0;
        font-size: 2.05rem;
        line-height: 1.05;
        font-weight: 900;
        white-space: normal;
        overflow-wrap: normal;
        word-break: normal;
    }

    .s2-summary-subtext {
        color: #E6D2A9;
        font-size: 0.82rem;
        margin-top: 0.25rem;
        line-height: 1.35;
    }

    @media (max-width: 1100px) {
        .s2-summary-grid {
            grid-template-columns: 1fr !important;
        }

        .s2-summary-card {
            padding: 11px 12px !important;
        }

        .s2-summary-title {
            font-size: 0.88rem !important;
        }

        .s2-summary-value {
            font-size: 1.75rem !important;
            line-height: 1.12 !important;
        }

        .s2-summary-subtext {
            font-size: 0.78rem !important;
        }
    }

    @media (max-width: 520px) {
        .s2-summary-value {
            font-size: 1.55rem !important;
        }

        .section-card {
            padding-left: 10px !important;
            padding-right: 10px !important;
        }
    }

    /* Final compact card numbers fix: works on phone and desktop */
    .prediction-value,
    .s2-summary-value {
        font-size: 1.32rem !important;
        line-height: 1.18 !important;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        white-space: normal !important;
    }

    .prediction-title,
    .s2-summary-title {
        font-size: 0.80rem !important;
        line-height: 1.25 !important;
    }

    .prediction-subtext,
    .s2-summary-subtext {
        font-size: 0.72rem !important;
        line-height: 1.30 !important;
    }

    .prediction-card,
    .s2-summary-card {
        padding: 9px 10px !important;
        border-radius: 12px !important;
        min-width: 0 !important;
    }

    /* Make all 3-card blocks adapt better to narrow phone screens */
    .prediction-grid,
    .s2-summary-grid {
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)) !important;
        gap: 8px !important;
    }

    @media (max-width: 700px) {
        .prediction-value,
        .s2-summary-value {
            font-size: 1.15rem !important;
            line-height: 1.18 !important;
        }

        .prediction-title,
        .s2-summary-title {
            font-size: 0.74rem !important;
        }

        .prediction-subtext,
        .s2-summary-subtext {
            font-size: 0.68rem !important;
        }

        .prediction-grid,
        .s2-summary-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            gap: 6px !important;
        }

        .prediction-card,
        .s2-summary-card {
            padding: 8px 8px !important;
        }
    }

    @media (max-width: 430px) {
        .prediction-value,
        .s2-summary-value {
            font-size: 1.00rem !important;
            line-height: 1.16 !important;
        }

        .prediction-title,
        .s2-summary-title {
            font-size: 0.68rem !important;
        }

        .prediction-subtext,
        .s2-summary-subtext {
            font-size: 0.62rem !important;
        }

        .prediction-card,
        .s2-summary-card {
            padding: 7px 6px !important;
        }
    }

    /* Mobile-friendly layer type selector */
    @media (max-width: 760px) {
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            min-width: 190px !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            min-height: 42px !important;
            font-size: 0.90rem !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
            font-size: 0.90rem !important;
            line-height: 1.10 !important;
        }
    }

    hr {
        border-color: rgba(242, 201, 139, 0.16);
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_tables():
    required = [
        DATA_DIR / "dashboard_indices_scores_v2.csv",
        DATA_DIR / "high_very_high_evolution_v2.csv",
        DATA_DIR / "risk_area_statistics_v2.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("Missing CSV file(s):\n" + "\n".join(missing))
        st.stop()

    indices = pd.read_csv(required[0])
    risk = pd.read_csv(required[1])
    risk_area = pd.read_csv(required[2])

    for df in [indices, risk, risk_area]:
        df["year"] = pd.Categorical(df["year"], categories=YEARS, ordered=True)
        df.sort_values("year", inplace=True)

    if "combined_surface_stress_0_100" not in indices.columns:
        # fallback approximation from the cleaned indicators
        if "ndvi_score_0_100" in indices.columns:
            veg_stress = 100 - indices["ndvi_score_0_100"].astype(float)
        else:
            ndvi = indices["ndvi_mean"].astype(float)
            veg_stress = 100 - ((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min()) * 100)

        if "ndwi_score_0_100" in indices.columns:
            moist_stress = 100 - indices["ndwi_score_0_100"].astype(float)
        else:
            ndwi = indices["ndwi_mean"].astype(float)
            moist_stress = 100 - ((ndwi - ndwi.min()) / (ndwi.max() - ndwi.min()) * 100)

        if "nddi_drought_score_0_100" in indices.columns:
            drought_score = indices["nddi_drought_score_0_100"].astype(float)
        else:
            nddi = indices["nddi_mean"].astype(float)
            drought_score = (nddi - nddi.min()) / (nddi.max() - nddi.min()) * 100

        indices["combined_surface_stress_0_100"] = 0.35 * veg_stress + 0.25 * moist_stress + 0.40 * drought_score

    if "nddi_drought_score_0_100" not in indices.columns:
        nddi = indices["nddi_mean"].astype(float)
        indices["nddi_drought_score_0_100"] = (nddi - nddi.min()) / (nddi.max() - nddi.min()) * 100

    return indices, risk, risk_area


indices_df, risk_df, risk_area_df = load_tables()


# ============================================================
# LAYER INFORMATION
# ============================================================

LAYER_INFO = {
    "NDVI": {
        "title": "NDVI – Vegetation Condition",
        "description": "NDVI describes the vigor and density of vegetation cover. Higher values generally indicate healthier vegetation and lower surface degradation pressure.",
        "science": "NDVI is derived from red and near-infrared Sentinel-2 bands and is commonly used to monitor vegetation productivity and drought response.",
        "chart_title": "Temporal Evolution of Mean NDVI",
        "column": "ndvi_mean",
        "color": "#56C667",
        "unit": "",
        "y_title": "Mean NDVI",
    },
    "NDWI": {
        "title": "NDWI – Surface Moisture Condition",
        "description": "NDWI reflects relative surface moisture conditions. Lower values tend to indicate drier conditions and reduced water content in vegetation or soil.",
        "science": "NDWI is based on near-infrared and short-wave infrared reflectance and is useful for identifying moisture dynamics in semi-arid environments.",
        "chart_title": "Temporal Evolution of Mean NDWI",
        "column": "ndwi_mean",
        "color": "#4DA3FF",
        "unit": "",
        "y_title": "Mean NDWI",
    },
    "NDDI": {
        "title": "NDDI – Surface Drought Intensity",
        "description": "NDDI summarizes surface dryness by combining vegetation and moisture information. The dashboard chart shows the normalized drought score from 0 to 100 for easier interpretation.",
        "science": "NDDI integrates NDVI and NDWI behaviour to identify relative surface drought intensity and spatial stress patterns.",
        "chart_title": "Temporal Evolution of NDDI Drought Score",
        "column": "nddi_drought_score_0_100",
        "color": "#FF8C00",
        "unit": "/100",
        "y_title": "Drought score (0–100)",
    },
    "Risk Classification": {
        "title": "AI Desertification Risk Classification",
        "description": "This layer shows the annual desertification risk classification derived from the Random Forest model using remote sensing and environmental predictors.",
        "science": "The risk model integrates spectral indicators, land conditions and climatic stress to identify the spatial distribution of desertification vulnerability.",
        "chart_title": "Evolution of High + Very High Desertification Risk",
        "column": "high_very_high_percent",
        "color": "#D32F2F",
        "unit": "%",
        "y_title": "Area affected (%)",
    },
    "Prototype 2024-2025": {
        "title": "Prototype 2024–2025 – Current Situation",
        "description": "This zoomed layout focuses on the selected prototype area affected by high and very high desertification risk during the current hydrological year.",
        "science": "The prototype map provides a local-scale view of the current desertification situation, showing how severe risk is spatially concentrated inside the selected hotspot area.",
        "chart_title": "Evolution of High + Very High Desertification Risk",
        "column": "high_very_high_percent",
        "color": "#D32F2F",
        "unit": "%",
        "y_title": "Area affected (%)",
    },
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def initialize_state():
    defaults = {
        "animating": False,
        "animation_index": 0,
        "current_layer": "NDVI",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def image_path(layer_type, year):
    yf = YEAR_FILE[year]
    if layer_type in ["NDVI", "NDWI", "NDDI"]:
        return IMAGES_DIR / f"{layer_type}_{yf}.png"
    if layer_type == "Risk Classification":
        return IMAGES_DIR / f"Risk_{yf}.png"
    if layer_type == "Prototype 2024-2025":
        return IMAGES_DIR / "Prototype_2024_2025.png"
    return None


@st.cache_data(show_spinner=False)
def image_to_base64_full_layout(path_str):
    """
    Read the ArcGIS layout export, crop useless margins, remove the light
    page/map background, and recolor edge coordinates to white so the layout
    fits the dark dashboard better while preserving grid, north arrow and
    scale bar.
    """
    path = Path(path_str)
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3].astype(np.int16)

    # Estimate page background from image corners
    sample = max(8, min(arr.shape[0], arr.shape[1]) // 40)
    corners = np.concatenate([
        rgb[:sample, :sample].reshape(-1, 3),
        rgb[:sample, -sample:].reshape(-1, 3),
        rgb[-sample:, :sample].reshape(-1, 3),
        rgb[-sample:, -sample:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0)

    # Crop useless external margins
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    content_mask = dist > 18
    if content_mask.any():
        ys, xs = np.where(content_mask)
        pad = max(6, min(arr.shape[0], arr.shape[1]) // 120)
        y0 = max(0, ys.min() - pad)
        y1 = min(arr.shape[0], ys.max() + pad + 1)
        x0 = max(0, xs.min() - pad)
        x1 = min(arr.shape[1], xs.max() + pad + 1)
        arr = arr[y0:y1, x0:x1].copy()
        rgb = arr[:, :, :3].astype(np.int16)

    h, w = arr.shape[:2]

    # 1) Remove external page background connected to edges
    edge_bg = np.sqrt(((rgb - bg) ** 2).sum(axis=2)) <= 22
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    for x in range(w):
        if edge_bg[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if edge_bg[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if edge_bg[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if edge_bg[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and edge_bg[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    arr[visited, 3] = 0

    # 2) Remove the light background inside the map frame (keep grid/objects)
    rgb = arr[:, :, :3].astype(np.int16)
    bright_gray = (
        (arr[:, :, 3] > 0)
        & (rgb.mean(axis=2) > 210)
        & (np.abs(rgb[:, :, 0] - rgb[:, :, 1]) < 10)
        & (np.abs(rgb[:, :, 1] - rgb[:, :, 2]) < 10)
    )
    arr[bright_gray, 3] = 0

    # 3) Recolor only edge coordinate annotations to white for better readability,
    # while keeping the study-area boundary black.
    rgb = arr[:, :, :3]
    edge_band = (
        (np.arange(h)[:, None] < int(h * 0.16))
        | (np.arange(h)[:, None] > int(h * 0.84))
        | (np.arange(w)[None, :] < int(w * 0.16))
        | (np.arange(w)[None, :] > int(w * 0.84))
    )
    dark_candidates = (
        (arr[:, :, 3] > 0)
        & edge_band
        & (rgb[:, :, 0] < 140)
        & (rgb[:, :, 1] < 140)
        & (rgb[:, :, 2] < 140)
    )

    transparent = (arr[:, :, 3] == 0).astype(np.int32)
    colorful = (
        (arr[:, :, 3] > 0)
        & (
            (np.abs(rgb[:, :, 0] - rgb[:, :, 1]) > 20)
            | (np.abs(rgb[:, :, 1] - rgb[:, :, 2]) > 20)
            | (np.abs(rgb[:, :, 0] - rgb[:, :, 2]) > 20)
        )
    ).astype(np.int32)

    def neighbour_count(mask, radius=2):
        """Return a same-size neighbourhood count without shape mismatch."""
        mask = mask.astype(np.int16)
        padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="constant")
        total = np.zeros(mask.shape, dtype=np.int16)
        window = 2 * radius + 1

        for dy in range(window):
            for dx in range(window):
                total += padded[dy:dy + h, dx:dx + w]

        return total

    trans_count = neighbour_count(transparent, radius=2)
    color_count = neighbour_count(colorful, radius=2)

    # Convert neutral dark cartographic elements to white for the dark dashboard.
    # This intentionally includes coordinates, grid labels, north arrow, scale bar
    # and the study-area boundary. It avoids recoloring coloured raster pixels.
    rgb_i = arr[:, :, :3].astype(np.int16)
    neutral_dark = (
        (arr[:, :, 3] > 0)
        & (rgb_i[:, :, 0] < 155)
        & (rgb_i[:, :, 1] < 155)
        & (rgb_i[:, :, 2] < 155)
        & (np.abs(rgb_i[:, :, 0] - rgb_i[:, :, 1]) <= 38)
        & (np.abs(rgb_i[:, :, 1] - rgb_i[:, :, 2]) <= 38)
        & (np.abs(rgb_i[:, :, 0] - rgb_i[:, :, 2]) <= 38)
    )

    arr[neutral_dark, 0] = 245
    arr[neutral_dark, 1] = 245
    arr[neutral_dark, 2] = 245

    out_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    out_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()



def display_map(layer_type, year):
    path = image_path(layer_type, year)
    if path is not None and path.exists():
        img_base64 = image_to_base64_full_layout(str(path))
        st.markdown(
            f"""
            <div class="main-map-card layout-map-card">
                <img class="dashboard-map layout-map-image" src="data:image/png;base64,{img_base64}">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Map image not found: {path.name if path else 'unknown'}")


def legend_html(layer_type):
    items = LEGEND_INFO.get(layer_type, [])
    if not items:
        return ""

    labels = LEGEND_LABELS.get(layer_type, {}).get(get_lang())
    if labels and len(labels) == len(items):
        items = [(items[i][0], labels[i]) for i in range(len(items))]

    item_html = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{color};"></span><span>{label}</span></div>'
        for color, label in items
    )

    return f"""
    <div class="legend-card">
        <div class="legend-title">{tr('legend_title')}</div>
        <div class="legend-grid">{item_html}</div>
        <div class="legend-note">{tr('legend_note')}</div>
    </div>
    """


def map_metadata_html(layer_type):
    if layer_type == "NDVI":
        formula = "NDVI = (NIR − RED) / (NIR + RED)"
        source = "ESA Copernicus Sentinel-2 MSI (Level-2A, S2A/S2B)"
    elif layer_type == "NDWI":
        formula = "NDWI = (NIR − SWIR) / (NIR + SWIR)"
        source = "ESA Copernicus Sentinel-2 MSI (Level-2A, S2A/S2B)"
    elif layer_type == "NDDI":
        formula = "NDDI = (NDVI − NDWI) / (NDVI + NDWI)"
        source = "ESA Copernicus Sentinel-2 MSI (Level-2A, S2A/S2B)"
    elif layer_type == "Risk Classification":
        formula = "Random Forest classification based on NDVI, NDWI, NDDI and environmental predictors"
        source = "Author's Random Forest classification using Sentinel-2 indices and environmental variables"
    else:
        formula = "Zoomed Random Forest desertification risk classification for the selected prototype area"
        source = "Derived from the Random Forest desertification risk classification, 2024–2025"

    return f"""
    <div class="map-meta-card">
        <div class="map-meta-title">{tr('map_ref')}</div>
        <div class="map-meta-row"><span class="map-meta-label">{tr('source')}:</span> {source}</div>
        <div class="map-meta-row"><span class="map-meta-label">{tr('resolution')}:</span> 10 m</div>
        <div class="map-meta-row"><span class="map-meta-label">{tr('coordinate_system')}:</span> ETRS 1989 ETRS UTM Zone 30 N</div>
        <div class="map-meta-row"><span class="map-meta-label">{tr('epsg')}:</span> 25830</div>
        <div class="map-meta-row"><span class="map-meta-label">{tr('formula_method')}:</span> {formula}</div>
    </div>
    """



def format_drought_score(value):
    """Format NDDI drought score without showing confusing 0.0/100."""
    try:
        if value is None or np.isnan(value):
            return "not available"
        value = float(value)
    except Exception:
        return "not available"

    if value < 1:
        return "lowest relative drought level"
    return f"{value:.1f}/100"



def selected_metrics_html(layer_type, year):
    if layer_type in ["Risk Classification", "Prototype 2024-2025"]:
        row = risk_df[risk_df["year"].astype(str) == year]
        if row.empty:
            return ""
        vh = float(row["very_high_percent"].iloc[0])
        hvh = float(row["high_very_high_percent"].iloc[0])
        return (
            f'<div class="metric-row"><span class="metric-label">{tr("very_high")}</span><span class="metric-value">{vh:.2f}%</span></div>'
            f'<div class="metric-row"><span class="metric-label">{tr("high_very_high")}</span><span class="metric-value">{hvh:.2f}%</span></div>'
        )

    row = indices_df[indices_df["year"].astype(str) == year]
    if row.empty:
        return ""

    if layer_type == "NDVI":
        value = float(row["ndvi_mean"].iloc[0])
        return f'<div class="metric-row"><span class="metric-label">{tr("mean_ndvi")}</span><span class="metric-value">{value:.3f}</span></div>'
    if layer_type == "NDWI":
        value = float(row["ndwi_mean"].iloc[0])
        return f'<div class="metric-row"><span class="metric-label">{tr("mean_ndwi")}</span><span class="metric-value">{value:.3f}</span></div>'
    if layer_type == "NDDI":
        value = float(row["nddi_drought_score_0_100"].iloc[0])
        return f'<div class="metric-row"><span class="metric-label">{tr("drought_level")}</span><span class="metric-value">{format_drought_score(value)}</span></div>'
    return ""


def get_year_df(layer_type):
    if layer_type in ["NDVI", "NDWI", "NDDI"]:
        return indices_df.copy()
    return risk_df.copy()


def chart_range(values):
    arr = np.array(values, dtype=float)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    span = max_v - min_v
    if span == 0:
        span = 1.0
    return [min_v - span * 0.16, max_v + span * 0.22]


def evolution_chart(layer_type, selected_year):
    info = LAYER_INFO[layer_type]
    df = get_year_df(layer_type)
    df["year_str"] = df["year"].astype(str)
    y_col = info["column"]

    selected_row = df[df["year_str"] == selected_year]
    selected_value = float(selected_row[y_col].iloc[0]) if not selected_row.empty else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["year_str"],
            y=df[y_col],
            mode="lines+markers",
            name=lt(layer_type, "chart_title"),
            line=dict(color=info["color"], width=4, shape="spline", smoothing=0.45),
            marker=dict(size=8, color=info["color"], line=dict(color="#F6F4F0", width=1)),
            hovertemplate=f"<b>%{{x}}</b><br>{tr('value')}: %{{y:.3f}}<extra></extra>",
        )
    )

    if selected_value is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected_year],
                y=[selected_value],
                mode="markers",
                marker=dict(size=24, color="#F6F4F0", line=dict(color=info["color"], width=6)),
                showlegend=False,
                hovertemplate=f"<b>{tr('selected_year')}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=selected_year,
            y=selected_value,
            text=f"<b>{selected_year}</b><br>{format_drought_score(selected_value) if layer_type == 'NDDI' else str(round(selected_value, 2)) + info['unit']}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-48,
            bgcolor="rgba(8,16,24,0.96)",
            bordercolor=info["color"],
            borderwidth=2,
            font=dict(color="#F6F4F0", size=12),
        )

    fig.update_layout(
        title=dict(text=lt(layer_type, "chart_title"), font=dict(size=20, color="#F2C98B"), x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,24,0.78)",
        font=dict(color="#F6F4F0", size=12),
        height=270,
        margin=dict(l=50, r=20, t=62, b=52),
        xaxis=dict(title=tr("hydro_axis"), gridcolor="rgba(242,201,139,0.10)", tickangle=-30, linecolor="rgba(246,244,240,0.30)"),
        yaxis=dict(title=lt(layer_type, "y_title"), gridcolor="rgba(242,201,139,0.10)", range=chart_range(df[y_col])),
        showlegend=False,
    )
    return fig


def risk_double_chart(selected_year):
    df = risk_df.copy()
    df["year_str"] = df["year"].astype(str)
    row = df[df["year_str"] == selected_year]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["year_str"],
            y=df["high_very_high_percent"],
            mode="lines+markers",
            name=tr("high_very_high"),
            line=dict(color="#FB8C00", width=5, shape="spline", smoothing=0.45),
            marker=dict(size=10, color="#FB8C00", line=dict(color="#F6F4F0", width=1.5)),
            hovertemplate=f"<b>%{{x}}</b><br>{tr('high_very_high')}: %{{y:.2f}}%<extra></extra>",
        )
    )

    if not row.empty:
        hvh = float(row["high_very_high_percent"].iloc[0])
        fig.add_trace(go.Scatter(x=[selected_year], y=[hvh], mode="markers", marker=dict(size=28, color="#F6F4F0", line=dict(color="#FB8C00", width=7)), showlegend=False))
        fig.add_annotation(x=selected_year, y=hvh, text=f"<b>{selected_year}</b><br>{hvh:.1f}%", showarrow=True, arrowhead=2, ax=0, ay=-48, bgcolor="rgba(8,16,24,0.96)", bordercolor="#FB8C00", borderwidth=2, font=dict(color="#F6F4F0", size=12))

    fig.update_layout(
        title=dict(text=tr("risk_chart_title"), font=dict(size=20, color="#F2C98B"), x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,24,0.78)",
        font=dict(color="#F6F4F0", size=12),
        height=270,
        margin=dict(l=50, r=20, t=62, b=52),
        xaxis=dict(title=dict(text=tr("hydro_axis"), font=dict(color="#F6F4F0")), tickfont=dict(color="#F6F4F0"), gridcolor="rgba(242,201,139,0.10)", tickangle=-30, linecolor="rgba(246,244,240,0.30)"),
        yaxis=dict(title=dict(text=tr("area_affected"), font=dict(color="#F6F4F0")), tickfont=dict(color="#F6F4F0"), gridcolor="rgba(242,201,139,0.10)", ticksuffix="%", range=chart_range(df["high_very_high_percent"])),
        showlegend=False,
    )
    return fig


def risk_distribution_chart(selected_year):
    df = risk_area_df[risk_area_df["year"].astype(str) == selected_year].copy()
    order = ["Very Low", "Low", "Moderate", "High", "Very High"]
    colors = {
        "Very Low": "#1B5E20",
        "Low": "#4CE600",
        "Moderate": "#FFFF00",
        "High": "#FB8C00",
        "Very High": "#D32F2F",
    }
    df["risk_class"] = pd.Categorical(df["risk_class"], categories=order, ordered=True)
    df = df.sort_values("risk_class")

    fig = go.Figure(
        go.Bar(
            y=df["risk_class"].astype(str),
            x=df["percentage"],
            orientation="h",
            marker=dict(color=[colors[c] for c in df["risk_class"].astype(str)]),
            text=[f"{v:.1f}%" for v in df["percentage"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=f"Risk Class Distribution – {selected_year}", font=dict(size=18, color="#F2C98B"), x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,24,0.78)",
        font=dict(color="#F6F4F0", size=12),
        height=260,
        margin=dict(l=95, r=36, t=60, b=40),
        xaxis=dict(title="Percentage of study area (%)", ticksuffix="%", gridcolor="rgba(242,201,139,0.10)"),
        yaxis=dict(title=""),
    )
    return fig


def scientific_description_html(layer_type, year=None):
    if layer_type == "Prototype 2024-2025":
        explanation = lt(layer_type, "science")
    elif layer_type == "Risk Classification" and year == "2024-2025":
        explanation = {
            "en": "In 2024–2025, High and Very High classes highlight the most exposed sectors, where low vegetation cover and surface dryness are dominant.",
            "fr": "En 2024–2025, les classes élevé et très élevé mettent en évidence les secteurs les plus exposés, dominés par une faible végétation et une forte sécheresse de surface.",
            "es": "En 2024–2025, las clases alto y muy alto destacan los sectores más expuestos, dominados por baja vegetación y sequedad superficial."
        }.get(get_lang())
    else:
        explanation = lt(layer_type, "science")
    return f'<div class="scientific-note"><b>{tr("layer_explanation")}:</b> {explanation}</div>'


def scenario_metric_cards(predicted_vh, predicted_hvh, combined_stress):
    return f"""
    <div class="prediction-grid">
        <div class="prediction-card">
            <div class="prediction-title">{tr('est_vh')}</div>
            <div class="prediction-value">{predicted_vh:.1f}%</div>
            <div class="prediction-subtext">{tr('est_vh_sub')}</div>
        </div>
        <div class="prediction-card">
            <div class="prediction-title">{tr('est_hvh')}</div>
            <div class="prediction-value">{predicted_hvh:.1f}%</div>
            <div class="prediction-subtext">{tr('est_hvh_sub')}</div>
        </div>
        <div class="prediction-card">
            <div class="prediction-title">{tr('stress_score')}</div>
            <div class="prediction-value">{combined_stress:.1f}/100</div>
            <div class="prediction-subtext">{tr('stress_sub')}</div>
        </div>
    </div>
    """


def scenario_prediction_section():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"## {tr('prediction_title')}")
    st.markdown(f'<div class="small-text">{tr("prediction_intro")}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        ndvi = st.slider(tr("expected_ndvi"), -0.05, 0.50, 0.18, 0.01)
        ndwi = st.slider(tr("expected_ndwi"), -0.60, 0.10, -0.30, 0.01)
        drought_score = st.slider(tr("expected_drought"), 0, 100, 54, 1)
    with c2:
        precipitation = st.slider(tr("precipitation"), 20, 140, 75, 5)
        temperature = st.slider(tr("temperature"), -2.0, 4.0, 1.0, 0.1)

    ndvi_min, ndvi_max = float(indices_df["ndvi_mean"].min()), float(indices_df["ndvi_mean"].max())
    ndwi_min, ndwi_max = float(indices_df["ndwi_mean"].min()), float(indices_df["ndwi_mean"].max())
    vegetation_score = np.clip((ndvi - ndvi_min) / max(ndvi_max - ndvi_min, 1e-9) * 100, 0, 100)
    moisture_score = np.clip((ndwi - ndwi_min) / max(ndwi_max - ndwi_min, 1e-9) * 100, 0, 100)
    vegetation_stress = 100 - vegetation_score
    moisture_stress = 100 - moisture_score
    precipitation_stress = np.clip(100 - precipitation, 0, 100)
    temperature_stress = np.clip((temperature + 2) / 6 * 100, 0, 100)
    combined_stress = 0.25 * vegetation_stress + 0.20 * moisture_stress + 0.25 * drought_score + 0.20 * precipitation_stress + 0.10 * temperature_stress

    hist = indices_df[["year", "combined_surface_stress_0_100"]].merge(risk_df[["year", "very_high_percent", "high_very_high_percent"]], on="year", how="inner")
    x = hist["combined_surface_stress_0_100"].astype(float).values
    predicted_vh = float(np.clip(np.polyval(np.polyfit(x, hist["very_high_percent"].astype(float).values, 1), combined_stress), 0, 100))
    predicted_hvh = float(np.clip(np.polyval(np.polyfit(x, hist["high_very_high_percent"].astype(float).values, 1), combined_stress), 0, 100))

    st.markdown(scenario_metric_cards(predicted_vh, predicted_hvh, combined_stress), unsafe_allow_html=True)

    if predicted_vh >= 30:
        level, color, interpretation = tr("critical"), "#D32F2F", tr("critical_text")
    elif predicted_vh >= 20:
        level, color, interpretation = tr("high_concern"), "#FB8C00", tr("high_text")
    elif predicted_vh >= 10:
        level, color, interpretation = tr("moderate_concern"), "#FFD54F", tr("moderate_text")
    else:
        level, color, interpretation = tr("low_concern"), "#56C667", tr("low_text")

    st.markdown(f'<div class="scenario-box"><b style="color:{color};">{tr("scenario_assessment")}: {level}</b><br>{interpretation}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="scientific-note"><b>{tr("risk_explanation")}:</b> {tr("prediction_explain")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def safe_float(row, column, default=np.nan):
    if column in row.index:
        try:
            return float(row[column])
        except Exception:
            return default
    return default


def previous_year_of(year):
    idx = YEARS.index(year)
    return YEARS[idx - 1] if idx > 0 else None


def direction_text(value, mean_value, higher_is_stress=False):
    if np.isnan(value) or np.isnan(mean_value):
        return "No comparison available"
    if abs(value - mean_value) < 1e-9:
        return "close to the 10-year mean"
    if higher_is_stress:
        return "above the 10-year mean, indicating stronger stress" if value > mean_value else "below the 10-year mean, indicating weaker stress"
    return "above the 10-year mean" if value > mean_value else "below the 10-year mean"


def change_text(value, previous_value, unit="", positive_name="increase", negative_name="decrease"):
    if previous_value is None or np.isnan(value) or np.isnan(previous_value):
        return "First year of the series"
    diff = value - previous_value
    if abs(diff) < 0.005:
        return f"stable compared with previous year"
    word = positive_name if diff > 0 else negative_name
    return f"{word} of {abs(diff):.2f}{unit} from previous year"


def rank_text(df, column, year, higher_is_stronger=True):
    if column not in df.columns:
        return "Ranking unavailable"
    temp = df.copy()
    temp["year_str"] = temp["year"].astype(str)
    temp[column] = temp[column].astype(float)
    ascending = not higher_is_stronger
    ordered = temp.sort_values(column, ascending=ascending).reset_index(drop=True)
    pos = ordered.index[ordered["year_str"] == year]
    if len(pos) == 0:
        return "Ranking unavailable"
    return f"{int(pos[0]) + 1} / {len(ordered)}"


def current_index_row(year):
    row = indices_df[indices_df["year"].astype(str) == year]
    return row.iloc[0] if not row.empty else None


def current_risk_row(year):
    row = risk_df[risk_df["year"].astype(str) == year]
    return row.iloc[0] if not row.empty else None


def detailed_analysis_html(layer_type, year):
    idx_row = current_index_row(year)
    risk_row = current_risk_row(year)
    prev_year = previous_year_of(year)
    prev_idx = current_index_row(prev_year) if prev_year else None
    prev_risk = current_risk_row(prev_year) if prev_year else None

    lang = get_lang()

    if layer_type == "NDVI":
        value = safe_float(idx_row, "ndvi_mean")
        prev = safe_float(prev_idx, "ndvi_mean") if prev_idx is not None else np.nan
        rank = rank_text(indices_df, "ndvi_mean", year, True)
        risk = safe_float(risk_row, "high_very_high_percent")
        if np.isnan(prev) or abs(value - prev) < 0.005:
            trend_en, trend_fr, trend_es = "stable", "stable", "estable"
        elif value > prev:
            trend_en, trend_fr, trend_es = "higher", "plus élevée", "más alta"
        else:
            trend_en, trend_fr, trend_es = "lower", "plus faible", "más baja"
        texts = {
            "en": f"Mean NDVI is {value:.3f}. Vegetation is {trend_en} compared with the previous year. Severe risk reaches {risk:.2f}%, showing that vegetation condition alone does not explain risk; moisture and land exposure remain important.",
            "fr": f"Le NDVI moyen est {value:.3f}. La végétation est {trend_fr} par rapport à l’année précédente. Le risque sévère atteint {risk:.2f} %, ce qui montre que la végétation seule n’explique pas le risque ; l’humidité et l’exposition du sol restent importantes.",
            "es": f"El NDVI medio es {value:.3f}. La vegetación es {trend_es} respecto al año anterior. El riesgo severo alcanza {risk:.2f} %, lo que muestra que la vegetación por sí sola no explica el riesgo; la humedad y la exposición del suelo siguen siendo importantes.",
        }
        card1, card2, card3 = f"{value:.3f}", rank, f"{risk:.2f}%"
        label1, label2, label3 = tr("mean_ndvi"), tr("ten_year_rank"), tr("risk_context")
    elif layer_type == "NDWI":
        value = safe_float(idx_row, "ndwi_mean")
        prev = safe_float(prev_idx, "ndwi_mean") if prev_idx is not None else np.nan
        rank = rank_text(indices_df, "ndwi_mean", year, True)
        risk = safe_float(risk_row, "high_very_high_percent")
        if np.isnan(prev) or abs(value - prev) < 0.005:
            trend_en, trend_fr, trend_es = "stable", "stable", "estable"
        elif value > prev:
            trend_en, trend_fr, trend_es = "wetter", "plus humide", "más húmeda"
        else:
            trend_en, trend_fr, trend_es = "drier", "plus sèche", "más seca"
        texts = {
            "en": f"Mean NDWI is {value:.3f}. Surface moisture is {trend_en} compared with the previous year. Severe risk is {risk:.2f}%, confirming the role of water stress in desertification vulnerability.",
            "fr": f"Le NDWI moyen est {value:.3f}. L’humidité de surface est {trend_fr} par rapport à l’année précédente. Le risque sévère est de {risk:.2f} %, confirmant le rôle du stress hydrique dans la vulnérabilité à la désertification.",
            "es": f"El NDWI medio es {value:.3f}. La humedad superficial es {trend_es} respecto al año anterior. El riesgo severo es {risk:.2f} %, confirmando el papel del estrés hídrico en la vulnerabilidad a la desertificación.",
        }
        card1, card2, card3 = f"{value:.3f}", rank, f"{risk:.2f}%"
        label1, label2, label3 = tr("mean_ndwi"), tr("ten_year_rank"), tr("risk_context")
    elif layer_type == "NDDI":
        value = safe_float(idx_row, "nddi_drought_score_0_100")
        rank = rank_text(indices_df, "nddi_drought_score_0_100", year, True)
        vh = safe_float(risk_row, "very_high_percent")
        texts = {
            "en": f"The NDDI drought level is {format_drought_score(value)}. This indicator combines vegetation weakness and moisture deficit. Very High risk covers {vh:.2f}%, highlighting where surface drought becomes critical.",
            "fr": f"Le niveau de sécheresse NDDI est {format_drought_score(value)}. Cet indicateur combine faiblesse de la végétation et déficit d’humidité. Le risque très élevé couvre {vh:.2f} %, indiquant les zones où la sécheresse de surface devient critique.",
            "es": f"El nivel de sequía NDDI es {format_drought_score(value)}. Este indicador combina debilidad de la vegetación y déficit de humedad. El riesgo muy alto cubre {vh:.2f} %, señalando dónde la sequía superficial se vuelve crítica.",
        }
        card1, card2, card3 = format_drought_score(value), rank, f"{vh:.2f}%"
        label1, label2, label3 = tr("drought_level"), tr("ten_year_rank"), tr("very_high")
    else:
        hvh = safe_float(risk_row, "high_very_high_percent")
        vh = safe_float(risk_row, "very_high_percent")
        prev = safe_float(prev_risk, "high_very_high_percent") if prev_risk is not None else np.nan
        rank = rank_text(risk_df, "high_very_high_percent", year, True)
        if np.isnan(prev) or abs(hvh - prev) < 0.1:
            change_en, change_fr, change_es = "remained stable", "est restée stable", "se mantuvo estable"
        elif hvh > prev:
            change_en, change_fr, change_es = "increased", "augmenté", "aumentado"
        else:
            change_en, change_fr, change_es = "decreased", "diminué", "disminuido"
        if layer_type == "Prototype 2024-2025":
            context_en = "The prototype zoom shows the current hotspot in detail, with severe risk concentrated at local scale."
            context_fr = "Le zoom prototype montre le hotspot actuel en détail, avec une concentration locale du risque sévère."
            context_es = "El zoom del prototipo muestra el hotspot actual en detalle, con concentración local del riesgo severo."
        else:
            context_en = "The map summarizes the spatial pattern of severe desertification risk across the park."
            context_fr = "La carte résume la distribution spatiale du risque sévère dans le parc."
            context_es = "El mapa resume el patrón espacial del riesgo severo en el parque."
        texts = {
            "en": f"High + Very High risk affects {hvh:.2f}% of the study area, including {vh:.2f}% Very High risk. Compared with the previous year, the severe-risk surface has {change_en}. {context_en}",
            "fr": f"Le risque élevé + très élevé affecte {hvh:.2f} % de la zone d’étude, dont {vh:.2f} % en risque très élevé. Par rapport à l’année précédente, la surface à risque sévère a {change_fr}. {context_fr}",
            "es": f"El riesgo alto + muy alto afecta al {hvh:.2f} % del área de estudio, incluido un {vh:.2f} % de riesgo muy alto. En comparación con el año anterior, la superficie de riesgo severo ha {change_es}. {context_es}",
        }
        card1, card2, card3 = f"{hvh:.2f}%", f"{vh:.2f}%", rank
        label1, label2, label3 = tr("high_very_high"), tr("very_high"), tr("ten_year_rank")

    text_analysis = texts.get(lang, texts["en"])
    return f"""
    <div class="deep-analysis-card">
        <div class="deep-analysis-title">{tr('detailed')} – {year}</div>
        <div class="deep-analysis-grid">
            <div class="deep-analysis-item"><div class="deep-analysis-label">{label1}</div><div class="deep-analysis-value">{card1}</div></div>
            <div class="deep-analysis-item"><div class="deep-analysis-label">{label2}</div><div class="deep-analysis-value">{card2}</div></div>
            <div class="deep-analysis-item"><div class="deep-analysis-label">{label3}</div><div class="deep-analysis-value">{card3}</div></div>
        </div>
        <div class="deep-analysis-text"><b>{tr('status')}:</b> {text_analysis}</div>
    </div>
    """



def render_dashboard(layer_type, year):
    info = LAYER_INFO[layer_type]
    metrics_html = selected_metrics_html(layer_type, year)

    col_map, col_side = st.columns([1.15, 0.85], gap="large")

    with col_map:
        st.markdown(f"## {lt(layer_type, 'title')}")
        st.markdown(f'<div class="small-text">{lt(layer_type, "description")}</div>', unsafe_allow_html=True)
        display_map(layer_type, year)

    with col_side:
        st.markdown(
            f"""
            <div class="side-panel">
                <div class="selected-year">{tr("selected_year")}: <span class="selected-year-value">{year}</span></div>
                {metrics_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if layer_type in ["Risk Classification", "Prototype 2024-2025"]:
            st.plotly_chart(risk_double_chart(year), use_container_width=True)
        else:
            st.plotly_chart(evolution_chart(layer_type, year), use_container_width=True)

        st.markdown(scientific_description_html(layer_type, year), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    info_col_1, info_col_2 = st.columns([1.0, 1.0], gap="large")
    with info_col_1:
        st.markdown(legend_html(layer_type), unsafe_allow_html=True)
    with info_col_2:
        st.markdown(map_metadata_html(layer_type), unsafe_allow_html=True)

    st.markdown('<div class="full-width-analysis">', unsafe_allow_html=True)
    st.markdown(detailed_analysis_html(layer_type, year), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if layer_type == "Risk Classification":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(risk_distribution_chart(year), use_container_width=True)
        st.markdown(
            f"""
            <div class="small-text">
            {tr("risk_distribution_note")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def animation_controls(selected_layer):
    if st.session_state.current_layer != selected_layer:
        st.session_state.current_layer = selected_layer
        st.session_state.animating = False
        st.session_state.animation_index = 0



# ============================================================
# SENTINEL-2 PRODUCTS EXPLORER SECTION – M. JOSÉ CORRECTION
# ============================================================

ROI_ASSET = "projects/graduationproject-493110/assets/ROI_Cabo_Gata_land_only"

RF_RISK_ASSETS = {
    "2015-2016": "projects/graduationproject-493110/assets/RF_Risk_2015_2016",
    "2016-2017": "projects/graduationproject-493110/assets/RF_Risk_2016_2017",
    "2017-2018": "projects/graduationproject-493110/assets/RF_Risk_2017_2018",
    "2018-2019": "projects/graduationproject-493110/assets/RF_Risk_2018_2019",
    "2019-2020": "projects/graduationproject-493110/assets/RF_Risk_2019_2020",
    "2020-2021": "projects/graduationproject-493110/assets/RF_Risk_2020_2021",
    "2021-2022": "projects/graduationproject-493110/assets/RF_Risk_2021_2022",
    "2022-2023": "projects/graduationproject-493110/assets/RF_Risk_2022_2023",
    "2023-2024": "projects/graduationproject-493110/assets/RF_Risk_2023_2024",
    "2024-2025": "projects/graduationproject-493110/assets/RF_Risk_2024_2025",
}


S2_EXPLORER_TEXT = {
    "en": {
        "section_title": "Sentinel-2 Products Explorer",
        "section_description": "The Sentinel-2 Products Explorer displays the complete validated catalog of image entries and generated products used in the project. Products can be filtered by hydrological year, Sentinel-2 image date and product type.",
        "image_entries": "Sentinel-2 image entries",
        "validated_catalog": "Validated image catalog",
        "generated_products": "Generated products",
        "products_per_image": "Products per image",
        "dynamic_selection": "Displayed dynamically by selection",
        "hydrological_year": "Hydrological year",
        "image_entry": "Sentinel-2 image date / image entry",
        "generated_product": "Generated product",
        "sentinel_date": "Sentinel-2 image date",
        "ee_error": "Earth Engine is not initialized. The catalog is visible, but the selected map cannot be loaded yet. For local testing, authenticate Earth Engine. For Streamlit Cloud, add Earth Engine credentials in secrets.",
        "missing_files": "Sentinel-2 Products Explorer CSV files were not found. Please add Sentinel2_Image_Dates_Catalog_READY_2074.csv and Sentinel2_Products_Catalog_READY_8296.csv to the data folder or repository root.",
        "expander": "View complete indexed products for selected hydrological year",
        "download": "Download complete 8,296 products catalog",
        "legend_ndvi": "NDVI Classification",
        "legend_ndwi": "NDWI Classification",
        "legend_nddi": "NDDI Classification",
        "legend_risk": "Desertification Risk Classes",
        "legend_note_report": "Same classification colors as the final report.",
        "product_risk": "RF Risk Map",
        "product_name": "Product name",
        "product_type": "Product type",
        "image_date": "Image date",
    },
    "fr": {
        "section_title": "Explorateur des produits Sentinel-2",
        "section_description": "L’explorateur des produits Sentinel-2 affiche le catalogue validé complet des entrées d’images et des produits générés utilisés dans le projet. Les produits peuvent être filtrés par année hydrologique, date d’image Sentinel-2 et type de produit.",
        "image_entries": "Entrées d’images Sentinel-2",
        "validated_catalog": "Catalogue d’images validé",
        "generated_products": "Produits générés",
        "products_per_image": "Produits par image",
        "dynamic_selection": "Affiché dynamiquement selon la sélection",
        "hydrological_year": "Année hydrologique",
        "image_entry": "Date d’image Sentinel-2 / entrée d’image",
        "generated_product": "Produit généré",
        "sentinel_date": "Date de l’image Sentinel-2",
        "ee_error": "Earth Engine n’est pas initialisé. Le catalogue est visible, mais la carte sélectionnée ne peut pas encore être chargée. Pour un test local, authentifiez Earth Engine. Pour Streamlit Cloud, ajoutez les identifiants Earth Engine dans les secrets.",
        "missing_files": "Les fichiers CSV de l’explorateur des produits Sentinel-2 sont introuvables. Ajoutez Sentinel2_Image_Dates_Catalog_READY_2074.csv et Sentinel2_Products_Catalog_READY_8296.csv dans le dossier data ou à la racine du dépôt.",
        "expander": "Voir les produits indexés pour l’année hydrologique sélectionnée",
        "download": "Télécharger le catalogue complet des 8 296 produits",
        "legend_ndvi": "Classification NDVI",
        "legend_ndwi": "Classification NDWI",
        "legend_nddi": "Classification NDDI",
        "legend_risk": "Classes de risque de désertification",
        "legend_note_report": "Même symbologie que dans le rapport final.",
        "product_risk": "Carte de risque RF",
        "product_name": "Nom du produit",
        "product_type": "Type de produit",
        "image_date": "Date de l’image",
    },
    "es": {
        "section_title": "Explorador de productos Sentinel-2",
        "section_description": "El explorador de productos Sentinel-2 muestra el catálogo validado completo de entradas de imagen y productos generados utilizados en el proyecto. Los productos pueden filtrarse por año hidrológico, fecha de imagen Sentinel-2 y tipo de producto.",
        "image_entries": "Entradas de imagen Sentinel-2",
        "validated_catalog": "Catálogo de imágenes validado",
        "generated_products": "Productos generados",
        "products_per_image": "Productos por imagen",
        "dynamic_selection": "Mostrado dinámicamente según la selección",
        "hydrological_year": "Año hidrológico",
        "image_entry": "Fecha de imagen Sentinel-2 / entrada de imagen",
        "generated_product": "Producto generado",
        "sentinel_date": "Fecha de la imagen Sentinel-2",
        "ee_error": "Earth Engine no está inicializado. El catálogo es visible, pero la carta seleccionada aún no puede cargarse. Para pruebas locales, autentica Earth Engine. Para Streamlit Cloud, añade las credenciales de Earth Engine en los secrets.",
        "missing_files": "No se encontraron los archivos CSV del explorador de productos Sentinel-2. Añade Sentinel2_Image_Dates_Catalog_READY_2074.csv y Sentinel2_Products_Catalog_READY_8296.csv a la carpeta data o a la raíz del repositorio.",
        "expander": "Ver productos indexados para el año hidrológico seleccionado",
        "download": "Descargar el catálogo completo de 8.296 productos",
        "legend_ndvi": "Clasificación NDVI",
        "legend_ndwi": "Clasificación NDWI",
        "legend_nddi": "Clasificación NDDI",
        "legend_risk": "Clases de riesgo de desertificación",
        "legend_note_report": "Mismos colores de clasificación que en el informe final.",
        "product_risk": "Mapa de riesgo RF",
        "product_name": "Nombre del producto",
        "product_type": "Tipo de producto",
        "image_date": "Fecha de imagen",
    },
}


def s2tr(key):
    lang = get_lang()
    return S2_EXPLORER_TEXT.get(lang, S2_EXPLORER_TEXT["en"]).get(
        key,
        S2_EXPLORER_TEXT["en"].get(key, key),
    )


def s2_product_label(product_type):
    labels = {
        "NDVI": "NDVI",
        "NDWI": "NDWI",
        "NDDI": "NDDI",
        "RF Risk Map": s2tr("product_risk"),
    }
    return labels.get(product_type, product_type)


def s2_table_column_labels():
    return {
        "hydrological_year": s2tr("hydrological_year"),
        "image_date": s2tr("image_date"),
        "image_entry": s2tr("image_entry"),
        "product_type": s2tr("product_type"),
        "product_name": s2tr("product_name"),
    }


@st.cache_resource(show_spinner=False)
def initialize_earth_engine():
    """
    Initialize Earth Engine for the dynamic Sentinel-2 product map.

    Local use:
        Works after running Earth Engine authentication on the computer.

    Streamlit Cloud use:
        Add a service-account JSON in Streamlit Secrets using either:
        GEE_SERVICE_ACCOUNT_JSON containing the full JSON text,
        or a TOML table named [gee].
    """
    default_project_id = "graduationproject-493110"

    # 1) Streamlit Cloud: easiest format, one secret containing the full JSON.
    try:
        if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:
            gee_info = json.loads(st.secrets["GEE_SERVICE_ACCOUNT_JSON"])
            service_account = gee_info.get("client_email")
            project_id = gee_info.get("project_id", default_project_id)
            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=json.dumps(gee_info),
            )
            ee.Initialize(credentials, project=project_id)
            return True, None
    except Exception as secret_json_error:
        secret_json_message = str(secret_json_error)
    else:
        secret_json_message = None

    # 2) Streamlit Cloud: alternative TOML table format.
    try:
        if "gee" in st.secrets:
            gee_info = dict(st.secrets["gee"])
            service_account = gee_info.get("client_email")
            project_id = gee_info.get("project_id", default_project_id)
            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=json.dumps(gee_info),
            )
            ee.Initialize(credentials, project=project_id)
            return True, None
    except Exception as secret_table_error:
        secret_table_message = str(secret_table_error)
    else:
        secret_table_message = None

    # 3) Local computer: use the Earth Engine authentication already saved locally.
    try:
        ee.Initialize(project=default_project_id)
        return True, None
    except Exception as local_error:
        details = [str(local_error)]
        if secret_json_message:
            details.append("GEE_SERVICE_ACCOUNT_JSON error: " + secret_json_message)
        if secret_table_message:
            details.append("[gee] secrets error: " + secret_table_message)
        return False, " | ".join(details)


@st.cache_data(show_spinner=False)
def load_sentinel2_products_explorer_files():
    """
    Load the final validated Sentinel-2 catalogs:
    - 2,074 image entries
    - 8,296 generated products
    The function searches first in /data and then in the repository root.
    """
    image_files = [
        DATA_DIR / "Sentinel2_Image_Dates_Catalog_READY_2074.csv",
        BASE_DIR / "Sentinel2_Image_Dates_Catalog_READY_2074.csv",
        DATA_DIR / "Sentinel2_Image_Dates_Catalog_2015_2025_FINAL_2074.csv",
        BASE_DIR / "Sentinel2_Image_Dates_Catalog_2015_2025_FINAL_2074.csv",
    ]

    product_files = [
        DATA_DIR / "Sentinel2_Products_Catalog_READY_8296.csv",
        BASE_DIR / "Sentinel2_Products_Catalog_READY_8296.csv",
        DATA_DIR / "Sentinel2_Products_Catalog_2015_2025_FINAL_8296.csv",
        BASE_DIR / "Sentinel2_Products_Catalog_2015_2025_FINAL_8296.csv",
    ]

    summary_files = [
        DATA_DIR / "Sentinel2_Products_Summary_READY.csv",
        BASE_DIR / "Sentinel2_Products_Summary_READY.csv",
    ]

    image_path = next((p for p in image_files if p.exists()), None)
    product_path = next((p for p in product_files if p.exists()), None)
    summary_path = next((p for p in summary_files if p.exists()), None)

    if image_path is None or product_path is None:
        return None, None, None

    image_df = pd.read_csv(image_path).dropna(how="all")
    product_df = pd.read_csv(product_path).dropna(how="all")
    summary_df = pd.read_csv(summary_path).dropna(how="all") if summary_path else None

    image_df["image_date"] = pd.to_datetime(image_df["image_date"]).dt.strftime("%Y-%m-%d")
    product_df["image_date"] = pd.to_datetime(product_df["image_date"]).dt.strftime("%Y-%m-%d")

    # Make sure the dashboard always has the full Earth Engine image asset path.
    for df in [image_df, product_df]:
        if "ee_image_asset" not in df.columns:
            df["ee_image_asset"] = "COPERNICUS/S2_SR_HARMONIZED/" + df["system_index"].astype(str)

        if "image_entry" not in df.columns:
            df["image_entry"] = df["image_date"].astype(str) + " | " + df["system_index"].astype(str)

    image_df = image_df.sort_values(["hydrological_year", "image_date", "system_index"])
    product_df = product_df.sort_values(["hydrological_year", "image_date", "image_entry", "product_type"])

    return image_df, product_df, summary_df


def get_ee_image_asset(row):
    """Return a complete Earth Engine image path for the selected Sentinel-2 image."""
    if "ee_image_asset" in row and pd.notna(row["ee_image_asset"]):
        value = str(row["ee_image_asset"])
        if value.startswith("COPERNICUS/"):
            return value

    if "ee_image_id" in row and pd.notna(row["ee_image_id"]):
        value = str(row["ee_image_id"])
        if value.startswith("COPERNICUS/"):
            return value

    return "COPERNICUS/S2_SR_HARMONIZED/" + str(row["system_index"])


def mask_sentinel2_l2a(img, roi_geom):
    """Apply the same SCL cloud/shadow mask used for the project products."""
    scl = img.select("SCL")

    mask = (
        scl.neq(3)       # cloud shadow
        .And(scl.neq(8)) # medium cloud probability
        .And(scl.neq(9)) # high cloud probability
        .And(scl.neq(10))# cirrus
        .And(scl.neq(11))# snow / ice
    )

    scaled = img.select(["B2", "B3", "B4", "B8", "B11", "B12"]).multiply(0.0001)

    return (
        img.addBands(scaled, None, True)
        .updateMask(mask)
        .clip(roi_geom)
    )


def add_sentinel2_indices(img):
    """Calculate NDVI, NDWI and NDDI for the selected Sentinel-2 image date."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B8", "B11"]).rename("NDWI")

    denominator = ndvi.add(ndwi)
    safe_denominator = denominator.where(denominator.abs().lt(0.0001), 0.0001)

    nddi = ndvi.subtract(ndwi).divide(safe_denominator).rename("NDDI")

    return img.addBands([ndvi, ndwi, nddi])


def classify_ndvi(img, roi_geom):
    """Classify NDVI with the same classes/colors used in the final report."""
    ndvi = img.select("NDVI")

    return (
        ee.Image(1)
        .where(ndvi.gt(0.10).And(ndvi.lte(0.20)), 2)
        .where(ndvi.gt(0.20).And(ndvi.lte(0.40)), 3)
        .where(ndvi.gt(0.40).And(ndvi.lte(0.60)), 4)
        .where(ndvi.gt(0.60), 5)
        .rename("NDVI_Class")
        .toByte()
        .updateMask(ndvi.mask())
        .clip(roi_geom)
    )


def classify_ndwi(img, roi_geom):
    """Classify NDWI with the same classes/colors used in the final report."""
    ndwi = img.select("NDWI")

    return (
        ee.Image(1)
        .where(ndwi.gt(-0.25).And(ndwi.lte(-0.10)), 2)
        .where(ndwi.gt(-0.10).And(ndwi.lte(0.10)), 3)
        .where(ndwi.gt(0.10).And(ndwi.lte(0.30)), 4)
        .where(ndwi.gt(0.30), 5)
        .rename("NDWI_Class")
        .toByte()
        .updateMask(ndwi.mask())
        .clip(roi_geom)
    )


def classify_nddi(img, roi_geom):
    """Classify NDDI with the same classes/colors used in the final report."""
    nddi = img.select("NDDI")

    return (
        ee.Image(1)
        .where(nddi.gt(0.10).And(nddi.lte(0.20)), 2)
        .where(nddi.gt(0.20).And(nddi.lte(0.40)), 3)
        .where(nddi.gt(0.40).And(nddi.lte(0.60)), 4)
        .where(nddi.gt(0.60), 5)
        .rename("NDDI_Class")
        .toByte()
        .updateMask(nddi.mask())
        .clip(roi_geom)
    )


def get_selected_product_image(selected_row, selected_product):
    """Build the selected map layer.

    NDVI, NDWI and NDDI are generated from the selected Sentinel-2 acquisition date.
    A same-date mosaic is used to reduce tile gaps while keeping the selected date visible.
    The RF Risk Map uses the final RF asset for the selected hydrological year.
    """
    roi = ee.FeatureCollection(ROI_ASSET)
    roi_geom = roi.geometry()

    selected_date = pd.to_datetime(selected_row["image_date"])
    start_date = selected_date.strftime("%Y-%m-%d")
    end_date = (selected_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if selected_product in ["NDVI", "NDWI", "NDDI"]:

        def prepare_image(img):
            img = mask_sentinel2_l2a(img, roi_geom)
            img = add_sentinel2_indices(img)
            return img

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi_geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(prepare_image)
        )

        # Same-date median mosaic. This keeps the selected acquisition date
        # while reducing missing areas caused by tile limits, clouds or shadows.
        img = collection.median().clip(roi_geom)

        if selected_product == "NDVI":
            product_image = classify_ndvi(img, roi_geom)
            colors = [c.replace("#", "") for c, _ in LEGEND_INFO["NDVI"][:5]]

        elif selected_product == "NDWI":
            product_image = classify_ndwi(img, roi_geom)
            colors = [c.replace("#", "") for c, _ in LEGEND_INFO["NDWI"][:5]]

        else:
            product_image = classify_nddi(img, roi_geom)
            colors = [c.replace("#", "") for c, _ in LEGEND_INFO["NDDI"][:5]]

        return product_image, {"min": 1, "max": 5, "palette": colors}

    if selected_product == "RF Risk Map":
        hydro_year = str(selected_row["hydrological_year"])

        product_image = (
            ee.Image(RF_RISK_ASSETS[hydro_year])
            .select(0)
            .rename("Risk_Map")
            .toByte()
            .clip(roi_geom)
        )

        colors = [c.replace("#", "") for c, _ in LEGEND_INFO["Risk Classification"][:5]]
        return product_image, {"min": 1, "max": 5, "palette": colors}

    raise ValueError(f"Unknown selected product: {selected_product}")



def products_explorer_legend_html(product_type):
    """Compact HTML legend for the selected product.

    The HTML is generated without leading indentation so Streamlit does not
    interpret any part of it as a Markdown code block.
    """
    layer_key = "Risk Classification" if product_type == "RF Risk Map" else product_type
    items = LEGEND_INFO[layer_key][:5]

    labels = LEGEND_LABELS.get(layer_key, {}).get(get_lang())
    if labels:
        labels = labels[:5]
        items = [(items[i][0], labels[i]) for i in range(5)]

    title = {
        "NDVI": s2tr("legend_ndvi"),
        "NDWI": s2tr("legend_ndwi"),
        "NDDI": s2tr("legend_nddi"),
        "RF Risk Map": s2tr("legend_risk"),
    }.get(product_type, "Legend")

    item_html = ""
    for color, label in items:
        item_html += (
            '<div class="legend-item">'
            f'<span class="legend-swatch" style="background:{color};"></span>'
            f'<span>{label}</span>'
            '</div>'
        )

    return (
        '<div class="legend-card">'
        f'<div class="legend-title">{title}</div>'
        f'<div class="legend-grid">{item_html}</div>'
        '</div>'
    )


def add_ee_layer_to_folium_map(folium_map, ee_object, vis_params, name):
    """Add an Earth Engine object to a Folium map without using geemap.

    This avoids the geemap/xyz_to_folium compatibility error on some Windows/Python versions.
    """
    map_id = ee_object.getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    ).add_to(folium_map)

def sentinel2_products_explorer_section():
    """New dashboard section replacing the old Sentinel-2 summary table."""
    image_df, product_df, summary_df = load_sentinel2_products_explorer_files()

    if image_df is None or product_df is None:
        st.warning(s2tr("missing_files"))
        return

    ee_ok, ee_error = initialize_earth_engine()

    st.markdown(
        f"""
<div class="section-card">
    <h2>{s2tr("section_title")}</h2>
    <div class="small-text">
        {s2tr("section_description")}
    </div>
    <div class="s2-summary-grid">
        <div class="s2-summary-card">
            <div class="s2-summary-title">{s2tr("image_entries")}</div>
            <div class="s2-summary-value">2,074</div>
            <div class="s2-summary-subtext">{s2tr("validated_catalog")}</div>
        </div>
        <div class="s2-summary-card">
            <div class="s2-summary-title">{s2tr("generated_products")}</div>
            <div class="s2-summary-value">8,296</div>
            <div class="s2-summary-subtext">NDVI · NDWI · NDDI · {s2_product_label("RF Risk Map")}</div>
        </div>
        <div class="s2-summary-card">
            <div class="s2-summary-title">{s2tr("products_per_image")}</div>
            <div class="s2-summary-value">4</div>
            <div class="s2-summary-subtext">{s2tr("dynamic_selection")}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    years_available = sorted(image_df["hydrological_year"].dropna().unique())

    c1, c2, c3 = st.columns([1.1, 2.2, 1.1], gap="medium")

    with c1:
        selected_year = st.selectbox(
            s2tr("hydrological_year"),
            years_available,
            index=len(years_available) - 1,
            key="s2_products_year",
        )

    year_df = image_df[image_df["hydrological_year"] == selected_year].copy()

    with c2:
        selected_entry = st.selectbox(
            s2tr("image_entry"),
            year_df["image_entry"].tolist(),
            key="s2_products_image_entry",
        )

    product_options = ["NDVI", "NDWI", "NDDI", "RF Risk Map"]
    with c3:
        selected_product = st.selectbox(
            s2tr("generated_product"),
            product_options,
            format_func=s2_product_label,
            key="s2_products_product",
        )

    selected_row = year_df[year_df["image_entry"] == selected_entry].iloc[0]
    selected_product_label = s2_product_label(selected_product)

    st.markdown(
        f"""
<div style="
    background: rgba(8,16,24,0.82);
    border: 1px solid rgba(242,201,139,0.22);
    border-radius: 16px;
    padding: 12px 16px;
    margin-top: 0.6rem;
    margin-bottom: 0.7rem;
    text-align: center;
    box-shadow: 0 10px 28px rgba(0,0,0,0.26);
">
    <div style="color:#F2C98B; font-size:1.15rem; font-weight:900;">
        {selected_product_label}
    </div>
    <div style="color:#F6F4F0; font-size:1.05rem; margin-top:4px;">
        {s2tr("sentinel_date")}: <b>{selected_row['image_date']}</b>
    </div>
    <div style="color:#E6D2A9; font-size:0.88rem; margin-top:2px;">
        {s2tr("hydrological_year")}: {selected_year}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # NDVI / NDWI / NDDI are loaded dynamically from Earth Engine.
    # The RF Risk Map is displayed from the already exported annual layout image.
    # This avoids private-asset access errors while still showing the selected product
    # with the selected Sentinel-2 date and hydrological year.
    if selected_product == "RF Risk Map":
        display_map("Risk Classification", selected_year)
        st.markdown(products_explorer_legend_html(selected_product), unsafe_allow_html=True)

    elif not ee_ok:
        st.error(s2tr("ee_error"))
        st.code(str(ee_error))

    else:
        product_image, vis_params = get_selected_product_image(selected_row, selected_product)

        m = folium.Map(
            location=[36.85, -2.10],
            zoom_start=10,
            tiles="Esri.WorldImagery",
            attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        )

        add_ee_layer_to_folium_map(
            m,
            product_image,
            vis_params,
            f"{selected_product_label} | {s2tr('sentinel_date')}: {selected_row['image_date']}",
        )

        roi = ee.FeatureCollection(ROI_ASSET)
        add_ee_layer_to_folium_map(
            m,
            roi.style(**{
                "color": "F6F4F0",
                "fillColor": "00000000",
                "width": 2,
            }),
            {},
            "Study area boundary",
        )

        folium.LayerControl(collapsed=False).add_to(m)
        _map_state = st_folium(m, width=1250, height=620)
        st.markdown(products_explorer_legend_html(selected_product), unsafe_allow_html=True)

    selected_year_products = product_df[product_df["hydrological_year"] == selected_year].copy()

    with st.expander(s2tr("expander"), expanded=False):
        cols_to_show = [
            "hydrological_year",
            "image_date",
            "image_entry",
            "product_type",
            "product_name",
        ]
        available_cols = [c for c in cols_to_show if c in selected_year_products.columns]
        table_df = selected_year_products[available_cols].copy()
        if "product_type" in table_df.columns:
            table_df["product_type"] = table_df["product_type"].map(s2_product_label)
        table_df = table_df.rename(columns=s2_table_column_labels())
        st.dataframe(table_df, use_container_width=True, height=420)

    csv_download = product_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        s2tr("download"),
        data=csv_download,
        file_name="Sentinel2_Products_Catalog_READY_8296.csv",
        mime="text/csv",
        key="download_complete_sentinel2_products_catalog",
    )


# ============================================================
# INITIAL STATE
# ============================================================

initialize_state()

# ============================================================
# MAIN PAGE
# ============================================================

# Language selector
# Placed at the top-left of the dashboard, below the Streamlit top bar.
# No global selectbox CSS is used, so the Layer type selector keeps its normal size.
lang_left, lang_right = st.columns([0.12, 0.88])
with lang_left:
    language_choice = st.selectbox(
        "Language",
        list(LANG_OPTIONS.keys()),
        index=0,
        key="language_choice",
    )
st.session_state.language_code = LANG_OPTIONS[language_choice]

st.markdown(
    f"""
    <div class="hero-card">
        <h1>{tr('app_title')}</h1>
        <div class="subtitle">{tr('subtitle')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

sentinel2_products_explorer_section()

st.markdown(f'<div class="control-panel"><div class="control-title">{tr("controls")}</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns([1.25, 1.25, 1.15, 0.9, 0.65], gap="medium")

with c1:
    layer_type = st.selectbox(
        tr("layer_type"),
        ["NDVI", "NDWI", "NDDI", "Risk Classification", "Prototype 2024-2025"],
        key="main_layer_type",
        format_func=lambda x: LAYER_SELECTOR_LABELS.get(x, x),
    )

animation_controls(layer_type)

with c2:
    selected_year = st.select_slider(
        tr("hydro_year"),
        options=YEARS,
        value="2024-2025",
        key="main_selected_year",
    )

with c3:
    speed_key = st.radio(
        tr("animation_speed"),
        ["slow", "normal", "fast"],
        index=2,
        horizontal=False,
        key="main_speed",
        format_func=lambda x: tr(f"speed_{x}"),
    )

speed_seconds = {"slow": 0.35, "normal": 0.20, "fast": 0.05}[speed_key]

with c4:
    st.write("")
    st.write("")
    if st.button(tr("auto"), use_container_width=True):
        st.session_state.animating = True
        st.session_state.animation_index = 0
        st.rerun()

with c5:
    st.write("")
    st.write("")
    if st.button(tr("stop"), use_container_width=True):
        st.session_state.animating = False
        st.session_state.animation_index = 0
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

if layer_type == "Prototype 2024-2025":
    current_year = "2024-2025"
    st.info(tr("fixed_proto"))
elif st.session_state.animating:
    current_year = YEARS[st.session_state.animation_index]
    st.info(f"{tr('auto_running')}: {current_year}")
else:
    current_year = selected_year

render_dashboard(layer_type, current_year)

# During auto visualization, skip the lower heavy sections to keep the animation fast.
# They return automatically when the animation stops.
if not st.session_state.animating:
    st.markdown("---")
    scenario_prediction_section()

    st.markdown("---")
    st.markdown(
        f"""
        <div class="deep-analysis-card">
            <div class="deep-analysis-title">{tr('synthesis_title')}</div>
            <div class="deep-analysis-text">{tr('synthesis_text')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ANIMATION LOOP
# ============================================================

if st.session_state.animating:
    time.sleep(speed_seconds)
    if st.session_state.animation_index < len(YEARS) - 1:
        st.session_state.animation_index += 1
    else:
        st.session_state.animating = False
        st.session_state.animation_index = 0
    st.rerun()


# ============================================================
# FINAL PHONE READABILITY OVERRIDE
# This must stay near the end of app.py so it overrides earlier CSS.
# ============================================================

st.markdown(
    """
    <style>
    /* Make all large dashboard number cards readable on smartphones */
    .prediction-grid,
    .s2-summary-grid {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 8px !important;
    }

    .prediction-card,
    .s2-summary-card {
        padding: 8px 7px !important;
        border-radius: 12px !important;
        min-width: 0 !important;
        overflow: hidden !important;
    }

    .prediction-title,
    .s2-summary-title {
        font-size: 0.68rem !important;
        line-height: 1.20 !important;
        margin-bottom: 0.18rem !important;
    }

    .prediction-value,
    .s2-summary-value {
        font-size: 0.95rem !important;
        line-height: 1.12 !important;
        font-weight: 850 !important;
        letter-spacing: 0 !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
        white-space: normal !important;
    }

    .prediction-subtext,
    .s2-summary-subtext {
        font-size: 0.58rem !important;
        line-height: 1.22 !important;
        margin-top: 0.15rem !important;
    }

    /* Specifically avoid huge percentage values on the scenario cards */
    .prediction-card .prediction-value {
        font-size: 0.92rem !important;
    }

    /* Phone: make cards vertical instead of squeezed */
    @media (max-width: 760px) {
        .prediction-grid,
        .s2-summary-grid {
            grid-template-columns: 1fr !important;
            gap: 8px !important;
        }

        .prediction-card,
        .s2-summary-card {
            padding: 9px 10px !important;
        }

        .prediction-title,
        .s2-summary-title {
            font-size: 0.78rem !important;
        }

        .prediction-value,
        .s2-summary-value,
        .prediction-card .prediction-value {
            font-size: 1.10rem !important;
            line-height: 1.18 !important;
        }

        .prediction-subtext,
        .s2-summary-subtext {
            font-size: 0.70rem !important;
        }
    }

    /* Very small phone screens */
    @media (max-width: 430px) {
        .prediction-value,
        .s2-summary-value,
        .prediction-card .prediction-value {
            font-size: 0.95rem !important;
        }

        .section-card h2,
        h2 {
            font-size: 1.02rem !important;
        }

        .small-text {
            font-size: 0.78rem !important;
            line-height: 1.38 !important;
        }
    }
    
    /* Tiny top summary values */
    .s2-summary-value {
        font-size: 1.05rem !important;
        line-height: 1.12 !important;
    }

    .s2-summary-title {
        font-size: 0.72rem !important;
    }

    .s2-summary-subtext {
        font-size: 0.62rem !important;
    }

    /* Final generated products text */
    .s2-summary-card div {
        word-break: normal !important;
        overflow-wrap: normal !important;
    }
</style>
    """,
    unsafe_allow_html=True,
)

