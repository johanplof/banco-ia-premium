from sklearn.neural_network import MLPClassifier
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

# ============================================
# CARGAR API KEY
# ============================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================
# CONFIGURACIÓN
# ============================================

st.set_page_config(
    page_title="Banco IA Premium",
    page_icon="🏦",
    layout="wide"
)

# ============================================
# CSS
# ============================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: Arial;
}

.block-container {
    padding-top: 2rem;
}

.chat-box {
    background: #111827;
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 15px;
    border: 1px solid #1f2937;
}

.user-box {
    background: #1e293b;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
}

.ai-box {
    background: #111827;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
}

.sidebar {
    background: #020617;
}

.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# IA NEURONAL
# ============================================

X = np.array([
    [5000, 1000, 30],
    [2000, 1500, 45],
    [7000, 500, 28],
    [1500, 1200, 50],
    [9000, 2000, 35],
    [1000, 900, 60]
])

y = np.array([1, 0, 1, 0, 1, 0])

modelo = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=5000
)

modelo.fit(X, y)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2830/2830284.png",
        width=120
    )

    st.title("🏦 Banco IA")

    st.success("🟢 Sistema activo")

    st.markdown("## 📈 Créditos evaluados")
    st.markdown("# 12,584")

    st.markdown("## 💰 Capital analizado")
    st.markdown("# $8.2M")

    st.divider()

    st.markdown("""
### 🚀 Características

✅ Evaluación con IA  
✅ Red neuronal  
✅ Machine Learning  
✅ Riesgo crediticio  
✅ OpenAI GPT  
✅ Dashboard analítico  
""")

# ============================================
# TÍTULO
# ============================================

st.title("🏦 Banco IA Premium")

st.markdown("""
Sistema financiero inteligente basado en IA neuronal,
machine learning y evaluación crediticia avanzada.
""")

# ============================================
# CHAT MEMORY
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = 0

if "data" not in st.session_state:
    st.session_state.data = {}

# ============================================
# PREGUNTAS
# ============================================

preguntas = [
    ("ingresos", "💰 ¿Cuáles son tus ingresos mensuales?"),
    ("deudas", "📉 ¿Cuánto tienes en deudas actualmente?"),
    ("edad", "👤 ¿Cuál es tu edad?"),
    ("historial", "📊 ¿Cómo es tu historial crediticio? (bueno, regular, malo)"),
    ("tipo", "🏦 ¿Qué tipo de crédito deseas? (Vivienda, Vehículo, Consumo)"),
    ("monto", "💵 ¿Qué monto deseas solicitar?"),
    ("meses", "📅 ¿A cuántos meses?")
]

# ============================================
# MOSTRAR CHAT
# ============================================

for m in st.session_state.messages:

    with st.chat_message(
        m["role"],
        avatar="🏦" if m["role"] == "assistant" else "👤"
    ):
        st.markdown(m["content"])

# ============================================
# PRIMER MENSAJE
# ============================================

if len(st.session_state.messages) == 0:

    bienvenida = """
# 👋 Bienvenido a Banco IA Premium

Soy tu analista financiero inteligente.

Te ayudaré a evaluar tu perfil crediticio mediante IA neuronal.
"""

    st.session_state.messages.append({
        "role": "assistant",
        "content": bienvenida
    })

# ============================================
# HACER PREGUNTAS
# ============================================

if st.session_state.step < len(preguntas):

    clave_actual, pregunta_actual = preguntas[
        st.session_state.step
    ]

    with st.chat_message(
        "assistant",
        avatar="🏦"
    ):
        st.markdown(pregunta_actual)

# ============================================
# INPUT
# ============================================

respuesta = st.chat_input(
    "Escribe tu respuesta..."
)

# ============================================
# PROCESAR RESPUESTA
# ============================================

if respuesta:

    # MOSTRAR MENSAJE USUARIO
    st.session_state.messages.append({
        "role": "user",
        "content": respuesta
    })

    # GUARDAR RESPUESTA
    if st.session_state.step < len(preguntas):

        clave_actual, pregunta_actual = preguntas[
            st.session_state.step
        ]

        st.session_state.data[
            clave_actual
        ] = respuesta

        st.session_state.step += 1

        st.rerun()

# ============================================
# RESULTADO FINAL
# ============================================

if st.session_state.step >= len(preguntas):

    datos = st.session_state.data

    ingresos = float(datos["ingresos"])
    deudas = float(datos["deudas"])
    edad = float(datos["edad"])
    historial = datos["historial"]
    tipo = datos["tipo"]
    monto = float(datos["monto"])
    meses = int(datos["meses"])

    prediccion = modelo.predict([
        [ingresos, deudas, edad]
    ])

    aprobado = bool(prediccion[0])

    score = 85 if aprobado else 45

    riesgo = (
        "🟢 Bajo riesgo"
        if aprobado
        else "🔴 Alto riesgo"
    )

    tasa = 12 if aprobado else 25

    tasa_mensual = tasa / 12 / 100

    cuota = (
        monto *
        (
            tasa_mensual *
            (1 + tasa_mensual) ** meses
        )
        /
        (
            ((1 + tasa_mensual) ** meses) - 1
        )
    )

    # ========================================
    # OPENAI ANALYSIS
    # ========================================

    prompt = f"""
    Analiza este cliente bancario:

    Ingresos: {ingresos}
    Deudas: {deudas}
    Edad: {edad}
    Historial: {historial}
    Crédito: {tipo}
    Monto: {monto}

    Score: {score}
    Riesgo: {riesgo}

    Explica profesionalmente:
    - perfil financiero
    - riesgos
    - recomendación
    """

    respuesta_ia = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Eres un analista financiero senior."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    analisis = respuesta_ia.choices[0].message.content

    # ========================================
    # RESULTADO
    # ========================================

    st.divider()

    st.subheader("📋 Resultado final")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Score IA", score)

    with col2:
        st.metric("🏦 Riesgo", riesgo)

    with col3:
        st.metric("💵 Cuota", f"${cuota:,.0f}")

    if aprobado:
        st.success("✅ Crédito aprobado")
    else:
        st.error("❌ Crédito rechazado")

    st.markdown("## 🤖 Análisis IA")

    st.info(analisis)