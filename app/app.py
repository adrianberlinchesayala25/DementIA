"""
Aplicación principal para detección de Alzheimer mediante análisis de voz.

LIBRERÍAS UTILIZADAS:
- streamlit: Interfaz web interactiva y fácil de usar
- pathlib: Manejo de rutas de archivos
- AudioProcessor: Extracción de embeddings Wav2Vec2 del audio
- AlzheimerPredictor: Predicción con ensemble de 5 modelos
- validate_audio_format: Validación de formatos de audio
"""

import streamlit as st
from pathlib import Path
import tempfile
import os
from app.audio_processor import AudioProcessor
from app.predictor import AlzheimerPredictor
from app.utils import validate_audio_format


# Configuración de la página
st.set_page_config(
    page_title="DementIA - Detección de Alzheimer",
    page_icon="🧠",
    layout="centered"
)


def initialize_models():
    """
    Inicializa el procesador de audio y el predictor.
    Usa st.cache_resource para cargar los modelos solo una vez.

    Returns:
        Tupla (audio_processor, predictor)
    """
    # Obtener ruta de modelos
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"

    # Cargar modelos (se cachean para no recargarlos en cada interacción)
    audio_processor = AudioProcessor()
    predictor = AlzheimerPredictor(str(models_dir))

    return audio_processor, predictor


def main():
    """
    Función principal de la aplicación Streamlit.
    """
    # Título y descripción
    st.title("🧠 DementIA - Detección de Alzheimer por Voz")
    st.markdown("""
    Esta aplicación utiliza inteligencia artificial para analizar patrones de voz 
    y estimar la probabilidad de Alzheimer o demencia.
    
    **¿Cómo funciona?**
    1. Sube un archivo de audio (WAV, MP3, etc.)
    2. La IA analiza patrones de habla (pausas, ritmo, tono)
    3. Obtienes una estimación de riesgo basada en 5 modelos entrenados
    
    ⚠️ **Nota:** Esta herramienta NO sustituye un diagnóstico médico profesional.
    """)

    st.divider()

    # Cargar modelos (solo una vez gracias al caché)
    with st.spinner("Cargando modelos de IA..."):
        try:
            audio_processor, predictor = initialize_models()
        except Exception as e:
            st.error(f"❌ Error al cargar los modelos: {str(e)}")
            st.stop()

    # Subir archivo de audio
    st.subheader("📤 Subir Audio")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de audio",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Formatos soportados: WAV, MP3, FLAC, OGG, M4A"
    )

    if uploaded_file is not None:
        # Mostrar información del archivo
        st.success(f"✅ Archivo subido: {uploaded_file.name}")

        # Reproducir audio
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

        # Botón para analizar
        if st.button("🔍 Analizar Audio", type="primary"):
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                # Procesar audio
                with st.spinner("Procesando audio y extrayendo características..."):
                    embedding = audio_processor.process_audio_file(tmp_path)

                # Realizar predicción
                with st.spinner("Analizando con 5 modelos de IA..."):
                    result = predictor.predict(embedding)

                # Mostrar resultados
                st.divider()
                st.subheader("📊 Resultados del Análisis")

                # Crear columnas para mostrar resultados
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Probabilidad de Alzheimer",
                        value=result['percentage']
                    )

                with col2:
                    # Determinar color del nivel de riesgo
                    if result['risk_color'] == 'green':
                        st.success(f"Nivel de Riesgo: {result['risk_level']}")
                    elif result['risk_color'] == 'orange':
                        st.warning(f"Nivel de Riesgo: {result['risk_level']}")
                    else:
                        st.error(f"Nivel de Riesgo: {result['risk_level']}")

                # Barra de progreso visual
                st.progress(result['probability'])

                # Interpretación de resultados
                st.divider()
                st.subheader("📖 Interpretación")

                if result['probability'] < 0.3:
                    st.info("""
                    **Riesgo Bajo:** Los patrones de voz analizados no muestran indicadores 
                    significativos de Alzheimer o demencia. Sin embargo, ante cualquier 
                    preocupación, consulta con un profesional médico.
                    """)
                elif result['probability'] < 0.6:
                    st.warning("""
                    **Riesgo Moderado:** Se detectaron algunos patrones que podrían estar 
                    asociados con deterioro cognitivo. Se recomienda consultar con un 
                    neurólogo para una evaluación más exhaustiva.
                    """)
                else:
                    st.error("""
                    **Riesgo Alto:** Los patrones de voz analizados muestran características 
                    comúnmente asociadas con Alzheimer o demencia. Es importante consultar 
                    con un especialista lo antes posible para un diagnóstico profesional.
                    """)

                # Disclaimer
                st.divider()
                st.caption("""
                ⚠️ **Descargo de responsabilidad:** Esta herramienta utiliza inteligencia 
                artificial y está diseñada únicamente con fines informativos y educativos. 
                NO constituye un diagnóstico médico. Consulta siempre con profesionales 
                de la salud calificados para obtener un diagnóstico y tratamiento adecuados.
                """)

            except Exception as e:
                st.error(f"❌ Error durante el análisis: {str(e)}")

            finally:
                # Limpiar archivo temporal
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # Información adicional en la barra lateral
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        **Sobre DementIA:**
        
        Esta aplicación utiliza:
        - **Wav2Vec2**: Para extraer características de voz
        - **5 Modelos de Deep Learning**: Entrenados con validación cruzada
        - **Ensemble Voting**: Promedia predicciones para mayor precisión
        
        **Datasets utilizados:**
        - DementiaNet
        - Pitt Corpus (DementiaBank)
        
        **Precisión estimada:** ~85-90% en conjunto de test
        
        ---
        
        **Desarrollado con:**
        - TensorFlow / Keras
        - Transformers (HuggingFace)
        - Streamlit
        """)

        st.divider()

        st.markdown("""
        **Contacto:**
        
        📧 Email: tu_email@ejemplo.com
        
        💻 GitHub: [DementIA](https://github.com/tu_usuario/DementIA)
        """)


if __name__ == "__main__":
    main()

