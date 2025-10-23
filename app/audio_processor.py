"""
Módulo para procesar el audio del usuario y extraer embeddings Wav2Vec2.
Convierte el audio subido en un vector de 1024 dimensiones listo para predicción.
"""

import numpy as np
import librosa
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
from app.utils import preprocess_user_audio


class AudioProcessor:
    """
    Clase para procesar audios y extraer embeddings con Wav2Vec2.
    """

    def __init__(self):
        """
        Inicializa el procesador cargando el modelo Wav2Vec2 preentrenado.
        """
        print("📥 Cargando modelo Wav2Vec2...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        print("✅ Wav2Vec2 cargado correctamente")

    def extract_embedding(self, audio_path):
        """
        Extrae el embedding de 1024 dimensiones de un archivo de audio.

        Pasos:
        1. Preprocesar el audio (resample, normalizar, filtrar)
        2. Pasar por Wav2Vec2 para obtener embeddings temporales
        3. Promediar embeddings para obtener un vector fijo de 1024 dimensiones

        Args:
            audio_path: Ruta del archivo de audio del usuario

        Returns:
            Embedding de 1024 dimensiones (numpy array)
        """
        # 1. Preprocesar audio (normalización, filtro, etc.)
        audio, sr = preprocess_user_audio(audio_path)

        # 2. Procesar con Wav2Vec2
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="tf", padding=True)

        # 3. Extraer embeddings de la última capa oculta
        outputs = self.model(inputs.input_values)
        embeddings = outputs.last_hidden_state

        # 4. Promediar embeddings temporales para obtener vector fijo
        embedding = np.mean(embeddings.numpy(), axis=1).squeeze()

        return embedding

    def process_audio_file(self, audio_path):
        """
        Procesa un archivo de audio completo y devuelve su embedding.

        Args:
            audio_path: Ruta del archivo de audio

        Returns:
            Embedding de 1024 dimensiones
        """
        try:
            embedding = self.extract_embedding(audio_path)
            return embedding

        except Exception as e:
            raise Exception(f"Error al procesar el audio: {str(e)}")

