"""
Funciones auxiliares reutilizables para la aplicación.
Incluye validación, conversión, normalización y carga de modelos.
"""

import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
import tensorflow as tf
from tensorflow import keras


def validate_audio_format(file_path):
    """
    Valida que el archivo sea un formato de audio compatible.

    Args:
        file_path: Ruta del archivo a validar

    Returns:
        True si es válido, False en caso contrario
    """
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_extension = Path(file_path).suffix.lower()

    return file_extension in valid_extensions


def convert_to_wav_16khz(input_path, output_path=None):
    """
    Convierte cualquier audio a formato WAV 16 kHz mono.

    Args:
        input_path: Ruta del audio de entrada
        output_path: Ruta donde guardar el audio convertido (opcional)

    Returns:
        Ruta del archivo convertido
    """
    # Cargar audio en cualquier formato
    audio, sr = librosa.load(input_path, sr=16000, mono=True)

    # Si no se especifica output_path, crear uno temporal
    if output_path is None:
        base_dir = Path(input_path).parent
        output_path = base_dir / f"{Path(input_path).stem}_converted.wav"

    # Guardar como WAV 16 kHz
    sf.write(output_path, audio, 16000)

    return str(output_path)


def normalize_audio_rms(audio, target_rms=0.1):
    """
    Normaliza el volumen del audio usando RMS normalization.

    Args:
        audio: Array de audio
        target_rms: RMS objetivo (default: 0.1)

    Returns:
        Audio normalizado
    """
    # Calcular RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # Evitar división por cero
    if rms == 0:
        return audio

    # Normalizar
    audio = audio * (target_rms / rms)

    # Evitar clipping
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def apply_bandpass_filter(audio, sr=16000, lowcut=80, highcut=8000):
    """
    Aplica filtro pasa-banda 80-8000 Hz (calidad telefónica).

    Args:
        audio: Array de audio
        sr: Frecuencia de muestreo
        lowcut: Frecuencia de corte inferior
        highcut: Frecuencia de corte superior

    Returns:
        Audio filtrado
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)

    return filtered_audio


def preprocess_user_audio(audio_path):
    """
    Preprocesa el audio del usuario aplicando todos los pasos necesarios.

    Args:
        audio_path: Ruta del audio del usuario

    Returns:
        Audio preprocesado (numpy array) y sample rate
    """
    # 1. Cargar audio a 16 kHz mono
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # 2. Normalizar volumen
    audio = normalize_audio_rms(audio)

    # 3. Aplicar filtro pasa-banda
    audio = apply_bandpass_filter(audio, sr)

    return audio, sr


def load_ensemble_models(models_dir):
    """
    Carga los 5 modelos entrenados para ensemble voting.

    Args:
        models_dir: Ruta de la carpeta que contiene los modelos

    Returns:
        Lista con los 5 modelos cargados
    """
    models = []
    models_path = Path(models_dir)

    for i in range(1, 6):
        model_path = models_path / f"modelo_fold{i}.h5"

        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

        model = keras.models.load_model(model_path)
        models.append(model)

    return models


def ensemble_predict(models, embedding):
    """
    Realiza ensemble voting promediando las predicciones de los 5 modelos.

    Args:
        models: Lista de modelos cargados
        embedding: Embedding de 1024 dimensiones (numpy array)

    Returns:
        Probabilidad promedio de Alzheimer (0-1)
    """
    # Expandir dimensiones para batch
    embedding_batch = np.expand_dims(embedding, axis=0)

    # Obtener predicciones de cada modelo
    predictions = []
    for model in models:
        pred = model.predict(embedding_batch, verbose=0)
        predictions.append(pred[0][0])

    # Promediar predicciones
    ensemble_probability = np.mean(predictions)

    return ensemble_probability


def format_probability(probability):
    """
    Formatea la probabilidad como porcentaje legible.

    Args:
        probability: Probabilidad (0-1)

    Returns:
        String formateado (ej: "75.3%")
    """
    return f"{probability * 100:.1f}%"


def get_risk_level(probability):
    """
    Determina el nivel de riesgo según la probabilidad.

    Args:
        probability: Probabilidad de Alzheimer (0-1)

    Returns:
        Tupla (nivel, color) con el nivel de riesgo y color sugerido
    """
    if probability < 0.3:
        return ("Bajo", "green")
    elif probability < 0.6:
        return ("Moderado", "orange")
    else:
        return ("Alto", "red")

