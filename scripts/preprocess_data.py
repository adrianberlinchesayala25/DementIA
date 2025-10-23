"""
Script para preprocesar los audios originales.
- Resample a 16 kHz (obligatorio para Wav2Vec2)
- Normalización de volumen (RMS normalization)
- Filtro pasa-banda 80-8000 Hz (calidad telefónica)
"""

import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from tqdm import tqdm


def resample_audio(audio, original_sr, target_sr=16000):
    """
    Resamplea el audio a la frecuencia de muestreo objetivo.

    Args:
        audio: Array de audio
        original_sr: Frecuencia de muestreo original
        target_sr: Frecuencia de muestreo objetivo (16000 Hz para Wav2Vec2)

    Returns:
        Audio resampleado
    """
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio


def normalize_volume(audio):
    """
    Normaliza el volumen del audio usando RMS normalization.
    Esto asegura que todos los audios tengan un nivel de volumen similar.

    Args:
        audio: Array de audio

    Returns:
        Audio normalizado
    """
    # Calcular RMS (Root Mean Square)
    rms = np.sqrt(np.mean(audio ** 2))

    # Evitar división por cero
    if rms == 0:
        return audio

    # Normalizar a RMS objetivo de 0.1 (nivel conservador)
    target_rms = 0.1
    audio = audio * (target_rms / rms)

    # Asegurar que no haya clipping (valores fuera de [-1, 1])
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def bandpass_filter(audio, sr, lowcut=80, highcut=8000):
    """
    Aplica filtro pasa-banda para simular calidad telefónica.
    Elimina frecuencias extremas irrelevantes para la voz.

    Args:
        audio: Array de audio
        sr: Frecuencia de muestreo
        lowcut: Frecuencia de corte inferior (80 Hz)
        highcut: Frecuencia de corte superior (8000 Hz)

    Returns:
        Audio filtrado
    """
    # Diseñar filtro Butterworth de orden 5
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')

    # Aplicar filtro
    filtered_audio = signal.filtfilt(b, a, audio)

    return filtered_audio


def preprocess_audio_file(input_path, output_path, target_sr=16000):
    """
    Preprocesa un archivo de audio completo.

    Args:
        input_path: Ruta del archivo de audio original
        output_path: Ruta donde se guardará el audio preprocesado
        target_sr: Frecuencia de muestreo objetivo

    Returns:
        True si el procesamiento fue exitoso, False en caso contrario
    """
    try:
        # Cargar audio
        audio, sr = librosa.load(input_path, sr=None, mono=True)

        # 1. Resample a 16 kHz
        audio = resample_audio(audio, sr, target_sr)

        # 2. Normalizar volumen
        audio = normalize_volume(audio)

        # 3. Aplicar filtro pasa-banda
        audio = bandpass_filter(audio, target_sr)

        # Guardar audio preprocesado
        sf.write(output_path, audio, target_sr)

        return True

    except Exception as e:
        print(f"❌ Error procesando {input_path}: {e}")
        return False


def process_dataset(dataset_name):
    """
    Procesa todos los audios de un dataset específico.

    Args:
        dataset_name: Nombre del dataset ('dementianet' o 'pitt_corpus')
    """
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "og_audios" / dataset_name
    output_dir = base_dir / "data" / "processed_audios" / dataset_name

    # Crear carpeta de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtener lista de audios
    audio_files = list(input_dir.glob("*.wav"))

    if len(audio_files) == 0:
        print(f"⚠️ No se encontraron archivos .wav en {input_dir}")
        return

    print(f"\n📁 Procesando {dataset_name}...")
    print(f"   Total de archivos: {len(audio_files)}")

    # Procesar cada archivo con barra de progreso
    successful = 0
    for audio_file in tqdm(audio_files, desc=f"   Preprocesando"):
        output_path = output_dir / audio_file.name
        if preprocess_audio_file(audio_file, output_path):
            successful += 1

    print(f"✅ {successful}/{len(audio_files)} archivos procesados correctamente")


def main():
    """
    Función principal que orquesta el preprocesamiento de todos los datasets.
    """
    print("=" * 50)
    print("PREPROCESAMIENTO DE AUDIOS")
    print("=" * 50)
    print("Pasos:")
    print("  1. Resample a 16 kHz")
    print("  2. Normalización RMS")
    print("  3. Filtro pasa-banda 80-8000 Hz")
    print("=" * 50)

    # Procesar ambos datasets
    process_dataset("dementianet")
    process_dataset("pitt_corpus")

    print("\n✅ Preprocesamiento completado")
    print("Los audios procesados están en: data/processed_audios/")


if __name__ == "__main__":
    main()

