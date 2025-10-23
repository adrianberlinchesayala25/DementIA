"""
Script para aplicar Data Augmentation a los audios preprocesados.
Técnicas:
- Gaussian Noise (ruido blanco)
- Pitch Shifting (cambio de tono)
- Time Stretching (cambio de velocidad)
- Time Shifting (desplazamiento temporal)

Genera ~6x más datos (de 410 → ~2500 audios)
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import audiomentations as A


def create_augmentation_pipeline():
    """
    Crea el pipeline de augmentation con audiomentations.

    Returns:
        Objeto Compose con todas las transformaciones
    """
    augment = A.Compose([
        # 1. Gaussian Noise: añade ruido blanco (SNR entre 15-25 dB)
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),

        # 2. Pitch Shifting: cambio de tono (-2 a +2 semitonos)
        A.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),

        # 3. Time Stretching: cambio de velocidad (0.9x a 1.1x)
        A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),

        # 4. Time Shifting: desplazamiento temporal (±10% de la duración)
        A.Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5),
    ])

    return augment


def augment_audio(audio, sr, augment_pipeline, n_variations=5):
    """
    Aplica data augmentation a un audio y genera múltiples variaciones.

    Args:
        audio: Array de audio
        sr: Frecuencia de muestreo
        augment_pipeline: Pipeline de transformaciones
        n_variations: Número de variaciones a generar

    Returns:
        Lista de audios aumentados (incluyendo el original)
    """
    augmented_audios = [audio]  # Incluir audio original

    for i in range(n_variations):
        # Aplicar transformaciones aleatorias
        augmented = augment_pipeline(samples=audio, sample_rate=sr)
        augmented_audios.append(augmented)

    return augmented_audios


def process_dataset_augmentation(dataset_name, n_variations=5):
    """
    Aplica data augmentation a todos los audios de un dataset.

    Args:
        dataset_name: Nombre del dataset
        n_variations: Número de variaciones por audio (default: 5)
    """
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "processed_audios" / dataset_name
    output_dir = base_dir / "data" / "audios_data_augmentation" / dataset_name

    # Crear carpeta de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtener lista de audios preprocesados
    audio_files = list(input_dir.glob("*.wav"))

    if len(audio_files) == 0:
        print(f"⚠️ No se encontraron archivos en {input_dir}")
        return

    print(f"\n📁 Aplicando augmentation a {dataset_name}...")
    print(f"   Total de archivos: {len(audio_files)}")
    print(f"   Variaciones por archivo: {n_variations + 1} (original + {n_variations})")
    print(f"   Total esperado: {len(audio_files) * (n_variations + 1)} archivos")

    # Crear pipeline de augmentation
    augment_pipeline = create_augmentation_pipeline()

    # Procesar cada archivo
    total_generated = 0
    for audio_file in tqdm(audio_files, desc="   Augmentando"):
        # Cargar audio preprocesado
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)

        # Aplicar augmentation
        augmented_audios = augment_audio(audio, sr, augment_pipeline, n_variations)

        # Guardar todas las variaciones
        for i, aug_audio in enumerate(augmented_audios):
            # Nombre: original.wav → original_aug0.wav, original_aug1.wav, etc.
            base_name = audio_file.stem
            output_name = f"{base_name}_aug{i}.wav"
            output_path = output_dir / output_name

            sf.write(output_path, aug_audio, sr)
            total_generated += 1

    print(f"✅ {total_generated} archivos generados")


def main():
    """
    Función principal que aplica data augmentation a todos los datasets.
    """
    print("=" * 50)
    print("DATA AUGMENTATION")
    print("=" * 50)
    print("Técnicas aplicadas:")
    print("  1. Gaussian Noise (ruido blanco)")
    print("  2. Pitch Shifting (cambio de tono)")
    print("  3. Time Stretching (cambio de velocidad)")
    print("  4. Time Shifting (desplazamiento temporal)")
    print("=" * 50)

    # Aplicar augmentation a ambos datasets (5 variaciones por audio)
    process_dataset_augmentation("dementianet", n_variations=5)
    process_dataset_augmentation("pitt_corpus", n_variations=5)

    print("\n✅ Data augmentation completado")
    print("Los audios aumentados están en: data/audios_data_augmentation/")


if __name__ == "__main__":
    main()

