"""
Script para extraer embeddings de Wav2Vec2.
Convierte los audios aumentados en vectores de 1024 dimensiones.
Genera: X_train.npy, y_train.npy, X_test.npy, y_test.npy
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
from sklearn.model_selection import train_test_split


def load_wav2vec2_model():
    """
    Carga el modelo Wav2Vec2 preentrenado desde HuggingFace.

    Returns:
        processor: Procesador de Wav2Vec2
        model: Modelo TensorFlow de Wav2Vec2
    """
    print("📥 Cargando Wav2Vec2...")

    # Cargar procesador y modelo preentrenado
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    print("✅ Wav2Vec2 cargado correctamente")
    return processor, model


def extract_embedding(audio_path, processor, model):
    """
    Extrae el embedding de 1024 dimensiones de un audio usando Wav2Vec2.

    Args:
        audio_path: Ruta del archivo de audio
        processor: Procesador de Wav2Vec2
        model: Modelo de Wav2Vec2

    Returns:
        Embedding de 1024 dimensiones (numpy array)
    """
    # Cargar audio (Wav2Vec2 espera 16 kHz)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Procesar audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="tf", padding=True)

    # Extraer embeddings (última capa oculta)
    outputs = model(inputs.input_values)
    embeddings = outputs.last_hidden_state

    # Promediar embeddings temporales para obtener un vector fijo de 1024
    embedding = np.mean(embeddings.numpy(), axis=1).squeeze()

    return embedding


def load_dataset_with_labels(dataset_name):
    """
    Carga todos los audios de un dataset y asigna etiquetas.

    Args:
        dataset_name: Nombre del dataset

    Returns:
        Lista de tuplas (audio_path, label)
    """
    base_dir = Path(__file__).parent.parent
    audio_dir = base_dir / "data" / "audios_data_augmentation" / dataset_name

    audio_files = list(audio_dir.glob("*.wav"))

    # Asignar etiquetas según el nombre del archivo
    # IMPORTANTE: Ajustar según la estructura real de los datasets
    # Ejemplo: si el archivo contiene "alzheimer" o "dementia" → label=1
    #          si contiene "control" → label=0

    data = []
    for audio_file in audio_files:
        # Lógica de etiquetado (ajustar según tus datasets)
        file_name_lower = audio_file.name.lower()

        if "alzheimer" in file_name_lower or "dementia" in file_name_lower or "ad" in file_name_lower:
            label = 1  # Alzheimer
        elif "control" in file_name_lower or "cn" in file_name_lower:
            label = 0  # Control
        else:
            # Si no se puede determinar, asignar según estructura del dataset
            # IMPORTANTE: Revisar manualmente la estructura de los datasets
            label = 0  # Valor por defecto (ajustar según necesidad)

        data.append((str(audio_file), label))

    return data


def extract_all_embeddings():
    """
    Extrae embeddings de todos los audios y crea train/test splits.
    """
    print("=" * 50)
    print("EXTRACCIÓN DE EMBEDDINGS WAV2VEC2")
    print("=" * 50)

    # Cargar modelo Wav2Vec2
    processor, model = load_wav2vec2_model()

    # Cargar datasets con etiquetas
    print("\n📁 Cargando datasets...")
    dementianet_data = load_dataset_with_labels("dementianet")
    pitt_data = load_dataset_with_labels("pitt_corpus")

    # Combinar ambos datasets
    all_data = dementianet_data + pitt_data
    print(f"   Total de audios: {len(all_data)}")

    # Agrupar audios originales (para evitar data leakage)
    # Los audios aumentados del mismo original deben ir al mismo conjunto
    grouped_data = {}
    for audio_path, label in all_data:
        # Extraer nombre base (sin _aug0, _aug1, etc.)
        base_name = Path(audio_path).stem.split('_aug')[0]

        if base_name not in grouped_data:
            grouped_data[base_name] = []

        grouped_data[base_name].append((audio_path, label))

    # Dividir grupos en train/test (80/20)
    group_keys = list(grouped_data.keys())
    train_groups, test_groups = train_test_split(
        group_keys,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Expandir grupos a listas de audios
    train_data = [item for group in train_groups for item in grouped_data[group]]
    test_data = [item for group in test_groups for item in grouped_data[group]]

    print(f"   Train: {len(train_data)} audios ({len(train_groups)} grupos)")
    print(f"   Test: {len(test_data)} audios ({len(test_groups)} grupos)")

    # Extraer embeddings de train
    print("\n🔄 Extrayendo embeddings de TRAIN...")
    X_train = []
    y_train = []

    for audio_path, label in tqdm(train_data, desc="   Train"):
        embedding = extract_embedding(audio_path, processor, model)
        X_train.append(embedding)
        y_train.append(label)

    # Extraer embeddings de test
    print("\n🔄 Extrayendo embeddings de TEST...")
    X_test = []
    y_test = []

    for audio_path, label in tqdm(test_data, desc="   Test"):
        embedding = extract_embedding(audio_path, processor, model)
        X_test.append(embedding)
        y_test.append(label)

    # Convertir a numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Guardar embeddings
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print("\n💾 Guardando embeddings...")
    np.save(embeddings_dir / "X_train.npy", X_train)
    np.save(embeddings_dir / "y_train.npy", y_train)
    np.save(embeddings_dir / "X_test.npy", X_test)
    np.save(embeddings_dir / "y_test.npy", y_test)

    print(f"✅ Embeddings guardados en: {embeddings_dir}")
    print(f"   X_train.shape: {X_train.shape}")
    print(f"   y_train.shape: {y_train.shape}")
    print(f"   X_test.shape: {X_test.shape}")
    print(f"   y_test.shape: {y_test.shape}")


def main():
    """
    Función principal.
    """
    extract_all_embeddings()


if __name__ == "__main__":
    main()

