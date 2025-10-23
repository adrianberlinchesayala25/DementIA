"""
Script para descargar los datasets DementiaNet y Pitt Corpus.
Descarga y organiza los audios en la carpeta data/og_audios/
"""

from pathlib import Path


def create_directories():
    """
    Crea las carpetas necesarias para almacenar los datasets.
    """
    base_dir = Path(__file__).parent.parent
    og_audios_dir = base_dir / "data" / "og_audios"

    # Crear subcarpetas para cada dataset
    (og_audios_dir / "dementianet").mkdir(parents=True, exist_ok=True)
    (og_audios_dir / "pitt_corpus").mkdir(parents=True, exist_ok=True)

    print("✅ Carpetas creadas correctamente")
    return og_audios_dir


def download_dementianet(output_dir):
    """
    Descarga el dataset DementiaNet.

    Args:
        output_dir: Ruta donde se guardará el dataset
    """
    print("\n📥 Descargando DementiaNet...")

    print("⚠️ IMPORTANTE: Debes obtener el dataset DementiaNet manualmente desde:")
    print("   https://dementia.talkbank.org/ (o la fuente oficial)")
    print("   Y colocarlo en: data/og_audios/dementianet/")
    print("\n   Este script es un placeholder. Descarga manual requerida.")


def download_pitt_corpus(output_dir):
    """
    Descarga el Pitt Corpus (DementiaBank).

    Args:
        output_dir: Ruta donde se guardará el dataset
    """
    print("\n📥 Descargando Pitt Corpus...")

    print("⚠️ IMPORTANTE: Debes obtener el Pitt Corpus manualmente desde:")
    print("   https://dementia.talkbank.org/access/English/Pitt.html")
    print("   Y colocarlo en: data/og_audios/pitt_corpus/")
    print("\n   Requiere registro en TalkBank para acceso.")


def verify_downloads(og_audios_dir):
    """
    Verifica que los datasets se hayan descargado correctamente.

    Args:
        og_audios_dir: Ruta de la carpeta de audios originales
    """
    print("\n🔍 Verificando descargas...")

    dementianet_files = list((og_audios_dir / "dementianet").glob("*.wav"))
    pitt_files = list((og_audios_dir / "pitt_corpus").glob("*.wav"))

    print(f"   DementiaNet: {len(dementianet_files)} archivos .wav")
    print(f"   Pitt Corpus: {len(pitt_files)} archivos .wav")

    if len(dementianet_files) > 0 and len(pitt_files) > 0:
        print("✅ Datasets verificados correctamente")
    else:
        print("⚠️ Algunos datasets parecen estar vacíos. Verifica las descargas.")


def main():
    """
    Función principal que orquesta la descarga de datasets.
    """
    print("=" * 50)
    print("DESCARGA DE DATASETS")
    print("=" * 50)

    # Crear estructura de carpetas
    og_audios_dir = create_directories()

    # Intentar descargar datasets
    download_dementianet(og_audios_dir / "dementianet")
    download_pitt_corpus(og_audios_dir / "pitt_corpus")

    # Verificar descargas
    verify_downloads(og_audios_dir)

    print("\n" + "=" * 50)
    print("INSTRUCCIONES:")
    print("=" * 50)
    print("1. Descarga manualmente los datasets desde las URLs indicadas")
    print("2. Coloca los archivos .wav en las carpetas correspondientes:")
    print("   - data/og_audios/dementianet/")
    print("   - data/og_audios/pitt_corpus/")
    print("3. Ejecuta nuevamente este script para verificar")
    print("=" * 50)


if __name__ == "__main__":
    main()

