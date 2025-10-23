# 🧠 DementIA - Detección de Alzheimer mediante Análisis de Voz

Aplicación de inteligencia artificial que analiza patrones de voz para estimar la probabilidad de Alzheimer o demencia.

## 📋 Descripción

DementIA utiliza **Deep Learning** y **Wav2Vec2** para extraer características de audio y clasificar patrones de habla asociados con deterioro cognitivo. El sistema emplea **5 modelos entrenados con validación cruzada** y **ensemble voting** para obtener predicciones robustas.

### ✨ Características

- 🎤 Análisis de voz con Wav2Vec2 (embeddings de 1024 dimensiones)
- 🤖 Ensemble de 5 modelos entrenados con 5-Fold Cross Validation
- 📊 Precisión estimada: ~85-90%
- 🌐 Interfaz web intuitiva con Streamlit
- 🔒 Procesamiento local (privacidad garantizada)

---

## 🏗️ Arquitectura del Proyecto

```
DementIA/
├── data/
│   ├── og_audios/              # Audios originales (DementiaNet + Pitt Corpus)
│   ├── processed_audios/       # Audios preprocesados (16kHz, normalizados)
│   ├── audios_data_augmentation/ # Audios aumentados (~2500 muestras)
│   └── embeddings/             # Embeddings Wav2Vec2 (X_train, y_train, etc.)
├── models/                     # 5 modelos entrenados (.h5)
├── scripts/                    # Scripts de preprocesamiento
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   ├── data_augmentation.py
│   └── extract_embeddings.py
├── colab/
│   └── training_colab.ipynb   # Notebook para entrenar en Google Colab
├── app/                        # Aplicación web
│   ├── app.py                  # Interfaz principal (Streamlit)
│   ├── audio_processor.py      # Procesamiento de audio + Wav2Vec2
│   ├── predictor.py            # Ensemble voting
│   ├── utils.py                # Funciones auxiliares
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/DementIA.git
cd DementIA
```

### 2. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate  # En Windows
source venv/bin/activate  # En Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install -r app/requirements.txt
```

---

## 📊 Pipeline de Datos

### Paso 1: Descargar Datasets

```bash
python scripts/download_datasets.py
```

Descarga **DementiaNet** y **Pitt Corpus** y colócalos en `data/og_audios/`.

### Paso 2: Preprocesar Audios

```bash
python scripts/preprocess_data.py
```

Aplica:
- Resample a 16 kHz
- Normalización RMS
- Filtro pasa-banda 80-8000 Hz

### Paso 3: Data Augmentation

```bash
python scripts/data_augmentation.py
```

Genera ~6x más datos aplicando:
- Gaussian Noise
- Pitch Shifting
- Time Stretching
- Time Shifting

### Paso 4: Extraer Embeddings

```bash
python scripts/extract_embeddings.py
```

Extrae embeddings de **1024 dimensiones** con Wav2Vec2 y guarda:
- `X_train.npy`, `y_train.npy`
- `X_test.npy`, `y_test.npy`

---

## 🎓 Entrenamiento en Google Colab

### 1. Subir el proyecto a GitHub

```bash
git add .
git commit -m "Proyecto completo"
git push origin main
```

### 2. Abrir Colab y clonar el repositorio

Abre `colab/training_colab.ipynb` en Google Colab:

```python
!git clone https://github.com/tu_usuario/DementIA.git
%cd DementIA
```

### 3. Configurar GPU

En Colab: **Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)**

### 4. Ejecutar todas las celdas

El notebook entrena automáticamente los 5 modelos con:
- Arquitectura: 1024 → 64 → 32 → 1
- BatchNormalization + LeakyReLU + L2(0.01)
- EarlyStopping (patience=10)
- 5-Fold Cross Validation

### 5. Descargar modelos entrenados

Al final del notebook se genera `models.zip`. Descárgalo y extrae los archivos `.h5` en la carpeta `models/`.

### 6. Subir modelos a GitHub

```bash
git add models/
git commit -m "Modelos entrenados"
git push origin main
```

---

## 🖥️ Ejecutar la Aplicación

### Iniciar la app con Streamlit

```bash
streamlit run app/app.py
```

La aplicación se abrirá en `http://localhost:8501`

### Uso

1. **Subir un archivo de audio** (WAV, MP3, etc.)
2. **Hacer clic en "Analizar Audio"**
3. **Ver resultados:**
   - Probabilidad de Alzheimer (%)
   - Nivel de riesgo (Bajo/Moderado/Alto)
   - Interpretación detallada

---

## 🧪 Tecnologías Utilizadas

| Componente | Tecnología |
|------------|------------|
| **Framework DL** | TensorFlow / Keras |
| **Embeddings** | Wav2Vec2 (HuggingFace) |
| **Procesamiento Audio** | Librosa, Soundfile, Audiomentations |
| **Interfaz** | Streamlit |
| **Entrenamiento** | Google Colab (GPU NVIDIA) |
| **Datasets** | DementiaNet, Pitt Corpus |

---

## 📈 Arquitectura del Modelo

```
Audio (WAV) 
    ↓
Preprocesamiento (16kHz, normalización, filtro)
    ↓
Wav2Vec2 (congelado)
    ↓
Embeddings (1024 dimensiones)
    ↓
Dense(64) + BatchNorm + LeakyReLU + L2(0.01)
    ↓
Dense(32) + BatchNorm + LeakyReLU + L2(0.01)
    ↓
Dense(1) + Sigmoid
    ↓
Probabilidad de Alzheimer (0-1)
```

**Ensemble Voting:** Promedio de 5 modelos entrenados con cross-validation.

---

## ⚠️ Descargo de Responsabilidad

Esta herramienta está diseñada **únicamente con fines educativos e informativos**. 

**NO constituye un diagnóstico médico**. Consulta siempre con profesionales de la salud calificados para obtener un diagnóstico y tratamiento adecuados.

---

## 📝 Licencia

MIT License - Ver archivo `LICENSE`

---

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@tu_usuario](https://github.com/tu_usuario)
- Email: tu_email@ejemplo.com

---

## 🙏 Agradecimientos

- **DementiaBank** (Pitt Corpus)
- **DementiaNet Dataset**
- **HuggingFace** (Wav2Vec2)
- **Google Colab** (GPU gratuita)

