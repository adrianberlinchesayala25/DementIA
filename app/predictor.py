"""
Módulo para realizar predicciones usando ensemble voting con los 5 modelos entrenados.
Combina las predicciones de los 5 modelos para obtener una predicción más robusta.
"""

from pathlib import Path
from app.utils import load_ensemble_models, ensemble_predict, format_probability, get_risk_level


class AlzheimerPredictor:
    """
    Clase para predecir la probabilidad de Alzheimer usando ensemble de 5 modelos.
    """

    def __init__(self, models_dir):
        """
        Inicializa el predictor cargando los 5 modelos entrenados.

        Args:
            models_dir: Ruta de la carpeta que contiene los modelos (.h5)
        """
        print("📥 Cargando modelos entrenados...")
        self.models = load_ensemble_models(models_dir)
        print(f"✅ {len(self.models)} modelos cargados correctamente")

    def predict(self, embedding):
        """
        Realiza predicción usando ensemble voting (promedio de los 5 modelos).

        Args:
            embedding: Vector de 1024 dimensiones (embedding de Wav2Vec2)

        Returns:
            Diccionario con:
            - probability: probabilidad de Alzheimer (0-1)
            - percentage: probabilidad formateada como porcentaje
            - risk_level: nivel de riesgo ("Bajo", "Moderado", "Alto")
            - risk_color: color sugerido para mostrar ("green", "orange", "red")
        """
        # Obtener predicción promedio de los 5 modelos
        probability = ensemble_predict(self.models, embedding)

        # Formatear resultados
        percentage = format_probability(probability)
        risk_level, risk_color = get_risk_level(probability)

        return {
            'probability': float(probability),
            'percentage': percentage,
            'risk_level': risk_level,
            'risk_color': risk_color
        }

    def predict_from_embedding(self, embedding):
        """
        Alias de predict() para mayor claridad.

        Args:
            embedding: Vector de 1024 dimensiones

        Returns:
            Diccionario con resultados de la predicción
        """
        return self.predict(embedding)

