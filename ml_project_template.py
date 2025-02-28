"""
Makine Öğrenmesi Proje Şablonu
------------------------------
Bu şablon, herhangi bir makine öğrenmesi projesi için temel yapıyı sağlar.
Veri analizi, önişleme, model eğitimi ve değerlendirme adımlarını içerir.
"""

import warnings
import logging
import yaml
import joblib
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
from datetime import datetime
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_curve, auc

from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline

# Uyarıları kapatma
warnings.filterwarnings('ignore')

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLProject:
    """
    Makine öğrenmesi projelerini yönetmek için ana sınıf.
    """
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Args:
            config_path: Yapılandırma dosyasının yolu
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = None
        
        # Çıktı dizini oluşturma
        self.output_dir = Path('outputs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        YAML yapılandırma dosyasını yükler.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Yapılandırma dosyası yüklenirken hata: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Veriyi yapılandırma dosyasında belirtilen kaynaktan yükler.
        """
        try:
            data_path = self.config['data_path']
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError("Desteklenmeyen dosya formatı")
            
            logger.info(f"Veri başarıyla yüklendi. Boyut: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Veri yüklenirken hata: {e}")
            raise

    def perform_eda(self, df: pd.DataFrame) -> None:
        """
        Kapsamlı bir keşifsel veri analizi gerçekleştirir.
        """
        eda_dir = self.output_dir / 'eda'
        eda_dir.mkdir(exist_ok=True)
        
        # Temel istatistikler
        logger.info("Temel istatistikler hesaplanıyor...")
        stats_file = eda_dir / 'basic_statistics.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Veri Özeti:\n")
            f.write(df.describe().to_string())
            f.write("\n\nEksik Değerler:\n")
            f.write(df.isnull().sum().to_string())
            
        # Korelasyon analizi
        self._plot_correlations(df, eda_dir)
        
        # Dağılım grafikleri
        self._plot_distributions(df, eda_dir)
        
        logger.info(f"EDA sonuçları {eda_dir} dizinine kaydedildi")

    def _plot_correlations(self, df: pd.DataFrame, save_dir: Path) -> None:
        """
        Sayısal ve kategorik değişkenler için korelasyon analizleri yapar.
        """
        # Sayısal değişkenler için korelasyon matrisi
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
            plt.title("Sayısal Değişkenler Korelasyon Matrisi")
            plt.tight_layout()
            plt.savefig(save_dir / 'numeric_correlations.png')
            plt.close()

    def _plot_distributions(self, df: pd.DataFrame, save_dir: Path) -> None:
        """
        Tüm sayısal değişkenler için dağılım grafikleri çizer.
        """
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f"{col} Dağılımı")
                # Dosya adındaki özel karakterleri temizle
                safe_col_name = "".join(c if c.isalnum() else "_" for c in str(col))
                plt.savefig(save_dir / f'dist_{safe_col_name}.png')
                plt.close()
            except Exception as e:
                logger.warning(f"{col} sütunu için dağılım grafiği çizilemedi: {e}")

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Veri önişleme adımlarını gerçekleştirir.
        """
        preprocessors = {}
        df_processed = df.copy()

        # Eksik değer işleme
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df[col].mode()[0], inplace=True)

        # Kategorik değişken kodlama
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            preprocessors[f'le_{col}'] = le

        # Ölçeklendirme
        num_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
        preprocessors['scaler'] = scaler

        logger.info("Veri önişleme tamamlandı")
        return df_processed, preprocessors

    def prepare_model_data(self, df: pd.DataFrame) -> Tuple:
        """
        Model için veriyi hazırlar.
        """
        target_col = self.config['target_column']
        feature_cols = [col for col in df.columns if col != target_col]

        X = df[feature_cols]
        y = df[target_col]

        # Veri bölme
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=self.config['random_state']
        )

        logger.info("Veri eğitim, doğrulama ve test setlerine bölündü")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Yapılandırma dosyasında belirtilen modelleri eğitir.
        """
        for model_name, model_config in self.config['models'].items():
            logger.info(f"{model_name} modeli eğitiliyor...")
            
            # Model ve hiperparametre ızgarası oluşturma
            model_class = self._get_model_class(model_config['class'])
            model = model_class(**model_config.get('fixed_params', {}))
            param_grid = model_config.get('param_grid', {})

            # Grid Search ile model eğitimi
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring=self.config['scoring'],
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            self.models[model_name] = grid_search.best_estimator_
            self.results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }

            logger.info(f"{model_name} için en iyi parametreler: {grid_search.best_params_}")

    def evaluate_models(self, X_val: pd.DataFrame, y_val: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Eğitilen modelleri değerlendirir.
        """
        eval_dir = self.output_dir / 'model_evaluation'
        eval_dir.mkdir(exist_ok=True)

        best_score = float('-inf')
        for model_name, model in self.models.items():
            logger.info(f"{model_name} değerlendiriliyor...")

            # Tahminler
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # Metrikler
            metrics = self._calculate_metrics(y_val, y_val_pred, y_test, y_test_pred)
            self.results[model_name].update(metrics)

            # En iyi modeli seçme
            if metrics['val_score'] > best_score:
                best_score = metrics['val_score']
                self.best_model = model
                self.best_model_name = model_name

            # Görselleştirmeler
            self._plot_model_evaluation(model_name, y_test, y_test_pred, eval_dir)

        # Tüm modellerin ROC eğrilerini karşılaştır
        self.plot_all_roc_curves(X_test, y_test, eval_dir)

        # Sonuçları kaydetme
        self._save_results()

        # En iyi modeli raporla
        logger.info(f"En iyi model: {self.best_model_name} (ROC-AUC: {best_score:.3f})")

    def _calculate_metrics(self, y_val: pd.Series, y_val_pred: np.ndarray,
                         y_test: pd.Series, y_test_pred: np.ndarray) -> Dict:
        """
        Model değerlendirme metriklerini hesaplar.
        """
        is_classification = self.config['problem_type'] == 'classification'
        
        metrics = {}
        if is_classification:
            metrics.update({
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'classification_report': classification_report(y_test, y_test_pred)
            })
        else:
            metrics.update({
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'val_r2': r2_score(y_val, y_val_pred),
                'test_r2': r2_score(y_test, y_test_pred)
            })
        
        return metrics

    def _plot_model_evaluation(self, model_name: str, y_true: pd.Series,
                             y_pred: np.ndarray, save_dir: Path) -> None:
        """
        Model değerlendirme grafiklerini çizer.
        """
        is_classification = self.config['problem_type'] == 'classification'
        
        if is_classification:
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{model_name} - Confusion Matrix")
            plt.savefig(save_dir / f'{model_name}_confusion_matrix.png')
            plt.close()

            # ROC Eğrisi
            y_pred_proba = self.models[model_name].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Eğrisi')
            plt.legend(loc="lower right")
            plt.savefig(save_dir / f'{model_name}_roc_curve.png')
            plt.close()
        else:
            # Gerçek vs Tahmin
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Gerçek Değerler')
            plt.ylabel('Tahminler')
            plt.title(f"{model_name} - Gerçek vs Tahmin")
            plt.savefig(save_dir / f'{model_name}_predictions.png')
            plt.close()

    def plot_all_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series, save_dir: Path) -> None:
        """
        Tüm modellerin ROC eğrilerini tek bir grafikte çizer.
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Tüm Modeller - ROC Eğrileri Karşılaştırması')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_dir / 'all_models_roc_comparison.png')
        plt.close()

        # ROC-AUC skorlarını kaydet
        roc_auc_scores = {}
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            _, _, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[model_name] = roc_auc
        
        # Skorları JSON olarak kaydet
        with open(save_dir / 'roc_auc_scores.json', 'w') as f:
            json.dump(roc_auc_scores, f, indent=4)

    def save_model(self) -> None:
        """
        En iyi modeli ve önişleme nesnelerini kaydeder.
        """
        model_dir = self.output_dir / 'model'
        model_dir.mkdir(exist_ok=True)

        # Model kaydetme
        joblib.dump(self.best_model, model_dir / 'best_model.joblib')
        
        # Sonuçları JSON olarak kaydetme
        with open(model_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=4)

        logger.info(f"Model ve sonuçlar {model_dir} dizinine kaydedildi")

    @staticmethod
    def _get_model_class(class_name: str) -> BaseEstimator:
        """
        Model sınıfını dinamik olarak yükler.
        """
        try:
            module_name, class_name = class_name.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            logger.error(f"Model sınıfı yüklenirken hata: {e}")
            raise

def main():
    """
    Ana çalıştırma fonksiyonu
    """
    try:
        # Proje başlatma
        project = MLProject()
        logger.info("Proje başlatıldı")

        # Veri yükleme ve analiz
        df = project.load_data()
        project.perform_eda(df)

        # Veri önişleme
        df_processed, preprocessors = project.preprocess_data(df)

        # Model verisi hazırlama
        X_train, X_val, X_test, y_train, y_val, y_test = project.prepare_model_data(df_processed)

        # Model eğitimi
        project.train_models(X_train, y_train)

        # Model değerlendirme
        project.evaluate_models(X_val, y_val, X_test, y_test)

        # Model kaydetme
        project.save_model()

        logger.info("Proje başarıyla tamamlandı")

    except Exception as e:
        logger.error(f"Projede hata oluştu: {e}")
        raise

if __name__ == '__main__':
    main() 