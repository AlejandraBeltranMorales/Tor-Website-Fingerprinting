import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from scapy.all import rdpcap, Packet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NetworkTrafficClassifier:
    def __init__(self, train_folder: str, random_state: int = 42):
        """
        Initialize network traffic classifier with configurable parameters.
        
        Args:
            train_folder (str): Path to training PCAP files
            random_state (int): Seed for reproducibility
        """
        self.train_folder = os.path.expanduser(train_folder)
        self.random_state = random_state
        
        # Model components
        self.label_encoder = None
        self.scaler = None
        self.classifier = None
        self.variance_filter = None

    @staticmethod
    def extract_advanced_features(pcap_file: str) -> Optional[Dict[str, float]]:
        """
        Extract comprehensive features from a PCAP file.
        """
        try:
            packets = rdpcap(pcap_file)
            
            if not packets:
                logger.warning(f"No packets in {pcap_file}")
                return None

            packet_sizes = [int(len(pkt)) for pkt in packets]
            timestamps = [float(pkt.time) for pkt in packets]

            def packet_stat(has_attr):
                filtered = [int(len(pkt)) for pkt in packets if hasattr(pkt, has_attr)]
                return {
                    f'{has_attr}_count': len(filtered),
                    f'{has_attr}_total_size': sum(filtered),
                    f'{has_attr}_avg_size': np.mean(filtered) if filtered else 0.0
                }

            src_stats = packet_stat('src')
            dst_stats = packet_stat('dst')

            if len(timestamps) > 1:
                inter_arrival_times = np.diff(timestamps)
                timing_features = {
                    'avg_inter_arrival': float(np.mean(inter_arrival_times)),
                    'std_inter_arrival': float(np.std(inter_arrival_times)),
                    'median_inter_arrival': float(np.median(inter_arrival_times))
                }
            else:
                timing_features = {
                    'avg_inter_arrival': 0.0,
                    'std_inter_arrival': 0.0,
                    'median_inter_arrival': 0.0
                }

            features = {
                **{
                    'total_packets': len(packets),
                    'avg_packet_size': float(np.mean(packet_sizes)),
                    'std_packet_size': float(np.std(packet_sizes)),
                    'in_out_ratio': float(dst_stats.get('dst_total_size', 0)) / 
                                    (float(src_stats.get('src_total_size', 0)) + 1e-10)
                },
                **src_stats,
                **dst_stats,
                **timing_features
            }

            return {k: float(v) if isinstance(v, (int, np.integer, np.float64)) else v for k, v in features.items()}

        except Exception as e:
            logger.error(f"Feature extraction error in {pcap_file}: {e}")
            return None

    def process_pcap_folder(self) -> pd.DataFrame:
        """
        Process all PCAP files in the training folder in parallel.
        """
        if not os.path.exists(self.train_folder):
            raise FileNotFoundError(f"Training folder not found: {self.train_folder}")

        pcap_files = [
            os.path.join(self.train_folder, f) 
            for f in os.listdir(self.train_folder) 
            if f.endswith(".pcap")
        ]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.extract_advanced_features, pcap_files))

        valid_results = [
            {**result, 'filename': os.path.basename(file)} 
            for result, file in zip(results, pcap_files) 
            if result is not None
        ]

        if not valid_results:
            logger.error("No valid PCAP files were processed.")
            return pd.DataFrame()

        return pd.DataFrame(valid_results)

    def train_model(self, test_size: float = 0.2):
        """
        Train the classification model with advanced preprocessing.
        """
        df = self.process_pcap_folder()

        if df.empty:
            raise ValueError("No features were extracted from PCAP files. Please check input data.")

        df['label'] = df['filename'].apply(lambda x: x.split('_')[0])

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(df['label'])
        
        X = df.drop(columns=['label', 'filename']).to_numpy()
        y = y_encoded
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        self.variance_filter = VarianceThreshold(threshold=0.01)
        X_train_selected = self.variance_filter.fit_transform(X_train_resampled)
        X_test_selected = self.variance_filter.transform(X_test_scaled)
        
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.classifier.fit(X_train_selected, y_train_resampled)
        
        y_pred = self.classifier.predict(X_test_selected)
        
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict_test_data(self, test_folder: str, output_csv: str):
        """
        Predict labels for unlabeled test data and save results in CSV format.

        Args:
            test_folder (str): Path to test PCAP files
            output_csv (str): Path to save the predictions
        """
        
        label_to_int_mapping = {
            "appspot.com": 0,
            "archive.org": 1,
            "bitsnoop.com": 2,
            "blueskyswimwear.com": 3,
            "docs.google.com": 4,
            "drive.google.com": 5,
            "en.wikipedia.org": 6,
            "excite.co.jp": 7,
            "mp3lemon.org": 8,
            "newalbumreleases.net": 9,
            "nicovideo.jp": 10,
            "pbworks.com": 11,
            "scribd.com": 12,
            "sony.co.jp": 13,
            "t.co": 14,
            "thetibetpost.com": 15,
            "torrentfreak.com": 16,
            "www.amnesty.org": 17,
            "www.archive.org": 18,
            "www.bahai.com": 19,
            "www.cannabis.com": 20,
            "www.crazyshit.com": 21,
            "www.falundafa.org": 22,
            "www.filecrop.com": 23,
            "www.monova.org": 24,
            "www.openvpn.net": 25,
            "www.sbs.com.au": 26,
            "www.torrentsdownload.org": 27,
            "www.watchfreemovies.ch": 28,
            "www.xvideos.com": 29
        }

        test_folder = os.path.expanduser(test_folder)
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder not found: {test_folder}")

        # List and sort the PCAP files numerically
        pcap_files = [
            os.path.join(test_folder, f) 
            for f in os.listdir(test_folder) 
            if f.endswith(".pcap")
        ]
        pcap_files = sorted(pcap_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.extract_advanced_features, pcap_files))

        valid_results = [
            {**result, 'filename': os.path.basename(file)} 
            for result, file in zip(results, pcap_files) 
            if result is not None
        ]

        if not valid_results:
            raise ValueError("No valid features extracted from test PCAP files.")

        df_test = pd.DataFrame(valid_results)
        X_test = df_test.drop(columns=['filename']).to_numpy()

        # Scale and select features as per training
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.variance_filter.transform(X_test_scaled)

        # Predict and map labels
        predictions = self.classifier.predict(X_test_selected)
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        # Transform predictions using the mapping
        transformed_predictions = [
            label_to_int_mapping.get(label, -1)  # Use -1 for unknown labels
            for label in predicted_labels
        ]

        # Create a DataFrame for output
        result_df = pd.DataFrame({
            'filename': df_test['filename'],
            'prediction': transformed_predictions
        })

        # Ensure filenames are sorted numerically in the output
        result_df['filename'] = result_df['filename'].apply(
            lambda x: f"{int(x.split('.')[0])}.pcap"
        )
        result_df = result_df.sort_values(by='filename', key=lambda col: col.map(lambda x: int(x.split('.')[0])))

        
        result_df.to_csv(output_csv, index=False, sep=';', header=False)
        logger.info(f"Predictions saved to {output_csv}")



def main():
    classifier = NetworkTrafficClassifier("~/Unit4/task1/train/train_data")
    classifier.train_model()

    # Predict on test data
    classifier.predict_test_data("~/Unit4/task1/test/test_data", "output.csv")

if __name__ == "__main__":
    main()
