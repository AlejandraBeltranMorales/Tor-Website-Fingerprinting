import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scapy.all import rdpcap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NetworkTrafficClassifier:
    def __init__(self, train_folder: str, censored_folder: str, random_state: int = 42):
        """
        Initialize network traffic classifier with configurable parameters.
        
        Args:
            train_folder (str): Path to uncensored website training data
            censored_folder (str): Path to censored website data (Task 1)
            random_state (int): Seed for reproducibility
        """
        self.train_folder = os.path.expanduser(train_folder)
        self.censored_folder = os.path.expanduser(censored_folder)
        self.random_state = random_state
        
        # Predefined list of censored websites from task1
        self.censored_websites = {
            "appspot.com", "archive.org", "bitsnoop.com", "blueskyswimwear.com", 
            "docs.google.com", "drive.google.com", "en.wikipedia.org", "excite.co.jp", 
            "mp3lemon.org", "newalbumreleases.net", "nicovideo.jp", "pbworks.com", 
            "scribd.com", "sony.co.jp", "t.co", "thetibetpost.com", "torrentfreak.com", 
            "www.amnesty.org", "www.archive.org", "www.bahai.com", "www.cannabis.com", 
            "www.crazyshit.com", "www.falundafa.org", "www.filecrop.com", 
            "www.monova.org", "www.openvpn.net", "www.sbs.com.au", 
            "www.torrentsdownload.org", "www.watchfreemovies.ch", "www.xvideos.com"
        }
        
        # Model components
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

    def process_pcap_folder(self, folder: str, label: int) -> pd.DataFrame:
        """
        Process PCAP files from the folder and assign labels.
        """
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        pcap_files = [
            os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if f.endswith(".pcap")
        ]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.extract_advanced_features, pcap_files))

        valid_results = [
            {**result, 'filename': os.path.basename(file), 'label': label} 
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
        # Process uncensored and censored data
        uncensored_df = self.process_pcap_folder(self.train_folder, label=0)
        censored_df = self.process_pcap_folder(self.censored_folder, label=1)

        # Combine both datasets
        df = pd.concat([uncensored_df, censored_df], ignore_index=True)

        if df.empty:
            raise ValueError("No features were extracted from PCAP files. Please check input data.")

        X = df.drop(columns=['filename', 'label']).to_numpy()
        y = df['label'].to_numpy()

        # Create test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.variance_filter = VarianceThreshold(threshold=0.01)
        X_train_selected = self.variance_filter.fit_transform(X_train_scaled)
        X_test_selected = self.variance_filter.transform(X_test_scaled)
        
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=self.random_state
        )
        self.classifier.fit(X_train_selected, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, y_pred))

    def predict_test_data(self, test_folder: str, output_csv: str):
        """
        Predict labels for test data and save results in CSV format.
        """
        test_folder = os.path.expanduser(test_folder)
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder not found: {test_folder}")

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

        # Predict labels
        predictions = self.classifier.predict(X_test_selected)

        # Create a DataFrame for output
        result_df = pd.DataFrame({
            'filename': df_test['filename'],
            'prediction': predictions
        })

        # Ensure filenames are sorted numerically in the output
        result_df['filename'] = result_df['filename'].apply(
            lambda x: f"{int(x.split('.')[0])}.pcap"
        )
        result_df = result_df.sort_values(by='filename', key=lambda col: col.map(lambda x: int(x.split('.')[0])))

        result_df.to_csv(output_csv, index=False, sep=';', header=False)
        logger.info(f"Predictions saved to {output_csv}")

def main():
    classifier = NetworkTrafficClassifier("~/Unit4/task2/train/", "~/Unit4/task1/train/train_data")
    classifier.train_model()

    # Predict on test data
    classifier.predict_test_data("~/Unit4/task2/test/", "output.csv")

if __name__ == "__main__":
    main()
