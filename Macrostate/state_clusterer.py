import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import pandas as pd

class StateClusterer:
    def __init__(self, n_clusters=5, random_state=42, umap_jobs=1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.umap_jobs = umap_jobs
        self.scaler = StandardScaler()
        self.kmeans = None
        self.umap_reducer = None
        
    def prepare_features(self, features_dict):
        all_features = []
        file_indices = []
        time_indices = []
        for file_name, features in features_dict.items():
            all_features.append(features)
            file_indices.extend([file_name] * len(features))
            time_indices.extend(range(len(features)))
        features_array = np.vstack(all_features)
        self.file_indices = np.array(file_indices)
        self.time_indices = np.array(time_indices)
        return features_array
    
    def fit_cluster(self, features_array):
        X_scaled = self.scaler.fit_transform(features_array)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.umap_reducer = umap.UMAP(
            n_components=2, 
            random_state=self.random_state,
            n_jobs=self.umap_jobs
        )
        umap_embedding = self.umap_reducer.fit_transform(X_scaled)
        return cluster_labels, umap_embedding, X_scaled
    
    def analyze_clusters(self, features_array, cluster_labels):
        cluster_stats = {}
        feature_names = [
            'Polar_Order', 'Mean_Angle', 'Coordination_Num', 'Mean_Distance',
            'Angular_Vel', 'Rotation_Direction', 'Rot_Order', 'Center_Vel',
            'Velocity_Dispersion', 'Std_Nearest_Dist', 'Mean_Velocity'
        ]
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) == 0:
                continue
            cluster_features = features_array[mask]
            means = cluster_features.mean(axis=0)
            stds = cluster_features.std(axis=0)
            cluster_stats[cluster_id] = {
                'count': int(np.sum(mask)),
                'means': means,
                'stds': stds,
                'feature_names': feature_names
            }
        return cluster_stats
    
    def visualize_clusters(self, umap_embedding, cluster_labels, save_path='clusters_umap.png'):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                            c=cluster_labels, cmap='tab20', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title(f'K-means Clusters (n={self.n_clusters}) in UMAP space')
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, features_array, cluster_labels, umap_embedding, output_path='cluster_results.csv'):
        results_df = pd.DataFrame(features_array, 
            columns=[
                'Polar_Order', 'Mean_Angle', 'Coordination_Num', 'Mean_Distance',
                'Angular_Vel', 'Rotation_Direction', 'Rot_Order', 'Center_Vel',
                'Velocity_Dispersion', 'Std_Nearest_Dist', 'Mean_Velocity'
            ])
        results_df['cluster'] = cluster_labels
        results_df['umap1'] = umap_embedding[:, 0]
        results_df['umap2'] = umap_embedding[:, 1]
        results_df['file_name'] = self.file_indices
        results_df['time_step'] = self.time_indices
        results_df.to_csv(output_path, index=False)
        return results_df