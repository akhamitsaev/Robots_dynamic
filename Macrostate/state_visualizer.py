import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StateVisualizer:
    def __init__(self, robot_size=60.0):
        self.robot_size = robot_size
        sns.set_style("whitegrid")
        
    def plot_cluster_stats(self, cluster_stats, save_path='cluster_statistics.png'):
        n_clusters = len(cluster_stats)
        feature_names = cluster_stats[0]['feature_names']
        n_features = len(feature_names)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(feature_names):
            if i >= len(axes):
                break
            ax = axes[i]
            means = [cluster_stats[c]['means'][i] for c in range(n_clusters)]
            stds = [cluster_stats[c]['stds'][i] for c in range(n_clusters)]
            x_pos = np.arange(n_clusters)
            
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=plt.cm.tab20.colors[:n_clusters])
            ax.set_xlabel('Cluster')
            ax.set_ylabel(feature_name)
            ax.set_title(f'{feature_name}')
            ax.set_xticks(x_pos)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_feature_correlation(self, features_df, save_path='feature_correlation.png'):
        feature_cols = [
            'Polar_Order', 'Mean_Angle', 'Coordination_Num', 'Mean_Distance',
            'Angular_Vel', 'Rotation_Direction', 'Rot_Order', 'Center_Vel',
            'Velocity_Dispersion', 'Std_Nearest_Dist', 'Mean_Velocity'
        ]
        corr_matrix = features_df[feature_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_distribution(self, results_df, save_path='cluster_distribution.png'):
        cluster_counts = results_df['cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                      color=plt.cm.tab20.colors[:len(cluster_counts)])
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Samples')
        plt.title('Cluster Size Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        for i, (cluster_id, count) in enumerate(cluster_counts.items()):
            plt.text(cluster_id, count + max(cluster_counts.values)*0.01, 
                    f'{count}', ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_state_interpretation_table(self, cluster_stats):
        interpretation = []
        for cluster_id, stats in cluster_stats.items():
            means = stats['means']
            
            po = means[0]
            mean_angle = means[1]
            coord_num = means[2]
            mean_dist = means[3]
            angular_vel = means[4]
            rotation_dir = means[5]
            rot_order = means[6]
            center_vel = means[7]
            vel_disp = means[8]
            std_nearest = means[9]
            mean_vel = means[10]
            
            if po > 0.7 and mean_dist < self.robot_size * 2:
                state_type = "Compact Swarm (Рой)"
            elif abs(angular_vel) > 0.1 and rotation_dir > 0:
                state_type = "Clockwise Rotation (Правое вращение)"
            elif abs(angular_vel) > 0.1 and rotation_dir < 0:
                state_type = "Counter-Clockwise Rotation (Левое вращение)"
            elif coord_num > 4 and std_nearest < self.robot_size * 0.5:
                state_type = "Regular Structure (Мицелла)"
            elif po < 0.3 and vel_disp > 0.8:
                state_type = "Chaotic Motion (Хаос)"
            elif mean_dist > self.robot_size * 3:
                state_type = "Sparse Distribution (Разреженное)"
            elif center_vel > mean_vel * 0.7:
                state_type = "Collective Translation (Поступательное движение)"
            else:
                state_type = "Mixed/Transition State (Смешанное)"
            
            interpretation.append({
                'Cluster': cluster_id,
                'Samples': stats['count'],
                'State_Type': state_type,
                'PO': f"{po:.3f}",
                'Angular_Vel': f"{angular_vel:.3f}",
                'Coord_Num': f"{coord_num:.1f}",
                'Mean_Dist': f"{mean_dist:.1f}",
                'Vel_Disp': f"{vel_disp:.3f}"
            })
        
        return pd.DataFrame(interpretation)