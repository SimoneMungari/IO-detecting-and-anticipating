import pyfglt.fglt as fg
import sys
from raphtory import Graph, algorithms
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from scipy import stats

from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score, \
    recall_score, precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import List, Tuple, Dict, Set
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
class TemporalNetwork:

    def __init__(self):
        self.interactions = []  # List of (source, target, timestamp, interaction_type)
        self.nodes = set()
        self.interactions_dict = {}

    def add_interaction(self, source: str, target: str, timestamp: float,
                       interaction_type: str = 'retweet'):

        self.interactions.append((source, target, timestamp, interaction_type))
        self.nodes.add(source)
        self.nodes.add(target)

    def get_temporal_edges(self, time_window: Tuple[float, float] = None):
        if time_window is None:
            return [(s, t, ts, itype) for s, t, ts, itype in self.interactions]

        start, end = time_window
        return [(s, t, ts, itype) for s, t, ts, itype in self.interactions
                if start <= ts < end]

    def get_user_interactions(self, user: str):
        return self.interactions_dict[user]

    def get_snapshot(self, time_window: Tuple[float, float]):
        G = nx.DiGraph()
        edges = self.get_temporal_edges(time_window)
        for source, target, ts, itype in edges:
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
                G[source][target]['timestamps'].append(ts)
            else:
                G.add_edge(source, target, weight=1, timestamps=[ts],
                          interaction_type=itype)
        return G

def compute_graphlets_pyfglt(G: nx.DiGraph):

    F = fg.compute(G)
    return F

def extract_user_features(network: TemporalNetwork, user: str):

    features = {}

    # Get user's interactions
    user_interactions = network.get_user_interactions(user)

    if len(user_interactions) == 0:
        return None

    # Basic activity features
    features['total_interactions'] = len(user_interactions)

    if ablation == "twitter_interaction_types":
        interaction_types = user_interactions[:, -1]
        type_counts = Counter(interaction_types)
        sum_total = type_counts.get('retweet_only', 0) + type_counts.get('reply', 0) +\
                    type_counts.get('mention', 0) + type_counts.get('retweet_quote', 0)
        features['retweet_only_ratio'] = type_counts.get('retweet_only', 0) / sum_total
        features['reply_ratio'] = type_counts.get('reply', 0) / sum_total
        features['mention_ratio'] = type_counts.get('mention', 0) / sum_total
        features['retweet_quote_ratio'] = type_counts.get('retweet_quote', 0) / sum_total
        return features

    # Temporal patterns
    timestamps = sorted(user_interactions[:, 2])
    intervals = []
    sync_windows = 0
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        features['avg_inter_event_time'] = np.mean(intervals)
        features['std_inter_event_time'] = np.std(intervals)
        features['min_inter_event_time'] = np.min(intervals)
        features['max_inter_event_time'] = np.max(intervals)

        features['skewness'] = stats.skew(intervals)
        features['kurtosis'] = stats.kurtosis(intervals)

        # Synchronization score: how many interactions happen in tight windows
        sync_windows_0_1 = 0 # 10 seconds
        sync_windows_1 = 0 # 1 minutes
        sync_windows_5 = 0 # 5 minutes
        sync_windows_60 = 0 # 1 hour
        for i in range(len(intervals)):
            if intervals[i] < 10:
                sync_windows_0_1 += 1
            elif intervals[i] < 60:
                sync_windows_1 += 1
            elif intervals[i] < 60*5:
                sync_windows_5 += 1
            elif intervals[i] < 3600:
                sync_windows_60 += 1

        features['sync_ratio_0_1'] = sync_windows_0_1 / len(intervals)
        features['sync_ratio_1'] = sync_windows_1 / len(intervals)
        features['sync_ratio_5'] = sync_windows_5 / len(intervals)
        features['sync_ratio_60'] = sync_windows_60 / len(intervals)

        mu = np.mean(intervals)
        sigma = np.std(intervals)

        features['burstiness'] = (sigma ** 2) / mu if mu > 0 else 0

        # Entropy of interval distribution
        hist, _ = np.histogram(intervals, bins=min(20, len(intervals)))
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        features['interval_entropy'] = -np.sum(hist * np.log2(hist))

        # Gini coefficient: inequality measure
        # 0 = all intervals equal (regular), 1 = maximum inequality
        sorted_intervals = np.sort(intervals)
        n = len(intervals)
        index = np.arange(1, n + 1)
        features['gini_coefficient'] = (2 * np.sum(index * sorted_intervals)) / (n * np.sum(sorted_intervals)) - (
                    n + 1) / n

        # Memory coefficient: autocorrelation at lag 1
        if len(intervals) > 1:
            features['memory_coefficient'] = np.corrcoef(intervals[:-1], intervals[1:])[0, 1]
        else:
            features['memory_coefficient'] = 0

        # Short interval ratio: proportion of intervals below 10th percentile
        threshold = np.percentile(intervals, 10)
        features['short_interval_ratio'] = np.mean(intervals < threshold)

        # Long interval ratio: proportion of intervals above 90th percentile
        threshold = np.percentile(intervals, 90)
        features['long_interval_ratio'] = np.mean(intervals > threshold)

        # Simple change point detection using moving average
        window_size = max(3, len(intervals) // 5)

        # Compute moving average
        moving_avg = np.convolve(intervals, np.ones(window_size) / window_size, mode='valid')

        # Detect significant changes
        if len(moving_avg) > 1:
            changes = np.abs(np.diff(moving_avg))
            threshold = np.mean(changes) + 2 * np.std(changes)

            n_changes = np.sum(changes > threshold)
            features['n_change_points'] = n_changes
            features['change_point_ratio'] = n_changes / len(intervals)
        else:
            features['n_change_points'] = 0
            features['change_point_ratio'] = 0

        # Variance change detection
        # Split into halves and compare
        mid = len(intervals) // 2
        if mid > 1:
            first_half_var = np.var(intervals[:mid])
            second_half_var = np.var(intervals[mid:])

            # Ratio of variances (log scale to handle extremes)
            features['variance_change'] = np.log(second_half_var / first_half_var) if first_half_var > 0 else 0
        else:
            features['variance_change'] = 0

        if len(intervals) < 2:
            features['avg_acceleration'] = 0
            features['std_acceleration'] = 0
            features['max_acceleration'] = 0
        else:
            acceleration = np.diff(intervals)

            features['avg_acceleration'] = np.mean(acceleration)
            features['std_acceleration'] = np.std(acceleration)
            features['max_acceleration'] = np.max(np.abs(acceleration))

            # Velocity changes: number of speed-ups vs slow-downs
            speed_ups = np.sum(acceleration < 0)
            slow_downs = np.sum(acceleration > 0)

            features['speed_up_ratio'] = speed_ups / len(acceleration) if len(acceleration) > 0 else 0
            features['slow_down_ratio'] = slow_downs / len(acceleration) if len(acceleration) > 0 else 0

            # Jerk: rate of change of acceleration
            if len(acceleration) > 1:
                jerk = np.diff(acceleration)
                features['avg_jerk'] = np.mean(np.abs(jerk))
                features['max_jerk'] = np.max(np.abs(jerk))
            else:
                features['avg_jerk'] = 0
                features['max_jerk'] = 0

    else:
        features['avg_inter_event_time'] = 0
        features['std_inter_event_time'] = 0
        features['min_inter_event_time'] = 0
        features['max_inter_event_time'] = 0
        features['sync_ratio_0_1'] = 0
        features['sync_ratio_1'] = 0
        features['sync_ratio_5'] = 0
        features['sync_ratio_60'] = 0
        features['skewness'] = 0
        features['kurtosis'] = 0
        features['interval_entropy'] = 0
        features['gini_coefficient'] = 0
        features['memory_coefficient'] = 0
        features['burstiness'] = 0
        features['short_interval_ratio'] = 0
        features['long_interval_ratio'] = 0
        features['n_change_points'] = 0
        features['change_point_ratio'] = 0
        features['variance_change'] = 0
        features['avg_acceleration'] = 0
        features['std_acceleration'] = 0
        features['max_acceleration'] = 0
        features['speed_up_ratio'] = 0
        features['slow_down_ratio'] = 0
        features['avg_jerk'] = 0
        features['max_jerk'] = 0

    hour_bins = [int(ts) for s, t, ts, itype in user_interactions]
    hour_distribution = Counter(hour_bins)
    if hour_distribution:
        max_hour_activity = max(hour_distribution.values())
        features['temporal_concentration'] = max_hour_activity / len(user_interactions)
    else:
        features['temporal_concentration'] = 0

    return features


def prepare_user_classification_dataset(network: TemporalNetwork, labels: Dict[str, int]):
    """
    Prepare dataset for user-level malicious behavior classification

    Returns:
        X: Feature matrix
        y: Labels (0=benign, 1=malicious)
        users: List of user IDs
    """


    print(f"\nExtracting features for {len(network.nodes)} users...")

    sorted_users = sorted(network.nodes)

    # Serial processing
    X_data = []
    y_data = []
    user_list = []

    for idx, user in tqdm(enumerate(sorted_users), total=len(sorted_users)):

        features = extract_user_features(network, user)
        if features is None:
            continue

        X_data.append(features)
        y_data.append(labels.get(user))
        user_list.append(user)

    # Convert to DataFrame
    X = pd.DataFrame(X_data)
    y = np.array(y_data)

    # Handle any NaN or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, user_list


def train_malicious_user_classifier(X, y, user_list):
    """
    Train multiple classifiers for malicious user detection

    Uses stratified split to maintain class balance
    """
    print("\n" + "="*70)
    print("MALICIOUS USER CLASSIFICATION")
    print("="*70)

    # Check class distribution
    print(f"\nDataset shape: {X.shape}")
    print(f"Class distribution: {Counter(y)}")
    malicious_count = sum(y)
    benign_count = len(y) - malicious_count
    print(f"  Malicious users: {malicious_count} ({100*malicious_count/len(y):.1f}%)")
    print(f"  Benign users: {benign_count} ({100*benign_count/len(y):.1f}%)")

    seeds = [42, 45837, 92014, 18653, 60428]
    results = {}
    for seed in seeds:
        results[seed] = {}
        # Split data (stratified to maintain class balance)
        X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
            X, y, user_list, test_size=0.3, random_state=seed, stratify=y
        )

        if foundation:
            X_train = None
            y_train = None
            datasets = ['russia', 'cuba', 'iran', 'china', 'uae']
            for d in datasets:
                if d == dataset:
                    continue
                if "all" in ablation:
                    X_tmp, y_tmp, _ = read_features_ablation(d)
                else:
                    X_tmp = pd.read_csv('data/processed/{}_{}_{}_X.csv'.format(d, ablation, delta_seconds))
                    y_tmp = np.load('data/processed/{}_y.npy'.format(d))

                if X_train is None:
                    X_train = X_tmp
                    y_train = y_tmp
                else:
                    X_train = pd.concat((X_train, X_tmp))
                    y_train = np.concatenate((y_train, y_tmp))

        #     print(d, "finished")

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train class distribution: {Counter(y_train)}")
        print(f"Test class distribution: {Counter(y_test)}")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("scaled")

        # Define models
        if foundation:
            models = {
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=5,
                    random_state=seed
                ),
            }
        else:
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    random_state=seed, class_weight='balanced'
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=5,
                    random_state=seed
                ),
                'MLP': MLPClassifier(random_state=seed, max_iter=300, learning_rate_init=0.001),
                'SVM': SVC(kernel='rbf', probability=True, random_state=seed, class_weight='balanced'),
                'IF': IsolationForest(random_state=seed, n_estimators=200),
                'OneClassSVM': OneClassSVM(),
                'KNN': KNN()
            }


        for name, model in models.items():

            if name == "KNN":
                model.fit(X_train_scaled)
            else:
                if foundation:
                    import pickle
                    if not os.path.exists('{}_{}_{}_foundation.pkl'.format(dataset, name, seed)):
                        model.fit(X_train_scaled, y_train)
                        # model.save_model('{}_{}_foundation.json'.format(dataset, seed))
                        with open('{}_{}_{}_foundation.pkl'.format(dataset, name, seed), 'wb') as f:
                            pickle.dump(model, f)
                    else:
                        with open('{}_{}_{}_foundation.pkl'.format(dataset, name, seed), 'rb') as f:
                            model = pickle.load(f)
                else:
                    model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            if name in ["IF", "OneClassSVM", "LocalOutlierFactor"]:
                y_pred = (y_pred == -1).astype(int)
                y_pred_proba = -model.decision_function(X_test_scaled) # model.score_samples returns high values for normal points (0 -> Organic users), low values for abormal points (1 -> IO drivers)
            else:
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)

            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_micro = f1_score(y_test, y_pred, average="micro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            # Store results
            results[seed][name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc': auc,
                'ap': avg_precision,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'f1_weighted': f1_weighted,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
            }
            # print(model, f1_macro)


    return results, X.columns



def get_top_predictive_features(model, feature_names, top_n=15):
    """Get the most important features for classification"""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 60)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

    elif hasattr(model, 'coef_'):
        # Linear models
        coefficients = np.abs(model.coef_[0])
        indices = np.argsort(coefficients)[::-1][:top_n]

        print(f"\nTop {top_n} Most Important Features (by coefficient magnitude):")
        print("-" * 60)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_names[idx]:30s} {coefficients[idx]:.4f}")


def reindex_users(interactions_df):
    """
    Reindex all users in the interactions DataFrame

    Args:
        interactions_df: DataFrame with columns:
            - userid: Source user ID (string or int)
            - interaction_with_userid: Target user ID (string or int)
            - group: User group/label (optional)
            - interaction_form: Type of interaction (retweet, reply, etc.)
            - time: Timestamp of interaction
            - interaction_type: Additional interaction metadata (optional)

    Returns:
        reindexed_df: DataFrame with integer user IDs
        user_to_idx: Dictionary mapping original user ID to integer index
        idx_to_user: Dictionary mapping integer index to original user ID
    """


    users_from_source = set(interactions_df['userid'].unique())
    users_from_source = {u for u in users_from_source if pd.notna(u)}
    sorted_users = sorted(users_from_source, key=str)
    user_to_idx = {user: idx for idx, user in enumerate(sorted_users)}


    users_from_target = set(interactions_df['interaction_with_userid'].unique())-set(interactions_df['userid'].unique())
    users_from_target = {u for u in users_from_target if pd.notna(u)}
    sorted_users = sorted(users_from_target, key=str)
    last_idx = len(users_from_source)
    for idx, user in enumerate(sorted_users):
        user_to_idx[user] = last_idx + idx

    reindexed_df = interactions_df.copy()

    # Map user IDs to indices
    reindexed_df['userid_idx'] = reindexed_df['userid'].map(user_to_idx)
    reindexed_df['interaction_with_userid_idx'] = reindexed_df['interaction_with_userid'].map(user_to_idx)

    # Check for unmapped users (NaN values)
    unmapped_source = reindexed_df['userid_idx'].isna().sum()
    unmapped_target = reindexed_df['interaction_with_userid_idx'].isna().sum()

    if unmapped_source > 0 or unmapped_target > 0:
        print(f"  ⚠ Warning: {unmapped_source} unmapped source users, {unmapped_target} unmapped target users")
        # Remove rows with unmapped users
        reindexed_df = reindexed_df.dropna(subset=['userid_idx', 'interaction_with_userid_idx'])
        print(f"  Removed rows with unmapped users. New shape: {reindexed_df.shape}")

    # Convert to integers
    reindexed_df['userid_idx'] = reindexed_df['userid_idx'].astype(int)
    reindexed_df['interaction_with_userid_idx'] = reindexed_df['interaction_with_userid_idx'].astype(int)

    # print(f"  ✓ Reindexing complete")
    # print(f"  Reindexed DataFrame shape: {reindexed_df.shape}")

    # Reorder columns
    cols_order = ['userid', 'userid_idx', 'interaction_with_userid',
                  'interaction_with_userid_idx', 'time']

    # Add remaining columns
    for col in reindexed_df.columns:
        if col not in cols_order:
            cols_order.append(col)

    reindexed_df = reindexed_df[cols_order]

    # print("\n[Step 4] Summary Statistics")
    # print("-" * 80)
    # print(f"  Total interactions: {len(reindexed_df)}")
    # print(f"  Total users: {len(user_to_idx)}")
    # print(f"  Average interactions per user: {len(reindexed_df) / len(user_to_idx):.2f}")

    # Interaction statistics
    if 'interaction_form' in reindexed_df.columns:
        print(f"\n  Interaction forms:")
        for form, count in reindexed_df['interaction_form'].value_counts().items():
            print(f"    {form}: {count} ({100 * count / len(reindexed_df):.1f}%)")

    # Time range
    if 'time' in reindexed_df.columns:
        time_col = pd.to_datetime(reindexed_df['time'], errors='coerce')
        if time_col.notna().any():
            print(f"\n  Time range:")
            print(f"    Start: {time_col.min()}")
            print(f"    End: {time_col.max()}")
            print(f"    Duration: {time_col.max() - time_col.min()}")

    # Group distribution (if available)
    if 'group' in reindexed_df.columns:
        print(f"\n  Group distribution:")
        for group, count in reindexed_df['group'].value_counts().items():
            print(f"    {group}: {count} interactions")

    return reindexed_df


def read_features_ablation(dataset):
    X_tmp = None
    if "no_motif" in ablation:
        X_no_motif = pd.read_csv('data/processed/{}_no_motif_0_X.csv'.format(dataset))
        X_tmp = X_no_motif

    if "interaction_types" in ablation:
        X_twitter_interaction_types = pd.read_csv('data/processed/{}_twitter_interaction_types_0_X.csv'.format(dataset))
        X_twitter_interaction_types = X_twitter_interaction_types[
            ['retweet_only_ratio', 'reply_ratio', 'mention_ratio', 'retweet_quote_ratio']]
        if X_tmp is not None:
            X_tmp = pd.concat((X_tmp, X_twitter_interaction_types), axis=1)
        else:
            X_tmp = X_twitter_interaction_types

    if "static_motif" in ablation:
        X_static_motif = pd.read_csv('data/processed/{}_static_motif_0_X.csv'.format(dataset))
        cols = X_static_motif.columns[['motif_' in c for c in X_static_motif.columns]]
        X_static_motif = X_static_motif[cols]
        if X_tmp is not None:
            X_tmp = pd.concat((X_tmp, X_static_motif), axis=1)
        else:
            X_tmp = X_static_motif

    if "temporal" in ablation:
        temporal_cols = ["A_PRE_III", "A_PRE_IIO", "A_PRE_IOI", "A_PRE_IOO", "A_PRE_OII", "A_PRE_OIO", "A_PRE_OOI",
                         "A_PRE_OOO",
                         "B_MID_III", "B_MID_IIO", "B_MID_IOI", "B_MID_IOO", "B_MID_OII", "B_MID_OIO", "B_MID_OOI",
                         "B_MID_OOO",
                         "C_POST_III", "C_POST_IIO", "C_POST_IOI", "C_POST_IOO", "C_POST_OII", "C_POST_OIO",
                         "C_POST_OOI", "C_POST_OOO",
                         "2NODE_III", "2NODE_IIO", "2NODE_IOI", "2NODE_IOO", "2NODE_OII", "2NODE_OIO", "2NODE_OOI",
                         "2NODE_OOO",
                         "TRI_1", "TRI_2", "TRI_3", "TRI_4", "TRI_5", "TRI_6", "TRI_7", "TRI_8"
                         ]

    if "temporal_motif_60" in ablation:
        X_temporal_motif_60 = pd.read_csv('data/processed/{}_temporal_motif_60_X.csv'.format(dataset))
        X_temporal_motif_60 = X_temporal_motif_60[temporal_cols]
        cols = X_temporal_motif_60.columns
        dict_renaming = {}
        for c in cols:
            dict_renaming[c] = 'tmotif_60_{}'.format(c)
        X_temporal_motif_60.rename(columns=dict_renaming, inplace=True)
        if X_tmp is not None:
            X_tmp = pd.concat((X_tmp, X_temporal_motif_60), axis=1)
        else:
            X_tmp = X_temporal_motif_60

    if "temporal_motif_3600" in ablation:

        X_temporal_motif_3600 = pd.read_csv('data/processed/{}_temporal_motif_3600_X.csv'.format(dataset))
        X_temporal_motif_3600 = X_temporal_motif_3600[temporal_cols]
        cols = X_temporal_motif_3600.columns
        dict_renaming = {}
        for c in cols:
            dict_renaming[c] = 'tmotif_3600_{}'.format(c)
        X_temporal_motif_3600.rename(columns=dict_renaming, inplace=True)
        if X_tmp is not None:
            X_tmp = pd.concat((X_tmp, X_temporal_motif_3600), axis=1)
        else:
            X_tmp = X_temporal_motif_3600

    y_tmp = np.load('data/processed/{}_y.npy'.format(dataset))
    user_list_tmp = np.load('data/processed/{}_user_list.npy'.format(dataset))

    return X_tmp, y_tmp, user_list_tmp

dataset = sys.argv[1]
generate_features = True if len(sys.argv) > 2 and sys.argv[2] == "1" else False
ablation = sys.argv[3] if len(sys.argv) > 3 else "static_motif" # no_motif, twitter_interaction_types, static_motif, temporal_motif
delta_seconds = int(sys.argv[4]) if len(sys.argv) > 4 else 0
foundation = True if len(sys.argv) > 5 and sys.argv[5] == "1" else False
print("DATASET", dataset, "Generate Features", generate_features, "Ablation", ablation, "Delta seconds", delta_seconds,
      "Foundation", foundation)

if generate_features:
    cols = ['userid', 'interaction_with_userid', 'group', 'interaction_form', 'time']

    interactions_df = pd.read_csv('data/{}/interactions_graph_full.csv'.format(dataset))
    interactions_df = interactions_df[cols]

    interactions_df = reindex_users(interactions_df)
    interactions_df['userid'] = interactions_df['userid_idx']
    interactions_df['interaction_with_userid'] = interactions_df['interaction_with_userid_idx']
    interactions_df = interactions_df[cols]

    interactions_df['time'] = pd.to_numeric(
        pd.to_datetime(interactions_df['time'], utc=True, format='mixed').values) / 10 ** 9
    interactions_dict = interactions_df
    users_labels = interactions_df[['userid', 'group']].drop_duplicates()

    users_labels_dict = dict(users_labels.to_numpy())
    nodes = set(users_labels['userid'].tolist())

    network = TemporalNetwork()

    network.nodes = nodes

    interactions_df = interactions_df[['userid', 'interaction_with_userid', 'time', 'interaction_form']].sort_values(
        'time')
    interactions = interactions_df.to_numpy()
    network.interactions = interactions

    interactions_df_grouped = interactions_df.groupby('userid')
    interactions_df_grouped_inverse = interactions_df.groupby('interaction_with_userid')
    users_interactions = {}
    for node in list(nodes):
        out_int = interactions_df_grouped.get_group(node).to_numpy()
        try:
            in_int = interactions_df_grouped_inverse.get_group(node).to_numpy()
            users_interactions[node] = np.concatenate((out_int, in_int))
        except:
            users_interactions[node] = out_int

    network.interactions_dict = users_interactions

    print(f"  Network created with {len(network.nodes)} users")
    print(f"  Total interactions: {len(network.interactions)}")

    malicious_users = [u for u, l in users_labels_dict.items() if l == "IO"]
    benign_users = [u for u, l in users_labels_dict.items() if l == "Control"]

    print(f"  - Malicious users: {len(malicious_users)}")
    print(f"  - Benign users: {len(benign_users)}")

    # Step 2: Prepare classification dataset
    print("\n[Step 2] Preparing user classification dataset...")
    G = network.get_snapshot(None)
    H = nx.DiGraph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    if ablation == "no_motif" or ablation == "twitter_interaction_types":
        X, y, user_list = prepare_user_classification_dataset(network, users_labels_dict)
    elif ablation == "static_motif":
        graphlet_features = compute_graphlets_pyfglt(H.to_undirected())
        X = graphlet_features.iloc[:len(network.nodes)]
        #X, y, user_list = prepare_user_classification_dataset(network, users_labels_dict)

    elif ablation == "temporal_motif":
        self_loops = list(nx.selfloop_edges(H))
        H.remove_edges_from(self_loops)

        g = Graph()

        for node in network.nodes:
            g.add_node(timestamp=1, id=node)

        for inter in network.interactions:
            g.add_edge(timestamp=int(inter[2]), src=inter[0], dst=inter[1])

        res = algorithms.local_temporal_three_node_motifs(g, delta_seconds)
        cols_names = ["A_PRE_III", "A_PRE_IIO", "A_PRE_IOI", "A_PRE_IOO", "A_PRE_OII", "A_PRE_OIO", "A_PRE_OOI", "A_PRE_OOO",
                      "B_MID_III", "B_MID_IIO", "B_MID_IOI", "B_MID_IOO", "B_MID_OII", "B_MID_OIO", "B_MID_OOI", "B_MID_OOO",
                      "C_POST_III", "C_POST_IIO", "C_POST_IOI", "C_POST_IOO", "C_POST_OII", "C_POST_OIO", "C_POST_OOI", "C_POST_OOO",
                      "2NODE_III", "2NODE_IIO", "2NODE_IOI", "2NODE_IOO", "2NODE_OII", "2NODE_OIO", "2NODE_OOI", "2NODE_OOO",
                      "TRI_1", "TRI_2", "TRI_3", "TRI_4", "TRI_5", "TRI_6", "TRI_7", "TRI_8"
                      ]

        df = pd.DataFrame(np.array(list(res))[:len(network.nodes)], columns=cols_names)
        df.columns = df.columns.astype(str)
        X = df

    if ablation == "no_motif":
        y = (y == 'IO').astype('int')

        if not os.path.exists('data/processed'):
            os.makedirs("data/processed")

        user_list = np.array(user_list)
        y = np.array(y)

        user_list = user_list[X['total_interactions'] >= 10]
        y = y[X['total_interactions'] >= 10]
    else:
        users_total_interactions = np.array([len(network.get_user_interactions(user)) for user in sorted(network.nodes)])
        X['total_interactions'] = users_total_interactions

    X = X[X['total_interactions'] >= 10]

    del X['total_interactions']

    X.to_csv('data/processed/{}_{}_{}_X.csv'.format(dataset, ablation, delta_seconds), index=None)
    if ablation == "no_motif":
        np.save('data/processed/{}_y.npy'.format(dataset), y)
        np.save('data/processed/{}_user_list.npy'.format(dataset), user_list)

    y = np.load('data/processed/{}_y.npy'.format(dataset))
    user_list = np.load('data/processed/{}_user_list.npy'.format(dataset))

else:
    if "all" in ablation:
        X, y, user_list = read_features_ablation(dataset)
    else:
        X = pd.read_csv('data/processed/{}_{}_{}_X.csv'.format(dataset, ablation, delta_seconds))
        y = np.load('data/processed/{}_y.npy'.format(dataset))
        user_list = np.load('data/processed/{}_user_list.npy'.format(dataset))

# Step 3: Train classifiers
# print("\n[Step 3] Training malicious user classifiers...")
results, feature_names = \
    train_malicious_user_classifier(X, y, user_list)

# Feature importance
# print("\n" + "="*70)
# get_top_predictive_features(results[42]['Random Forest']['model'], feature_names)
# print("="*70)

seeds = [42, 45837, 92014, 18653, 60428]
models = list(results[seeds[0]].keys())
for model in models:
    print(f"\n{model}:")
    for metric in ["auc", "ap", "f1_macro"]:
        res = np.array([results[seed][model][metric] for seed in seeds])
        print(f"  {metric} {res.mean():.4f} ± {res.std():.4f} - Scores: {res}")
