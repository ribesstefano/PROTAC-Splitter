import joblib
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Literal

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False    

from .edge_features import extract_edge_features, get_edge_features


class GraphEdgeClassifier(BaseEstimator, ClassifierMixin):
    """
    Edge-level graph classifier for PROTACs with integrated pipeline building.
    """
    def __init__(
        self,
        graph_features: List[str],
        categorical_features: Optional[List[str]] = None,
        descriptor_features: Optional[List[str]] = None,
        fingerprint_features: Optional[List[str]] = None,
        use_descriptors: bool = True,
        use_fingerprints: bool = True,
        scaler_graph: Literal["passthrough", "standard"] = "passthrough",
        scaler_desc: Literal["passthrough", "standard"] = "passthrough",
        use_svd_fp: bool = True,
        n_svd_components: int = 100,
        binary: bool = False,
        smote_k_neighbors: Optional[int] = 5,
        xgb_params: Optional[dict] = None,
        n_bits: int = 512,
        radius: int = 6,
        descriptor_names: Optional[List[str]] = None
    ):
        self.graph_features = graph_features
        self.categorical_features = categorical_features
        self.descriptor_features = descriptor_features
        self.fingerprint_features = fingerprint_features
        self.use_descriptors = use_descriptors
        self.use_fingerprints = use_fingerprints
        self.scaler_graph = scaler_graph
        self.scaler_desc = scaler_desc
        self.use_svd_fp = use_svd_fp
        self.n_svd_components = n_svd_components
        self.binary = binary
        self.smote_k_neighbors = smote_k_neighbors
        self.xgb_params = xgb_params or {}
        self.n_bits = n_bits
        self.radius = radius
        self.descriptor_names = descriptor_names or [
            "MolWt", "HeavyAtomCount", "NumHAcceptors", "NumHDonors",
            "TPSA", "NumRotatableBonds", "RingCount", "MolLogP"
        ]
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        transformers = []
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features))
        if self.scaler_graph == "standard":
            transformers.append(("num", StandardScaler(), self.graph_features))
        else:
            transformers.append(("num", "passthrough", self.graph_features))

        if self.use_descriptors and self.descriptor_features:
            desc_block = (
                ("desc", StandardScaler(), self.descriptor_features)
                if self.scaler_desc == "standard"
                else ("desc", "passthrough", self.descriptor_features)
            )
            transformers.append(desc_block)

        if self.use_fingerprints and self.fingerprint_features:
            if self.use_svd_fp:
                fp_block = ("fp",
                    ImbPipeline([
                        ("svd", TruncatedSVD(n_components=self.n_svd_components, random_state=42))
                    ]),
                    self.fingerprint_features)
            else:
                fp_block = ("fp", "passthrough", self.fingerprint_features)
            transformers.append(fp_block)

        preprocessor = ColumnTransformer(transformers)

        # Define the classifier
        classifier = XGBClassifier(
            random_state=42,
            eval_metric="logloss" if self.binary else "mlogloss",
            objective="binary:logistic" if self.binary else "multi:softprob",
            **self.xgb_params
        )

        if self.smote_k_neighbors is not None:
            return ImbPipeline([
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42, k_neighbors=self.smote_k_neighbors)),
                ("clf", classifier)
            ])
        else:
            return Pipeline([
                ("preprocess", preprocessor),
                ("clf", classifier)
            ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, List[Dict], List[str]]) -> Any:
        X_proc = self._ensure_features(X)
        return self.pipeline.predict(X_proc)

    def predict_proba(self, X: Union[pd.DataFrame, List[Dict], List[str]]) -> Any:
        X_proc = self._ensure_features(X)
        return self.pipeline.predict_proba(X_proc)

    def save(self, path: Union[str, Path]):
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GraphEdgeClassifier":
        return joblib.load(str(path))

    @staticmethod
    def extract_graph_features(
        protac_smiles: Union[str, List[str]],
        wh_smiles: Optional[Union[str, List[str]]] = None,
        lk_smiles: Optional[Union[str, List[str]]] = None,
        e3_smiles: Optional[Union[str, List[str]]] = None,
        n_bits: int = 512,
        radius: int = 6,
        descriptor_names: Optional[List[str]] = None,
        verbose: int = 0
    ) -> pd.DataFrame:
        if any(x is None for x in [wh_smiles, lk_smiles, e3_smiles]):
            # Get features from PROTAC only, for inference
            return extract_edge_features(
                protac_smiles=protac_smiles,
                n_bits=n_bits,
                radius=radius,
                descriptor_names=descriptor_names,
            )
        else:
            # Get features and labels from all components, for training
            return get_edge_features(
                protac_smiles=protac_smiles,
                wh_smiles=wh_smiles,
                lk_smiles=lk_smiles,
                e3_smiles=e3_smiles,
                n_bits=n_bits,
                radius=radius,
                descriptor_names=descriptor_names,
                verbose=verbose
            )

    @staticmethod
    def build_multiclass_target(
        df: pd.DataFrame,
        poi_attachment_id: int = 1,
        e3_attachment_id: int = 2,
    ) -> pd.Series:
        """
        Returns multiclass target: 0 = no split, 1 = E3 split, 2 = WH split
        """
        assert ((df["label_e3_split"] + df["label_wh_split"]) <= 1).all()
        y = (
            df["label_wh_split"] * poi_attachment_id +
            df["label_e3_split"] * e3_attachment_id
        )
        return y.astype("int32")

    def _ensure_features(self, X: Union[pd.DataFrame, List[Dict], List[str]]) -> pd.DataFrame:
        """ Filter out features/columns that are are not used in the pipeline. """
        required_columns = (
            (self.graph_features or []) +
            (self.categorical_features or []) +
            (self.descriptor_features or []) +
            (self.fingerprint_features or [])
        )
        # If input is a DataFrame with SMILES, assume already featurized
        if isinstance(X, pd.DataFrame):
            Xf = X
        elif isinstance(X, list) and isinstance(X[0], dict):
            Xf = pd.DataFrame(X)
        else:
            raise ValueError("Provide either a DataFrame or list of feature dicts. Use extract_graph_features for SMILES.")
        missing = set(required_columns) - set(Xf.columns)
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}")
        return Xf[required_columns].copy()

    def predict_proba_from_smiles(
        self,
        protac_smiles: Union[str, List[str]],
        wh_smiles: Union[str, List[str]],
        lk_smiles: Union[str, List[str]],
        e3_smiles: Union[str, List[str]],
        verbose: int = 0,
    ):
        features = self.extract_graph_features(
            protac_smiles, wh_smiles, lk_smiles, e3_smiles,
            n_bits=self.n_bits,
            radius=self.radius,
            descriptor_names=self.descriptor_names,
            verbose=verbose
        )
        Xf = self._ensure_features(features)
        return self.pipeline.predict_proba(Xf)

    def predict_from_smiles(
        self,
        protac_smiles: Union[str, List[str]],
        wh_smiles: Union[str, List[str]],
        lk_smiles: Union[str, List[str]],
        e3_smiles: Union[str, List[str]],
        top_n: int = 1,
        return_array: bool = True,
        verbose: int = 0,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        For binary classification:
            For each SMILES, return the top_n edge chem_bond_idx indices among those predicted as class 1,
            sorted by predicted probability. If not enough edges are class 1, pad with -1.
        For multiclass:
            For each SMILES, return the chem_bond_idx with highest probability for class 1 (E3 split)
            and for class 2 (WH split). Shape: (num_smiles, 2).
            If no edge is predicted as that class, value is -1.
        """
        features = self.extract_graph_features(
            protac_smiles, wh_smiles, lk_smiles, e3_smiles,
            n_bits=self.n_bits,
            radius=self.radius,
            descriptor_names=self.descriptor_names,
            verbose=verbose
        )
        Xf = self._ensure_features(features)
        pred_proba = self.pipeline.predict_proba(Xf)
        pred_label = self.pipeline.predict(Xf)
        features = features.copy()
        features["pred_label"] = pred_label
        features["pred_proba"] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]

        unique_smiles = pd.Series(features["chem_mol_smiles"]).drop_duplicates().tolist()
        groupby = features.groupby("chem_mol_smiles")

        results = []

        if return_array:
            if pred_proba.shape[1] == 2:  # Binary case
                for mol_smiles in unique_smiles:
                    group = groupby.get_group(mol_smiles)
                    # Only consider edges predicted as label 1
                    edges_class1 = group[group["pred_label"] == 1]
                    # If none, pad with -1
                    if len(edges_class1) == 0:
                        results.append(np.full(top_n, -1))
                        continue
                    # Sort by proba, take top_n
                    top_edges = edges_class1.nlargest(top_n, "pred_proba")
                    idxs = top_edges["chem_bond_idx"].to_numpy()
                    if len(idxs) < top_n:
                        idxs = np.pad(idxs, (0, top_n - len(idxs)), constant_values=-1)
                    results.append(idxs[:top_n])
                return np.vstack(results)
            else:  # Multiclass case
                for mol_smiles in unique_smiles:
                    group = groupby.get_group(mol_smiles)
                    # For class 1
                    class1_idx = -1
                    if (group["pred_label"] == 1).any():
                        # Take the edge with highest class-1 probability
                        mask = group["pred_label"] == 1
                        idx1 = group.loc[mask, "pred_proba"].idxmax()
                        class1_idx = group.loc[idx1, "chem_bond_idx"]
                    # For class 2
                    class2_idx = -1
                    if (group["pred_label"] == 2).any():
                        mask = group["pred_label"] == 2
                        idx2 = group.loc[mask, "pred_proba"].idxmax()
                        class2_idx = group.loc[idx2, "chem_bond_idx"]
                    results.append([class1_idx, class2_idx])
                return np.array(results, dtype=int)
        else:
            return features

def get_classification_report(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(2)
    print(df_report)
    return df_report

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    if HAS_VISUALIZATION:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("Visualization libraries not available. Skipping confusion matrix plot.")
        print("Confusion Matrix:")
        print(cm)

def get_classification_report_and_plot(y_true, y_pred, labels):
    report = get_classification_report(y_true, y_pred, labels)
    plot_confusion_matrix(y_true, y_pred, labels)
    return report

def train_edge_classifier(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    model_filename: Optional[Union[str, Path]] = None,
    edge_classifier_kwargs: Optional[Dict[str, Any]] = None,
) -> GraphEdgeClassifier:
    """
    Train an edge-level graph classifier for PROTACs.
    
    Parameters:
        train_df (pd.DataFrame): Training data with features and labels.
        graph_features (List[str]): List of graph features to use.
        categorical_features (Optional[List[str]]): Categorical features to encode.
        descriptor_features (Optional[List[str]]): Descriptor features to use.
        fingerprint_features (Optional[List[str]]): Fingerprint features to use.
        use_descriptors (bool): Whether to use descriptors in the model.
        use_fingerprints (bool): Whether to use fingerprints in the model.
        scaler_graph (Literal["passthrough", "standard"]): Scaler for graph features.
        scaler_desc (Literal["passthrough", "standard"]): Scaler for descriptor features.
        use_svd_fp (bool): Whether to apply SVD on fingerprints.
        n_svd_components (int): Number of SVD components if used.
        binary (bool): Whether the task is binary classification.
        smote_k_neighbors (Optional[int]): Number of neighbors for SMOTE oversampling.
        xgb_params (Optional[dict]): Additional parameters for XGBClassifier.
        n_bits (int): Number of bits for fingerprint generation.
        radius (int): Radius for fingerprint generation.
        descriptor_names (Optional[List[str]]): Names of molecular descriptors to compute.

    Returns:
        GraphEdgeClassifier: Trained edge classifier instance.
    """
    sets = {}
    for set_name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if df is None or df.empty:
            continue
        print(f"Set: {set_name}, size: {len(df):,}")
        if 'PROTAC SMILES' not in df.columns or \
           'POI Ligand SMILES with direction' not in df.columns or \
           'Linker SMILES with direction' not in df.columns or \
           'E3 Binder SMILES with direction' not in df.columns:
            raise ValueError(f"DataFrame for {set_name} is missing required columns: 'PROTAC SMILES', 'POI Ligand SMILES with direction', 'Linker SMILES with direction', 'E3 Binder SMILES with direction'.")
        
        sets[set_name] = GraphEdgeClassifier.extract_graph_features(
            df['PROTAC SMILES'].tolist(),
            df['POI Ligand SMILES with direction'].tolist(),
            df['Linker SMILES with direction'].tolist(),
            df['E3 Binder SMILES with direction'].tolist(),
            verbose=1,
        )
        # Drop rows with label_e3_split + label_wh_split > 1
        sets[set_name] = sets[set_name][(sets[set_name]["label_e3_split"] + sets[set_name]["label_wh_split"]) <= 1]
        print(f"Set: {set_name}, size: {len(sets[set_name]):,}")
        
    train_set = sets["train"]
    label_cols = [c for c in train_set.columns if c.startswith("label_")]
    train_set = train_set.dropna(subset=label_cols)
    train_set = train_set[(train_set["label_e3_split"] + train_set["label_wh_split"]) <= 1]
    X_train = train_set.drop(columns=label_cols)
    y_train = train_set["label_is_split"].astype("int32") if clf.binary else GraphEdgeClassifier.build_multiclass_target(train_set)

    # Instantiate and train
    clf = GraphEdgeClassifier(**edge_classifier_kwargs or {
        "graph_features": [c for c in X_train.columns if c.startswith("graph_")],
        "categorical_features": ["chem_bond_type", "chem_atom_u", "chem_atom_v"],
        "fingerprint_features": [c for c in X_train.columns if c.startswith("chem_mol_fp_")],
        "use_descriptors": False,
        "use_fingerprints": True,
        "n_svd_components": 50,
        "binary": True,
        "smote_k_neighbors": 10,
        "xgb_params": {
            "max_depth": 6,
            "learning_rate": 0.3,
            "alpha": 0.1, # Default: 0
            "lambda": 0.5, # Default: 1
            "gamma": 0.1, # Default: 0
        },
    })

    clf.fit(X_train, y_train)
    
    if model_filename is not None:
        clf.save(model_filename)
        print(f"Model saved to {model_filename}")

    target_labels = ["No Split", "Split"] if clf.binary else ["No Split", "WH-Linker", "E3-Linker"]

    if val_df is not None:
        # Get validation data
        val_set = sets["val"].dropna(subset=label_cols)
        val_set = val_set[(val_set["label_e3_split"] + val_set["label_wh_split"]) <= 1]
        X_val = val_set.drop(columns=label_cols)
        y_val = val_set["label_is_split"].astype("int32") if clf.binary else GraphEdgeClassifier.build_multiclass_target(val_set)
        y_pred = clf.predict(X_val)
        report = get_classification_report_and_plot(y_val, y_pred, target_labels)
        print(f"Validation set classification report:\n{report.to_markdown(index=False)}")

    if test_df is not None:
        # Get test data
        test_set = sets["test"].dropna(subset=label_cols)
        test_set = test_set[(test_set["label_e3_split"] + test_set["label_wh_split"]) <= 1]
        X_test = test_set.drop(columns=label_cols)
        y_test = test_set["label_is_split"].astype("int32") if clf.binary else GraphEdgeClassifier.build_multiclass_target(test_set)
        y_pred = clf.predict(X_test)
        report = get_classification_report_and_plot(y_test, y_pred, target_labels)
        print(f"Test set classification report:\n{report.to_markdown(index=False)}")

    return clf