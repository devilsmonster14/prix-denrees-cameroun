import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def load_dataframe(db_session):
    from app.database import PriceObservation
    rows = db_session.query(PriceObservation).all()
    if not rows:
        return pd.DataFrame()
    data = [{
        "timestamp": r.timestamp,
        "ville": r.ville,
        "marche": r.marche,
        "produit": r.produit,
        "prix_unitaire": r.prix_unitaire,
        "devise": r.devise,
        "unite_mesure": r.unite_mesure,
        "remarque": r.remarque
    } for r in rows]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["mois"] = df["timestamp"].dt.month
    df["ville_code"] = df["ville"].astype("category").cat.codes
    df["produit_code"] = df["produit"].astype("category").cat.codes
    df["marche_code"] = df["marche"].astype("category").cat.codes
    return df


def describe_data(df):
    if df.empty:
        return {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}
    desc = df[numeric_cols].describe().to_dict()
    for col in numeric_cols:
        desc[col]["skew"] = float(df[col].skew())
        desc[col]["kurtosis"] = float(df[col].kurtosis())
    return desc


def categorical_summary(df, col=None):
    if df.empty:
        return {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if col:
        cat_cols = [col]
    result = {}
    for c in cat_cols:
        vc = df[c].value_counts().to_dict()
        mode = df[c].mode().iloc[0] if not df[c].mode().empty else None
        result[c] = {"frequencies": {str(k): int(v) for k, v in vc.items()}, "mode": str(mode)}
    return result


def correlation_matrix(df):
    if df.empty:
        return {"matrix": {}, "columns": []}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return {"matrix": {}, "columns": numeric_cols}
    corr = df[numeric_cols].corr().to_dict()
    return {"matrix": {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in corr.items()}, "columns": numeric_cols}


def plot_histograms(df):
    if df.empty:
        return []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []
    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2196F3")
        axes[i].set_title(f"Distribution de {col}", fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fréquence")
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return [{"title": "Histogrammes", "image": fig_to_base64(fig)}]


def plot_boxplots(df):
    if df.empty:
        return []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []
    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i], color="#4CAF50")
        axes[i].set_title(f"Boîte à moustaches - {col}", fontsize=11)
        axes[i].set_xlabel(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return [{"title": "Boxplots", "image": fig_to_base64(fig)}]


def plot_categorical(df):
    if df.empty:
        return []
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["remarque", "devise"]]
    if not cat_cols:
        return []
    n = len(cat_cols)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        vc = df[col].value_counts().head(15)
        sns.barplot(x=vc.values, y=vc.index, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Répartition de {col}", fontsize=11)
        axes[i].set_xlabel("Nombre")
        axes[i].set_ylabel(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return [{"title": "Catégorielles", "image": fig_to_base64(fig)}]


def plot_correlation_heatmap(df):
    if df.empty:
        return None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Matrice de corrélation", fontsize=14)
    return fig_to_base64(fig)


def plot_price_evolution(df):
    if df.empty or "timestamp" not in df.columns:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.to_period("M").astype(str)
    produits = df["produit"].unique()[:4]
    n = len(produits)
    if n == 0:
        return []
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 4*nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, prod in enumerate(produits):
        sub = df[df["produit"] == prod]
        if sub.empty:
            continue
        grouped = sub.groupby("date")["prix_unitaire"].mean().reset_index()
        axes[i].plot(grouped["date"], grouped["prix_unitaire"], marker="o", color="#FF5722")
        axes[i].set_title(f"Évolution - {prod}", fontsize=11)
        axes[i].set_xlabel("Mois")
        axes[i].set_ylabel("Prix (XAF)")
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return [{"title": "Évolution", "image": fig_to_base64(fig)}]


def plot_price_by_ville(df):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("ville")["prix_unitaire"].mean().sort_values().index
    sns.barplot(data=df, x="ville", y="prix_unitaire", order=order, ax=ax, palette="magma")
    ax.set_title("Prix moyen par ville", fontsize=14)
    ax.set_xlabel("Ville")
    ax.set_ylabel("Prix moyen (XAF)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_price_by_marche(df):
    if df.empty:
        return None
    top_marches = df["marche"].value_counts().head(10).index
    sub = df[df["marche"].isin(top_marches)]
    fig, ax = plt.subplots(figsize=(12, 6))
    order = sub.groupby("marche")["prix_unitaire"].mean().sort_values().index
    sns.barplot(data=sub, x="marche", y="prix_unitaire", order=order, ax=ax, palette="plasma")
    ax.set_title("Prix moyen par marché (top 10)", fontsize=14)
    ax.set_xlabel("Marché")
    ax.set_ylabel("Prix moyen (XAF)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig_to_base64(fig)


def run_regression(df, target, predictors):
    if df.empty or target not in df.columns:
        return {"error": "Données insuffisantes ou variable cible absente"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available = [p for p in predictors if p in numeric_cols and p != target]
    if not available:
        return {"error": "Aucun prédicteur numérique valide (différent de la cible)"}
    df_clean = df[[target] + available].dropna()
    if len(df_clean) < 3:
        return {"error": "Pas assez de données pour la régression"}
    try:
        X = df_clean[available].values.astype(float)
        y = df_clean[target].values.astype(float)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        n, p = X.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        intercept_val = model.intercept_
        if hasattr(intercept_val, 'item'):
            intercept_val = intercept_val.item()
        coefficients = {"intercept": float(intercept_val)}
        for i, pred in enumerate(available):
            coef_val = model.coef_[i]
            if hasattr(coef_val, 'item'):
                coef_val = coef_val.item()
            coefficients[pred] = float(coef_val)
        from statsmodels.api import OLS, add_constant
        X_sm = add_constant(X)
        ols_model = OLS(y, X_sm).fit()
        p_intercept = ols_model.pvalues[0]
        if hasattr(p_intercept, 'item'):
            p_intercept = p_intercept.item()
        p_values = {"intercept": float(p_intercept)}
        for i, pred in enumerate(available):
            pv = ols_model.pvalues[i + 1]
            if hasattr(pv, 'item'):
                pv = pv.item()
            p_values[pred] = float(pv)
        residuals = y - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.6, color="#2196F3")
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_title("Résidus vs Valeurs prédites")
        axes[0].set_xlabel("Valeurs prédites")
        axes[0].set_ylabel("Résidus")
        axes[1].scatter(y, y_pred, alpha=0.6, color="#4CAF50")
        axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        axes[1].set_title("Valeurs observées vs prédites")
        axes[1].set_xlabel("Observées")
        axes[1].set_ylabel("Prédites")
        plt.tight_layout()
        residual_plot = fig_to_base64(fig)
        return {
            "coefficients": coefficients,
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "p_values": p_values,
            "n_observations": int(n),
            "residual_plot": residual_plot
        }
    except Exception as e:
        return {"error": f"Erreur lors de la régression : {str(e)}"}


def run_pca(df, features=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if features:
        numeric_cols = [f for f in features if f in numeric_cols]
    if len(numeric_cols) < 2:
        return {"error": "Au moins 2 variables numériques requises pour l'ACP"}
    df_clean = df[numeric_cols].dropna()
    if len(df_clean) < 3:
        return {"error": "Pas assez de données pour l'ACP"}
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        explained_var = [float(v) for v in pca.explained_variance_ratio_]
        cumulative_var = [float(v) for v in np.cumsum(pca.explained_variance_ratio_)]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].bar(range(1, len(explained_var) + 1), explained_var, color="#2196F3", alpha=0.7, label="Individuelle")
        axes[0].plot(range(1, len(explained_var) + 1), cumulative_var, "ro-", label="Cumulée")
        axes[0].set_title("Variance expliquée par composante")
        axes[0].set_xlabel("Composante")
        axes[0].set_ylabel("Proportion de variance")
        axes[0].legend()
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, color="#4CAF50")
        axes[1].set_title("Plan factoriel (CP1 vs CP2)")
        axes[1].set_xlabel(f"CP1 ({explained_var[0]*100:.1f}%)")
        axes[1].set_ylabel(f"CP2 ({explained_var[1]*100:.1f}%)")
        loadings = pca.components_[:2].T
        for i, col in enumerate(numeric_cols):
            axes[1].annotate(col, (loadings[i, 0]*2, loadings[i, 1]*2), fontsize=9, color="red")
            axes[1].arrow(0, 0, loadings[i, 0]*2, loadings[i, 1]*2, color="red", alpha=0.5, head_width=0.05)
        plt.tight_layout()
        pca_plot = fig_to_base64(fig)
        return {
            "explained_variance": explained_var,
            "cumulative_variance": cumulative_var,
            "n_components": len(explained_var),
            "pca_plot": pca_plot
        }
    except Exception as e:
        return {"error": f"Erreur lors de l'ACP : {str(e)}"}


def run_kmeans(df, n_clusters=3, features=None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if features:
        numeric_cols = [f for f in features if f in numeric_cols]
    if len(numeric_cols) < 2:
        return {"error": "Au moins 2 variables numériques requises pour le clustering"}
    df_clean = df[numeric_cols].dropna()
    if len(df_clean) < n_clusters:
        return {"error": f"Pas assez de données pour {n_clusters} clusters"}
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        cluster_sizes = {str(k): int(v) for k, v in cluster_sizes.items()}
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_dict = {}
        for i in range(n_clusters):
            centers_dict[f"Cluster {i}"] = {col: float(centers[i, j]) for j, col in enumerate(numeric_cols)}
        if X_scaled.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", alpha=0.6)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c="red", marker="X")
            ax.set_title(f"K-means ({n_clusters} clusters)")
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            plt.colorbar(scatter, label="Cluster")
            plt.tight_layout()
            cluster_plot = fig_to_base64(fig)
        else:
            cluster_plot = None
        inertias = []
        K_range = range(2, min(11, len(df_clean)))
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(list(K_range), inertias, "bo-")
        ax2.set_title("Méthode du coude")
        ax2.set_xlabel("Nombre de clusters")
        ax2.set_ylabel("Inertie")
        plt.tight_layout()
        elbow_plot = fig_to_base64(fig2)
        return {
            "cluster_sizes": cluster_sizes,
            "cluster_centers": centers_dict,
            "cluster_plot": cluster_plot,
            "elbow_plot": elbow_plot,
            "inertia": float(kmeans.inertia_)
        }
    except Exception as e:
        return {"error": f"Erreur lors du clustering K-Means : {str(e)}"}


def run_knn(df, target, k=5):
    if df.empty or target not in df.columns:
        return {"error": "Variable cible absente"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        return {"error": "Aucune variable numérique disponible"}
    df_clean = df[numeric_cols + [target]].dropna()
    if len(df_clean) < 5:
        return {"error": "Pas assez de données pour la classification"}
    try:
        le = LabelEncoder()
        y = le.fit_transform(df_clean[target].astype(str))
        X = df_clean[numeric_cols].values.astype(float)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return {"error": "La variable cible doit avoir au moins 2 classes"}
        # Check if all classes have enough members for stratification
        class_counts = pd.Series(y).value_counts()
        min_count = class_counts.min()
        use_stratify = min_count >= 2
        stratify_param = y if use_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify_param
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        k = min(k, len(X_train))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_s, y_train)
        y_pred = knn.predict(X_test_s)
        accuracy = float(knn.score(X_test_s, y_test))
        classes = [str(c) for c in le.classes_]
        all_labels = list(range(len(classes)))
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        report = classification_report(y_test, y_pred, labels=all_labels, target_names=classes, output_dict=True, zero_division=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"Matrice de confusion (k-NN, k={k})")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        plt.tight_layout()
        cm_plot = fig_to_base64(fig)
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "classes": classes,
            "k": k,
            "confusion_matrix_plot": cm_plot
        }
    except Exception as e:
        return {"error": f"Erreur lors de la classification K-NN : {str(e)}"}


def run_decision_tree(df, target):
    if df.empty or target not in df.columns:
        return {"error": "Variable cible absente"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        return {"error": "Aucune variable numérique disponible"}
    df_clean = df[numeric_cols + [target]].dropna()
    if len(df_clean) < 5:
        return {"error": "Pas assez de données pour la classification"}
    try:
        le = LabelEncoder()
        y = le.fit_transform(df_clean[target].astype(str))
        X = df_clean[numeric_cols].values.astype(float)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return {"error": "La variable cible doit avoir au moins 2 classes"}
        class_counts = pd.Series(y).value_counts()
        min_count = class_counts.min()
        use_stratify = min_count >= 2
        stratify_param = y if use_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify_param
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree.fit(X_train_s, y_train)
        y_pred = tree.predict(X_test_s)
        accuracy = float(tree.score(X_test_s, y_test))
        classes = [str(c) for c in le.classes_]
        all_labels = list(range(len(classes)))
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        report = classification_report(y_test, y_pred, labels=all_labels, target_names=classes, output_dict=True, zero_division=0)
        importance = {col: float(imp) for col, imp in zip(numeric_cols, tree.feature_importances_)}
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title("Matrice de confusion (Arbre de décision)")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        plt.tight_layout()
        cm_plot = fig_to_base64(fig)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ax2.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp], color="#4CAF50")
        ax2.set_title("Importance des variables")
        ax2.set_xlabel("Importance")
        plt.tight_layout()
        importance_plot = fig_to_base64(fig2)
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "classes": classes,
            "feature_importance": importance,
            "confusion_matrix_plot": cm_plot,
            "importance_plot": importance_plot
        }
    except Exception as e:
        return {"error": f"Erreur lors de l'arbre de décision : {str(e)}"}
