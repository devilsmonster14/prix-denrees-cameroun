# PrixDenrées — Suivi des prix alimentaires au Cameroun

> **Cartographie et suivi mensuel des prix des denrées alimentaires de base dans les marchés informels urbains (Yaoundé, Douala, Bamenda, Bafoussam, Garoua, Maroua)**

Application web complète de collecte, d'analyse et de visualisation des prix des denrées alimentaires sur les marchés informels urbains du Cameroun.

---

## Sommaire

1. [Architecture du projet](#architecture-du-projet)
2. [Structure des fichiers](#structure-des-fichiers)
3. [Base de données — `database.py`](#base-de-données--databasepy)
4. [Validation des données — `schemas.py`](#validation-des-données--schemaspy)
5. [Module d'analyse — `analytics.py`](#module-danalyse--analyticspy)
6. [Application principale — `main.py`](#application-principale--mainpy)
7. [Templates HTML](#templates-html)
8. [Frontend — CSS et JavaScript](#frontend--css-et-javascript)
9. [Installation et exécution locale](#installation-et-exécution-locale)
10. [Déploiement sur Render](#déploiement-sur-render)
11. [Changer le nom du lien du site](#changer-le-nom-du-lien-du-site)
12. [Dépendances](#dépendances)

---

## Architecture du projet

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Navigateur  │────▶│   FastAPI    │────▶│   SQLite /      │
│  (HTML/JS)   │◀────│  (main.py)   │◀────│   PostgreSQL    │
└──────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                     ┌─────▼──────┐
                     │ analytics  │
                     │  (pandas,  │
                     │  sklearn,  │
                     │  matplotlib)│
                     └────────────┘
```

- **Backend** : FastAPI (Python) — serveur asynchrone léger avec documentation auto-générée
- **Frontend** : HTML5 + Bootstrap 5 + JavaScript natif (fetch API)
- **Base de données** : SQLite en développement, PostgreSQL en production
- **Analyse** : Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn, Statsmodels
- **Validation** : Pydantic v2
- **Moteur de templates** : Jinja2

---

## Structure des fichiers

```
INF232/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── Procfile                           # Commande de déploiement Render
├── runtime.txt                        # Version Python (3.11.6)
├── prix_denrees.db                    # Base SQLite (créée automatiquement)
│
└── app/
    ├── __init__.py                     # Package Python vide
    ├── main.py                         # Application FastAPI (14 endpoints)
    ├── database.py                     # Modèle SQLAlchemy + connexion BDD
    ├── schemas.py                      # Modèles Pydantic de validation
    ├── analytics.py                    # 12 fonctions d'analyse et de visualisation
    │
    ├── static/
    │   ├── css/style.css               # Styles personnalisés
    │   └── sample_template.csv         # Modèle CSV pour l'import
    │
    ├── templates/
    │   ├── base.html                   # Layout principal (navbar + footer)
    │   ├── index.html                  # Page d'accueil
    │   ├── collecte.html               # Formulaire + import CSV/JSON
    │   ├── donnees.html                # Tableau des données
    │   ├── analytics.html              # Statistiques descriptives + graphiques
    │   ├── avance.html                 # Analyses avancées (ML)
    │   ├── report.html                 # Rapport imprimable/PDF
    │   └── about.html                  # Documentation & crédits
    │
    └── uploads/                        # Fichiers temporaire (créé auto.)
```

---

## Base de données — `database.py`

### Connexion

```python
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./prix_denrees.db")
```

- En local : utilise SQLite (`prix_denrees.db` créé à la racine)
- En production : lit la variable d'environnement `DATABASE_URL` (PostgreSQL fourni par Render)
- `connect_args={"check_same_thread": False}` : autorise l'accès multi-thread à SQLite

### Modèle `PriceObservation`

| Colonne        | Type       | Description                              |
|----------------|------------|------------------------------------------|
| `id`           | Integer    | Clé primaire auto-incrémentée            |
| `timestamp`    | DateTime   | Date/heure de l'observation (auto)       |
| `ville`        | String(100)| Ville (Yaoundé, Douala, etc.)            |
| `marche`       | String(100)| Nom du marché                            |
| `produit`      | String(100)| Nom du produit                           |
| `prix_unitaire`| Float      | Prix unitaire en XAF                     |
| `devise`       | String(10) | Devise (XAF par défaut)                  |
| `unite_mesure` | String(50) | Unité de mesure (kg, litre, tas, etc.)   |
| `remarque`     | Text       | Observation libre (optionnel)            |

### Fonction `get_db()`

Générateur de session SQLAlchemy injecté via `Depends(get_db)` dans chaque endpoint. La session est automatiquement fermée après la requête.

---

## Validation des données — `schemas.py`

Modèles Pydantic v2 qui valident automatiquement les entrées :

- **`ObservationCreate`** : validation des soumissions (`prix_unitaire > 0`, champs non vides, longueurs max)
- **`ObservationResponse`** : format de sortie (inclut `id` et `timestamp`)
- **`RegressionRequest`** : `target` (str) + `predictors` (list[str])
- **`ClusteringRequest`** : `n_clusters` (2–10) + `features` optionnel
- **`KNNRequest`** : `target` (str) + `k` (1–20)
- **`PredictionRequest`** : `features` (list[float])

Pydantic lève automatiquement une erreur 422 si les données sont invalides.

---

## Module d'analyse — `analytics.py`

Ce module contient toutes les fonctions de calcul et de visualisation. Il est indépendant de FastAPI et peut être réutilisé.

### Fonctions utilitaires

| Fonction | Rôle |
|----------|------|
| `fig_to_base64(fig)` | Convertit un graphique Matplotlib en chaîne base64 (SVG) pour l'embedding HTML |
| `load_dataframe(db_session)` | Charge toutes les observations en DataFrame Pandas + ajoute les colonnes dérivées |

### Colonnes dérivées (ajoutées par `load_dataframe`)

Les colonnes catégorielles sont encodées numériquement pour les algorithmes de ML :

| Colonne       | Type  | Origine                              |
|---------------|-------|--------------------------------------|
| `mois`        | int   | `timestamp.dt.month`                 |
| `ville_code`  | int   | `ville` encodée en catégorie (0, 1…) |
| `produit_code`| int   | `produit` encodé en catégorie        |
| `marche_code` | int   | `marche` encodé en catégorie         |

### Statistiques descriptives

| Fonction | Sortie |
|----------|--------|
| `describe_data(df)` | Dictionnaire : count, mean, std, min, 25%, 50%, 75%, max pour chaque variable numérique |
| `categorical_summary(df)` | Comptage et fréquence des variables catégorielles |
| `compute_correlation(df)` | Matrice de corrélation Pearson entre variables numériques |

### Visualisations (→ base64 SVG)

| Fonction | Graphique |
|----------|-----------|
| `plot_histograms(df)` | Histogrammes de chaque variable numérique |
| `plot_boxplots(df)` | Boîtes à moustaches par variable numérique |
| `plot_categorical(df)` | Diagrammes en barres des variables catégorielles |
| `plot_correlation_heatmap(df)` | Heatmap de corrélation (Seaborn) |
| `plot_price_evolution(df)` | Évolution temporelle du prix moyen |
| `plot_price_by_ville(df)` | Prix moyen par ville |
| `plot_price_by_marche(df)` | Prix moyen par marché (top 10) |

### Analyses avancées (Machine Learning)

Chaque fonction renvoie un dictionnaire JSON-sérialisable avec résultats + graphiques.

#### 1. Régression linéaire — `run_regression(df, target, predictors)`

- **Algorithme** : `LinearRegression` de scikit-learn + `OLS` de statsmodels pour les p-values
- **Entrée** : variable cible (ex: `prix_unitaire`) + liste de prédicteurs numériques
- **Sortie** : coefficients, R², R² ajusté, p-values, nombre d'observations, graphique des résidus
- **Protection** : la cible ne peut pas être dans les prédicteurs ; try/except autour du calcul

#### 2. Analyse en Composantes Principales (ACP) — `run_pca(df, features=None)`

- **Algorithme** : `PCA` de scikit-learn sur données standardisées (`StandardScaler`)
- **Sortie** : variance expliquée (individuelle + cumulée), nombre de composantes, graphique (éboulis + plan factoriel avec vecteurs variables)

#### 3. K-Means — `run_kmeans(df, n_clusters=3, features=None)`

- **Algorithme** : `KMeans` de scikit-learn sur données standardisées
- **Sortie** : tailles de clusters, centres (échelle originale), graphique des clusters, méthode du coude, inertie

#### 4. K-Plus Proches Voisins (k-NN) — `run_knn(df, target, k=5)`

- **Algorithme** : `KNeighborsClassifier` de scikit-learn
- **Entrée** : variable cible catégorielle (ex: `ville`, `produit`) + k
- **Sortie** : accuracy, matrice de confusion (graphique), rapport de classification
- **Protection** : vérifie que chaque classe a ≥ 2 membres pour la stratification ; fallback sans stratification si nécessaire ; labels explicites pour gérer les classes absentes du split de test

#### 5. Arbre de décision — `run_decision_tree(df, target)`

- **Algorithme** : `DecisionTreeClassifier` (max_depth=5)
- **Sortie** : accuracy, importance des variables (graphique), matrice de confusion (graphique), rapport de classification
- **Même protection** que k-NN pour la stratification et les labels

---

## Application principale — `main.py`

### Configuration

```python
VILLES = ["Yaoundé", "Douala", "Bamenda", "Bafoussam", "Garoua", "Maroua", "Bertoua", "Ebolowa"]
PRODUITS = ["Riz local", "Riz importé", "Huile de palme", ...]  # 16 produits
UNITES = ["kg", "litre", "tas", "pièce", "botte", "sac 25kg", "sac 50kg", "seau"]
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 Mo
```

### Endpoints HTML (pages)

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Page d'accueil avec bouton de chargement des données d'exemple |
| GET | `/collecte` | Formulaire de saisie manuelle + import CSV/JSON avec aperçu |
| GET | `/donnees-page` | Tableau paginé des observations avec suppression individuelle |
| GET | `/analytics` | Statistiques descriptives + 17 graphiques intégrés |
| GET | `/avance` | Interface des analyses avancées (5 algorithmes ML) |
| GET | `/report` | Rapport automatique imprimable / exportable en PDF |
| GET | `/about` | Documentation, guide utilisateur, technologies |

### Endpoints API (JSON)

| Méthode | Route | Corps (JSON) | Description |
|---------|-------|-------------|-------------|
| POST | `/submit` | `ObservationCreate` | Soumission JSON d'une observation |
| POST | `/submit-form` | Formulaire HTML | Soumission depuis le formulaire web |
| POST | `/import-file` | Fichier CSV/JSON | Import en masse avec validation |
| POST | `/import-preview` | Fichier CSV/JSON | Aperçu des 5 premières lignes |
| GET | `/donnees` | — | Toutes les observations en JSON |
| DELETE | `/donnees/{obs_id}` | — | Supprimer une observation |
| DELETE | `/clear-data` | — | Supprimer toutes les observations |
| GET | `/load-sample` | — | Charger ~1300 observations d'exemple |
| POST | `/api/regression` | `{"target": "...", "predictors": [...]}` | Régression linéaire |
| POST | `/api/pca` | `{"features": [...]}` (optionnel) | Analyse en composantes principales |
| POST | `/api/kmeans` | `{"n_clusters": 3, "features": [...]}` | Clustering K-Means |
| POST | `/api/knn` | `{"target": "...", "k": 5}` | Classification k-NN |
| POST | `/api/decision-tree` | `{"target": "..."}` | Arbre de décision |

### Documentation auto-générée

- **Swagger UI** : `/docs` — interface interactive pour tester les endpoints
- **ReDoc** : `/redoc` — documentation lisible

### Données d'exemple (`/load-sample`)

Génère aléatoirement ~1300 observations couvrant :
- 6 villes × leurs marchés (2–4 par ville)
- 16 produits avec plages de prix réalistes (en XAF)
- 12 mois de l'année 2025
- Unités de mesure adaptées à chaque produit
- Remarques aléatoires (Rupture partielle, Qualité variable, etc.)

### Sérialisation JSON robuste

L'endpoint `/donnees` nettoie les valeurs `NaN` et `inf` en les remplaçant par `None` pour éviter les erreurs de sérialisation JSON.

---

## Templates HTML

Tous les templates héritent de `base.html` via `{% extends "base.html" %}` et `{% block content %}`.

### `base.html`

- Barre de navigation Bootstrap 5 responsive
- Favicon SVG intégré (emoji 📊)
- Footer avec crédits
- Liens vers Bootstrap CDN, `style.css`, scripts JS

### `collecte.html`

- **Onglet 1** : Formulaire de saisie manuelle (ville, marché, produit, prix, unité, remarque)
- **Onglet 2** : Import CSV/JSON avec aperçu avant validation
- Validation côté client (champs requis, prix > 0)
- Envoi via `fetch()` en JavaScript

### `avance.html`

Interface en 5 sections, chacune avec un formulaire et une zone de résultats :

1. **Régression** : sélection de la cible et des prédicteurs → coefficients, R², p-values, graphique résidus
2. **ACP** : nombre de composantes → variance expliquée, plan factoriel
3. **K-Means** : nombre de clusters → visualisation, méthode du coude
4. **k-NN** : variable cible + k → accuracy, matrice de confusion
5. **Arbre de décision** : variable cible → accuracy, importance des variables, matrice de confusion

Les résultats sont affichés dynamiquement via `fetch()` et injectés dans le DOM.

---

## Frontend — CSS et JavaScript

### `style.css`

- Thème personnalisé avec couleurs cohérentes (bleu #2196F3, vert #4CAF50)
- Cartes hero sur la page d'accueil
- Styles pour les tables de données, formulaires, graphiques
- Media query `@media print` pour le rapport (masque la navigation)

### JavaScript (intégré dans les templates)

- **`collecte.html`** : `fetch('/submit-form')` pour le formulaire, `fetch('/import-file')` et `fetch('/import-preview')` pour l'import
- **`donnees.html`** : `fetch('/donnees')` pour charger les données, `fetch('/donnees/{id}', {method:'DELETE'})` pour supprimer
- **`avance.html`** : appels `fetch('/api/regression')`, `/api/pca`, `/api/kmeans`, `/api/knn`, `/api/decision-tree` avec affichage dynamique des résultats et graphiques base64

---

## Installation et exécution locale

### Prérequis

- Python 3.11+ (testé avec 3.12)
- pip

### Étapes

```bash
# 1. Cloner le projet
cd INF232

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le serveur
uvicorn app.main:app --host 0.0.0.0 --port 8080

# 5. Ouvrir dans le navigateur
# http://localhost:8080
```

### Charger les données d'exemple

Au premier lancement, la base est vide. Cliquez sur **« Charger les données d'exemple »** sur la page d'accueil, ou visitez :

```
http://localhost:8080/load-sample
```

Cela génère ~1300 observations aléatoires réalistes.

---

## Déploiement sur Render

### Principe

Render est une plateforme PaaS (Platform as a Service) qui héberge automatiquement votre application à partir d'un dépôt Git. Le principe :

1. Vous poussez le code sur GitHub
2. Render détecte le `Procfile` et `runtime.txt`
3. Il installe les dépendances de `requirements.txt`
4. Il exécute la commande du `Procfile`
5. L'application est accessible via une URL publique

### Fichiers de déploiement

**`Procfile`** — commande exécutée par Render :
```
web: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT
```
- `gunicorn` : serveur WSGI de production
- `-w 4` : 4 workers (processus parallèles)
- `-k uvicorn.workers.UvicornWorker` : utilise Uvicorn comme worker ASGI
- `-b 0.0.0.0:$PORT` : écoute sur le port fourni par Render (`$PORT`)

**`runtime.txt`** — version Python :
```
3.11.6
```

### Étapes de déploiement

1. **Créer un dépôt GitHub** contenant tout le projet
2. **Créer un compte** sur [render.com](https://render.com)
3. **New → Web Service** → Connecter le dépôt GitHub
4. Configurer :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : (laisser vide, le `Procfile` est utilisé automatiquement)
   - **Environment** : Python 3
5. **Create Web Service** — Render construit et déploie
6. L'URL est de la forme : `https://votre-app.onrender.com`

### Base de données en production

En local, SQLite est utilisé (`prix_denrees.db`). En production sur Render :

- Créer une **PostgreSQL database** dans Render
- Copier l'**Internal Database URL** fournie par Render
- L'ajouter comme variable d'environnement :
  - **Key** : `DATABASE_URL`
  - **Value** : `postgresql://user:password@host:port/dbname`

Le code de `database.py` lit automatiquement cette variable :
```python
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./prix_denrees.db")
```

> **Note** : Si vous utilisez PostgreSQL, ajoutez le package `psycopg2-binary` dans `requirements.txt`.

---

## Changer le nom du lien du site

L'URL publique de l'application est déterminée par le **nom du service** sur Render.

### Méthode 1 : lors de la création

Sur Render, lors de la création du Web Service, le champ **Name** définit l'URL :

- Si vous mettez `prix-denrees-yaounde`, l'URL sera `https://prix-denrees-yaounde.onrender.com`

### Méthode 2 : renommer un service existant

1. Aller sur le **Dashboard** Render
2. Cliquer sur votre **Web Service**
3. Aller dans **Settings**
4. Modifier le champ **Name**
5. **Save** — l'URL change immédiatement

> ⚠️ L'ancienne URL ne redirige plus. Mettez à jour tout lien partagé.

### Méthode 3 : domaine personnalisé

Si vous possédez votre propre nom de domaine (ex: `prix-cameroun.com`) :

1. Dans **Settings** → **Custom Domains**
2. Ajouter votre domaine
3. Configurer un enregistrement DNS **CNAME** chez votre registrar :
   ```
   prix-cameroun.com  →  CNAME  →  prix-denrees.onrender.com
   ```
4. Attendre la propagation DNS (quelques minutes à quelques heures)

### Noms recommandés

| Nom du service | URL obtenue |
|----------------|-------------|
| `prix-denrees-cameroun` | `https://prix-denrees-cameroun.onrender.com` |
| `inf232-prix-marches` | `https://inf232-prix-marches.onrender.com` |
| `suivi-prix-alimentaires` | `https://suivi-prix-alimentaires.onrender.com` |

---

## Dépendances

| Package | Version | Rôle |
|---------|---------|------|
| `fastapi` | 0.104.1 | Framework web asynchrone |
| `uvicorn` | 0.24.0 | Serveur ASGI |
| `starlette` | ≥0.27,<0.29 | Sous-couche de FastAPI |
| `sqlalchemy` | 2.0.44 | ORM base de données |
| `pydantic` | 2.5.2 | Validation de données |
| `python-multipart` | 0.0.27 | Parsing de formulaires/fichiers |
| `pandas` | 3.0.2 | Manipulation de données tabulaires |
| `numpy` | 2.4.4 | Calcul numérique |
| `scipy` | 1.17.1 | Statistiques avancées |
| `scikit-learn` | 1.8.0 | Machine Learning (régression, ACP, k-means, k-NN, arbre) |
| `matplotlib` | 3.10.9 | Visualisation graphique |
| `seaborn` | 0.13.2 | Visualisation statistique |
| `statsmodels` | 0.14.6 | Régression OLS + p-values |
| `Jinja2` | 3.1.6 | Moteur de templates HTML |
| `aiofiles` | 25.1.0 | Fichiers asynchrones |
| `gunicorn` | 21.2.0 | Serveur WSGI de production |

---

## Licence

Projet académique — INF232, Université de Yaoundé I.
