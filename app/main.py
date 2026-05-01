import os
import json
import pandas as pd
import logging
from datetime import datetime
from fastapi import FastAPI, Request, Depends, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db, Base, engine, PriceObservation
from app.schemas import ObservationCreate, ObservationResponse
from app import analytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PrixDenrées - Suivi des prix alimentaires",
    description="Application de collecte et d'analyse des prix des denrées alimentaires dans les marchés informels urbains du Cameroun",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 Mo

VILLES = ["Yaoundé", "Douala", "Bamenda", "Bafoussam", "Garoua", "Maroua", "Bertoua", "Ebolowa"]
PRODUITS = [
    "Riz local", "Riz importé", "Huile de palme", "Manioc (tubercule)",
    "Farine de manioc", "Tomate", "Oignon", "Poisson séché",
    "Poisson frais", "Viande de boeuf", "Poulet", "Maïs",
    "Haricot", "Sucre", "Sel", "Savon"
]
UNITES = ["kg", "litre", "tas", "pièce", "botte", "sac 25kg", "sac 50kg", "seau"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "villes": VILLES, "produits": PRODUITS, "unites": UNITES})


@app.get("/collecte", response_class=HTMLResponse)
async def collecte_page(request: Request):
    return templates.TemplateResponse("collecte.html", {"request": request, "villes": VILLES, "produits": PRODUITS, "unites": UNITES})


@app.post("/submit")
async def submit_observation(obs: ObservationCreate, db: Session = Depends(get_db)):
    try:
        row = PriceObservation(
            ville=obs.ville,
            marche=obs.marche,
            produit=obs.produit,
            prix_unitaire=obs.prix_unitaire,
            devise=obs.devise,
            unite_mesure=obs.unite_mesure,
            remarque=obs.remarque
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info(f"Nouvelle observation: {row.id} - {row.produit} à {row.marche}")
        return {"status": "success", "id": row.id}
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur soumission: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/submit-form")
async def submit_form(
    ville: str = Form(...),
    marche: str = Form(...),
    produit: str = Form(...),
    prix_unitaire: float = Form(...),
    devise: str = Form("XAF"),
    unite_mesure: str = Form(...),
    remarque: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        if prix_unitaire <= 0:
            raise HTTPException(status_code=400, detail="Le prix doit être positif")
        row = PriceObservation(
            ville=ville,
            marche=marche,
            produit=produit,
            prix_unitaire=prix_unitaire,
            devise=devise,
            unite_mesure=unite_mesure,
            remarque=remarque if remarque else None
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return JSONResponse({"status": "success", "id": row.id})
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur soumission formulaire: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/import-file")
async def import_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10 Mo)")
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10 Mo)")
    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep=None, engine="python")
        elif filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV ou JSON.")
        required = {"ville", "marche", "produit", "prix_unitaire", "unite_mesure"}
        cols_lower = {c.lower().strip() for c in df.columns}
        if not required.issubset(cols_lower):
            missing = required - cols_lower
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing}")
        col_map = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        df["prix_unitaire"] = pd.to_numeric(df["prix_unitaire"], errors="coerce")
        df = df.dropna(subset=["prix_unitaire"])
        df = df[df["prix_unitaire"] > 0]
        count = 0
        for _, row in df.iterrows():
            obs = PriceObservation(
                ville=str(row.get("ville", "")),
                marche=str(row.get("marche", "")),
                produit=str(row.get("produit", "")),
                prix_unitaire=float(row["prix_unitaire"]),
                devise=str(row.get("devise", "XAF")),
                unite_mesure=str(row.get("unite_mesure", "kg")),
                remarque=str(row.get("remarque", "")) if "remarque" in row and pd.notna(row.get("remarque")) else None
            )
            db.add(obs)
            count += 1
        db.commit()
        logger.info(f"Import: {count} observations depuis {filename}")
        return {"status": "success", "imported": count}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur import: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'import: {str(e)}")


@app.post("/import-preview")
async def import_preview(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep=None, engine="python", nrows=5)
        elif filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                df = pd.DataFrame(data[:5])
            else:
                df = pd.DataFrame([data])
        else:
            raise HTTPException(status_code=400, detail="Format non supporté")
        preview = df.head(5).fillna("").to_dict(orient="records")
        columns = df.columns.tolist()
        return {"preview": preview, "columns": columns, "rows": len(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/donnees")
async def get_data(db: Session = Depends(get_db)):
    import math
    df = analytics.load_dataframe(db)
    if df.empty:
        return {"count": 0, "data": []}
    df["timestamp"] = df["timestamp"].astype(str)
    records = df.to_dict(orient="records")
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj
    return {"count": len(df), "data": clean(records)}


@app.get("/donnees-page", response_class=HTMLResponse)
async def donnees_page(request: Request, db: Session = Depends(get_db)):
    df = analytics.load_dataframe(db)
    records = []
    if not df.empty:
        df["timestamp"] = df["timestamp"].astype(str)
        records = df.to_dict(orient="records")
    return templates.TemplateResponse("donnees.html", {"request": request, "records": records, "count": len(records)})


@app.delete("/donnees/{obs_id}")
async def delete_observation(obs_id: int, db: Session = Depends(get_db)):
    obs = db.query(PriceObservation).filter(PriceObservation.id == obs_id).first()
    if not obs:
        raise HTTPException(status_code=404, detail="Observation non trouvée")
    db.delete(obs)
    db.commit()
    return {"status": "deleted"}


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request, db: Session = Depends(get_db)):
    df = analytics.load_dataframe(db)
    if df.empty:
        return templates.TemplateResponse("analytics.html", {"request": request, "has_data": False, "count": 0})
    desc = analytics.describe_data(df)
    cat_summary = analytics.categorical_summary(df)
    corr = analytics.correlation_matrix(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in ["remarque", "devise"]]
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "has_data": True,
        "count": len(df),
        "description": desc,
        "categorical_summary": cat_summary,
        "correlation": corr,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols
    })


@app.get("/api/plot/{plot_type}")
async def api_plot(plot_type: str, db: Session = Depends(get_db)):
    df = analytics.load_dataframe(db)
    if df.empty:
        return {"error": "Aucune donnée"}
    plot_map = {
        "histograms": analytics.plot_histograms,
        "boxplots": analytics.plot_boxplots,
        "categorical": analytics.plot_categorical,
        "correlation": analytics.plot_correlation_heatmap,
        "evolution": analytics.plot_price_evolution,
        "ville": analytics.plot_price_by_ville,
        "marche": analytics.plot_price_by_marche,
    }
    fn = plot_map.get(plot_type)
    if not fn:
        raise HTTPException(status_code=404, detail=f"Plot type '{plot_type}' not found")
    try:
        result = fn(df)
        return {"result": result}
    except Exception as e:
        logger.warning(f"Plot error ({plot_type}): {e}")
        return {"error": str(e)}


@app.get("/avance", response_class=HTMLResponse)
async def avance_page(request: Request, db: Session = Depends(get_db)):
    df = analytics.load_dataframe(db)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist() if not df.empty else []
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in ["remarque", "devise"]] if not df.empty else []
    return templates.TemplateResponse("avance.html", {
        "request": request,
        "has_data": not df.empty,
        "count": len(df) if not df.empty else 0,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols
    })


@app.post("/api/regression")
async def api_regression(body: dict = Body(...), db: Session = Depends(get_db)):
    from app.schemas import RegressionRequest
    req = RegressionRequest(**body)
    df = analytics.load_dataframe(db)
    result = analytics.run_regression(df, req.target, req.predictors)
    return result


@app.post("/api/pca")
async def api_pca(body: dict = Body(default={}), db: Session = Depends(get_db)):
    features = body.get("features") if body else None
    df = analytics.load_dataframe(db)
    result = analytics.run_pca(df, features)
    return result


@app.post("/api/kmeans")
async def api_kmeans(body: dict = Body(default={}), db: Session = Depends(get_db)):
    n_clusters = body.get("n_clusters", 3) if body else 3
    features = body.get("features") if body else None
    df = analytics.load_dataframe(db)
    result = analytics.run_kmeans(df, n_clusters=n_clusters, features=features)
    return result


@app.post("/api/knn")
async def api_knn(body: dict = Body(...), db: Session = Depends(get_db)):
    from app.schemas import KNNRequest
    req = KNNRequest(**body)
    df = analytics.load_dataframe(db)
    result = analytics.run_knn(df, req.target, req.k)
    return result


@app.post("/api/decision-tree")
async def api_decision_tree(body: dict = Body(...), db: Session = Depends(get_db)):
    target = body.get("target") if body else None
    if not target:
        raise HTTPException(status_code=400, detail="Variable cible requise")
    df = analytics.load_dataframe(db)
    result = analytics.run_decision_tree(df, target)
    return result


@app.get("/load-sample")
async def load_sample(db: Session = Depends(get_db)):
    existing = db.query(PriceObservation).count()
    if existing > 0:
        return {"status": "info", "message": f"La base contient déjà {existing} observations. Supprimez-les d'abord si vous voulez recharger."}
    import random
    marches = {
        "Yaoundé": ["Mfoundi", "Mvog-Ada", "Nkolndongo", "Ekounou"],
        "Douala": ["New-Bell", "Sandaga", "Deido", "Bonaberi"],
        "Bamenda": ["Main Market", "Nkwen", "Ntarfon"],
        "Bafoussam": ["Marché A", "Marché B", "Kouogouo"],
        "Garoua": ["Grand Marché", "Quartier Fadil"],
        "Maroua": ["Marché Central", "Domayo"]
    }
    produits_prix = {
        "Riz local": (350, 600), "Riz importé": (500, 800),
        "Huile de palme": (400, 700), "Manioc (tubercule)": (100, 300),
        "Farine de manioc": (200, 450), "Tomate": (300, 800),
        "Oignon": (200, 600), "Poisson séché": (500, 1500),
        "Poisson frais": (800, 2500), "Viande de boeuf": (1500, 3000),
        "Poulet": (2000, 4000), "Maïs": (150, 350),
        "Haricot": (300, 600), "Sucre": (500, 800),
        "Sel": (100, 250), "Savon": (150, 400)
    }
    unites_map = {
        "Riz local": "kg", "Riz importé": "kg", "Huile de palme": "litre",
        "Manioc (tubercule)": "tas", "Farine de manioc": "kg", "Tomate": "kg",
        "Oignon": "kg", "Poisson séché": "kg", "Poisson frais": "kg",
        "Viande de boeuf": "kg", "Poulet": "pièce", "Maïs": "kg",
        "Haricot": "kg", "Sucre": "kg", "Sel": "kg", "Savon": "pièce"
    }
    count = 0
    base_date = datetime(2025, 1, 1)
    for month in range(1, 13):
        for ville, marche_list in marches.items():
            for marche in marche_list:
                n_produits = random.randint(4, 8)
                selected = random.sample(list(produits_prix.keys()), min(n_produits, len(produits_prix)))
                for produit in selected:
                    low, high = produits_prix[produit]
                    prix = round(random.uniform(low, high), 0)
                    mois = month
                    annee = 2025
                    jour = random.randint(1, 28)
                    ts = datetime(annee, mois, jour, random.randint(6, 17), random.randint(0, 59))
                    obs = PriceObservation(
                        timestamp=ts,
                        ville=ville,
                        marche=marche,
                        produit=produit,
                        prix_unitaire=prix,
                        devise="XAF",
                        unite_mesure=unites_map.get(produit, "kg"),
                        remarque=random.choice([None, None, None, "Rupture partielle", "Qualité variable", "Affluence élevée"])
                    )
                    db.add(obs)
                    count += 1
    db.commit()
    logger.info(f"Chargement échantillon: {count} observations")
    return {"status": "success", "imported": count}


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/report", response_class=HTMLResponse)
async def report_page(request: Request, db: Session = Depends(get_db)):
    df = analytics.load_dataframe(db)
    if df.empty:
        return templates.TemplateResponse("report.html", {"request": request, "has_data": False})
    desc = analytics.describe_data(df)
    cat_summary = analytics.categorical_summary(df)
    corr_heatmap = analytics.plot_correlation_heatmap(df)
    histograms = analytics.plot_histograms(df)
    boxplots = analytics.plot_boxplots(df)
    price_evolution = analytics.plot_price_evolution(df)
    price_by_ville = analytics.plot_price_by_ville(df)
    return templates.TemplateResponse("report.html", {
        "request": request,
        "has_data": True,
        "count": len(df),
        "description": desc,
        "categorical_summary": cat_summary,
        "corr_heatmap": corr_heatmap,
        "histograms": histograms,
        "boxplots": boxplots,
        "price_evolution": price_evolution,
        "price_by_ville": price_by_ville,
        "generated_date": datetime.now().strftime("%d/%m/%Y")
    })


@app.delete("/clear-data")
async def clear_data(db: Session = Depends(get_db)):
    count = db.query(PriceObservation).delete()
    db.commit()
    return {"status": "success", "deleted": count}
