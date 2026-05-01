import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./prix_denrees.db")

# Render/Koyeb fournissent "postgres://" mais SQLAlchemy attend "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# connect_args spécifique à SQLite uniquement
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PriceObservation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ville = Column(String(100), nullable=False)
    marche = Column(String(100), nullable=False)
    produit = Column(String(100), nullable=False)
    prix_unitaire = Column(Float, nullable=False)
    devise = Column(String(10), default="XAF")
    unite_mesure = Column(String(50), nullable=False)
    remarque = Column(Text, nullable=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
