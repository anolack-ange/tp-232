# EduStat — Application de collecte et analyse des notes

Application Flask 100% Python pour la collecte, l'analyse descriptive et la prédiction des notes scolaires.

## Fonctionnalités

- **Collecte** : saisie des données étudiants (nom, filière, notes, absences, heures d'étude)
- **Liste** : tableau des étudiants avec mentions automatiques
- **Analyse descriptive** : statistiques (moyenne, médiane, écart-type), graphiques (distribution, mentions, filières, nuage de points)
- **Prédiction** : régression linéaire multiple OLS (sans bibliothèque externe) pour prédire la note finale

## Lancement local

```bash
pip install flask gunicorn
python app.py
# Accéder sur http://localhost:5000
```

## Déploiement sur Render.com (gratuit)

1. Créer un compte sur https://render.com
2. Nouveau projet → "Web Service"
3. Connecter votre dépôt GitHub contenant ce projet
4. Configurer :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app`
5. Déployer → obtenir le lien public

## Déploiement sur Railway.app (gratuit)

1. Créer un compte sur https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Railway détecte automatiquement Flask via le Procfile
4. Déployer → obtenir le lien public

## Structure du projet

```
edustat/
├── app.py              # Backend Flask + API REST + régression OLS
├── templates/
│   └── index.html      # Interface complète (HTML/CSS/JS)
├── requirements.txt
├── Procfile
└── data/
    └── etudiants.json  # Base de données locale (créée automatiquement)
```

## Modèle de prédiction

Régression linéaire multiple implementée from scratch (sans numpy/sklearn) :

```
Note_finale = β₀ + β₁×absences + β₂×moyenne_devoirs + β₃×heures_etude
```

Résolution par la méthode des moindres carrés ordinaires (OLS) :
```
β = (XᵀX)⁻¹ Xᵀy
```
