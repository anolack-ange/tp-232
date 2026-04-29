from flask import Flask, render_template, request, jsonify
import json, os, statistics
from datetime import datetime

app = Flask(__name__)
DATA_FILE = "data/etudiants.json"

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/etudiants", methods=["GET"])
def get_etudiants():
    return jsonify(load_data())

@app.route("/api/etudiants", methods=["POST"])
def add_etudiant():
    data = load_data()
    etudiant = request.json
    etudiant["id"] = int(datetime.now().timestamp() * 1000)
    etudiant["date_ajout"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    data.append(etudiant)
    save_data(data)
    return jsonify({"success": True, "etudiant": etudiant})

@app.route("/api/etudiants/<int:eid>", methods=["DELETE"])
def delete_etudiant(eid):
    data = load_data()
    data = [e for e in data if e.get("id") != eid]
    save_data(data)
    return jsonify({"success": True})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    data = load_data()
    if not data:
        return jsonify({"error": "Aucune donnée"})

    notes = [e["note_finale"] for e in data if "note_finale" in e]
    absences = [e["absences"] for e in data if "absences" in e]
    devoirs = [e["moyenne_devoirs"] for e in data if "moyenne_devoirs" in e]
    filieres = {}
    mentions = {"Très Bien": 0, "Bien": 0, "Assez Bien": 0, "Passable": 0, "Insuffisant": 0}

    for e in data:
        f = e.get("filiere", "Autre")
        filieres[f] = filieres.get(f, 0) + 1
        n = e.get("note_finale", 0)
        if n >= 16: mentions["Très Bien"] += 1
        elif n >= 14: mentions["Bien"] += 1
        elif n >= 12: mentions["Assez Bien"] += 1
        elif n >= 10: mentions["Passable"] += 1
        else: mentions["Insuffisant"] += 1

    def safe_stats(lst):
        if not lst: return {}
        return {
            "moyenne": round(statistics.mean(lst), 2),
            "mediane": round(statistics.median(lst), 2),
            "min": round(min(lst), 2),
            "max": round(max(lst), 2),
            "ecart_type": round(statistics.stdev(lst) if len(lst) > 1 else 0, 2)
        }

    return jsonify({
        "total": len(data),
        "notes": safe_stats(notes),
        "absences": safe_stats(absences),
        "devoirs": safe_stats(devoirs),
        "filieres": filieres,
        "mentions": mentions,
        "distribution": sorted(notes),
        "scatter": [{"x": e.get("absences", 0), "y": e.get("note_finale", 0),
                     "nom": e.get("nom", "?")} for e in data]
    })

@app.route("/api/prediction", methods=["POST"])
def predict():
    """Régression linéaire multiple : note_finale ~ absences + moyenne_devoirs + heures_etude"""
    data = load_data()
    body = request.json

    valid = [e for e in data if all(k in e for k in ["note_finale", "absences", "moyenne_devoirs", "heures_etude"])]

    if len(valid) < 3:
        # Fallback: modèle par défaut basé sur des coefficients pédagogiques
        abs_val = float(body.get("absences", 0))
        dev_val = float(body.get("moyenne_devoirs", 10))
        heure_val = float(body.get("heures_etude", 5))
        prediction = round(max(0, min(20, 0.6 * dev_val - 0.3 * abs_val + 0.4 * heure_val + 2.5)), 2)
        return jsonify({
            "prediction": prediction,
            "methode": "modèle par défaut (données insuffisantes)",
            "r2": None,
            "n": len(valid)
        })

    # Régression linéaire multiple (OLS) implémentée manuellement
    n = len(valid)
    X = [[1, e["absences"], e["moyenne_devoirs"], e["heures_etude"]] for e in valid]
    y = [e["note_finale"] for e in valid]

    # XtX et Xty
    def mat_mul(A, B):
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]

    def mat_vec(A, v):
        return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

    def transpose(A):
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    def inverse4(M):
        n = len(M)
        aug = [M[i][:] + [1 if i == j else 0 for j in range(n)] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            div = aug[col][col]
            if abs(div) < 1e-10:
                return None
            aug[col] = [x / div for x in aug[col]]
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
        return [row[n:] for row in aug]

    Xt = transpose(X)
    XtX = mat_mul(Xt, X)
    Xty = mat_vec(Xt, y)
    XtX_inv = inverse4(XtX)

    if XtX_inv is None:
        prediction = round(statistics.mean(y), 2)
        return jsonify({"prediction": prediction, "methode": "moyenne (matrice singulière)", "r2": None, "n": n})

    beta = mat_vec(XtX_inv, Xty)

    # Prédiction
    x_new = [1, float(body.get("absences", 0)), float(body.get("moyenne_devoirs", 10)), float(body.get("heures_etude", 5))]
    pred = sum(beta[i] * x_new[i] for i in range(4))
    pred = round(max(0, min(20, pred)), 2)

    # R²
    y_mean = statistics.mean(y)
    y_hat = [sum(beta[i] * X[j][i] for i in range(4)) for j in range(n)]
    ss_res = sum((y[j] - y_hat[j]) ** 2 for j in range(n))
    ss_tot = sum((y[j] - y_mean) ** 2 for j in range(n))
    r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0

    mention = "Très Bien" if pred >= 16 else "Bien" if pred >= 14 else "Assez Bien" if pred >= 12 else "Passable" if pred >= 10 else "Insuffisant"

    return jsonify({
        "prediction": pred,
        "mention": mention,
        "methode": "régression linéaire multiple (OLS)",
        "r2": r2,
        "n": n,
        "coefficients": {
            "constante": round(beta[0], 4),
            "absences": round(beta[1], 4),
            "moyenne_devoirs": round(beta[2], 4),
            "heures_etude": round(beta[3], 4)
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
