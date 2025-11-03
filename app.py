# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
from blockchain import Blockchain
from ai_model import detect_anomalies, train_if_missing
import os
from datetime import datetime

app = Flask(__name__)
CHAIN_FILE = "chain.json"
bc = Blockchain(persist_file=CHAIN_FILE)

# ensure model exists (trains if missing)
train_if_missing()

# Simple in-memory products list for UI convenience (persisted inside blocks too)
# product record: {product_id, name, created_at}
products = {}

@app.route("/", methods=["GET"])
def index():
    chain = bc.to_list()
    events = bc.get_all_events()
    alerts = detect_anomalies(bc)
    # products list: from events collect known product ids
    known_prods = set([e["data"].get("product_id") for e in events if e["data"].get("product_id")])
    for pid in known_prods:
        if pid not in products:
            products[pid] = {"product_id": pid, "name": f"Product {pid}", "created_at": None}
    return render_template("index.html", chain=chain, events=events, alerts=alerts, products=list(products.values()), chain_valid=bc.is_chain_valid())

@app.route("/add_product", methods=["POST"])
def add_product():
    pid = request.form.get("product_id") or f"P{int(datetime.utcnow().timestamp())}"
    name = request.form.get("name") or f"Product {pid}"
    products[pid] = {"product_id": pid, "name": name, "created_at": datetime.utcnow().isoformat()}
    # also log a genesis-style event for product creation
    data = {
        "product_id": pid,
        "stage": "Created",
        "location": request.form.get("location","Factory"),
        "transit_time_hours": 0,
        "skipped_stage": 0,
        "is_duplicate": 0
    }
    bc.add_block(data)
    return redirect(url_for("index"))

@app.route("/add_event", methods=["POST"])
def add_event():
    pid = request.form.get("product_id")
    stage = request.form.get("stage")
    location = request.form.get("location")
    transit = float(request.form.get("transit_time_hours") or 0)
    skipped = int(request.form.get("skipped_stage") or 0)
    dup = int(request.form.get("is_duplicate") or 0)
    data = {
        "product_id": pid,
        "stage": stage,
        "location": location,
        "transit_time_hours": transit,
        "skipped_stage": skipped,
        "is_duplicate": dup
    }
    bc.add_block(data)
    return redirect(url_for("index"))

@app.route("/api/chain")
def api_chain():
    return jsonify(bc.to_list())

@app.route("/api/alerts")
def api_alerts():
    alerts = detect_anomalies(bc)
    return jsonify({"alerts": alerts})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
