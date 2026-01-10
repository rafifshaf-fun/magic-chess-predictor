from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

dead_players = set()

@app.route('/api/session-stats', methods=['GET'])
def session_stats():
    return jsonify({'dead_players': list(dead_players), 'session_logs': 0})

@app.route('/api/toggle-death', methods=['POST'])
def toggle_death():
    data = request.json or {}
    player = data.get('player', '')
    if player in dead_players:
        dead_players.discard(player)
    else:
        dead_players.add(player)
    return jsonify({'dead_players': list(dead_players)})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    # TEMP: just confirm the wiring works
    return jsonify({
        "ok": True,
        "received": data
    }), 200

@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False)