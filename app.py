from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# Manual CORS headers - no external dependency needed
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Serve CSV files from root directory
@app.route('/data/<filename>')
def serve_csv(filename):
    try:
        print(f"Serving: {filename}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current dir: {os.listdir('.')}")
        return send_from_directory('.', filename)
    except Exception as e:
        print(f"Error serving CSV: {e}")
        return jsonify({'error': str(e)}), 500

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