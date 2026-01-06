from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='.')

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Your Render ML server URL
ML_SERVER = os.getenv("ML_SERVER_URL", "https://magic-chess-ml.onrender.com")

dead_players = set()

@app.route('/api/session-stats', methods=['GET'])
def session_stats():
    return jsonify({'dead_players': list(dead_players), 'session_logs': 0})

@app.route('/api/toggle-death', methods=['POST'])
def toggle_death():
    data = request.json
    player = data.get('player', '')
    
    if player in dead_players:
        dead_players.discard(player)
    else:
        dead_players.add(player)
    
    return jsonify({'dead_players': list(dead_players)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Forward prediction request to Render ML server"""
    try:
        import requests
        
        data = request.json
        
        # Call the ML server
        response = requests.post(
            f"{ML_SERVER}/api/predict",
            json=data,
            timeout=30
        )
        
        return response.json(), response.status_code
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
