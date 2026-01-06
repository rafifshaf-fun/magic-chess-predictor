from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Load all match data and train model once
def load_and_train_model():
    """Load CSV files and train Random Forest with opponent history feature"""
    
    training_data = []
    all_players = [f"Player {i}" for i in range(1, 9)]
    
    # Load all match CSV files
    for match_num in range(1, 10):
        csv_file = f"Match-{match_num}.csv"
        try:
            if not os.path.exists(csv_file):
                continue
                
            df = pd.read_csv(csv_file, index_col=0)
            
            # For each player, build tournament progression
            for player in all_players:
                if player not in df.columns:
                    continue
                
                opponents_faced = []
                
                # Get all opponents for this player
                for round_idx in range(len(df)):
                    round_name = df.index[round_idx]
                    opponent = df[player].iloc[round_idx]
                    
                    # Skip invalid entries
                    if pd.isna(opponent):
                        continue
                    if opponent in ["Creep", "Null"]:
                        continue
                    if str(opponent).startswith("Null"):
                        continue
                    
                    # Find next real opponent
                    next_opponent = None
                    for future_idx in range(round_idx + 1, len(df)):
                        future_opp = df[player].iloc[future_idx]
                        if pd.isna(future_opp):
                            continue
                        if str(future_opp) in ["Creep", "Null"]:
                            continue
                        if str(future_opp).startswith("Null"):
                            continue
                        next_opponent = str(future_opp).strip()
                        break
                    
                    if next_opponent:
                        training_data.append({
                            'player': player,
                            'round': round_name,
                            'last_opponent': str(opponent).strip(),
                            'opponents_faced_before': [str(o).strip() for o in opponents_faced],
                            'next_opponent': next_opponent
                        })
                    
                    opponents_faced.append(str(opponent).strip())
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    # Convert to features
    X = []
    y = []
    
    for sample in training_data:
        player = sample['player']
        round_name = sample['round']
        last_opponent = sample['last_opponent']
        next_opponent = sample['next_opponent']
        
        # Features: 8 (player) + 8 (last opponent) + 5 (round) + 2 (opponent history) = 23
        player_features = [1 if f"Player {i}" == player else 0 for i in range(1, 9)]
        opponent_features = [1 if f"Player {i}" == last_opponent else 0 for i in range(1, 9)]
        
        round_group = round_name.split('-')[0] if '-' in str(round_name) else 'I'
        round_features = [1 if round_group == rg else 0 for rg in ['I', 'II', 'III', 'IV', 'V']]
        
        faced_before = 1 if sample['opponents_faced_before'] else 0
        count_faced = len(sample['opponents_faced_before'])
        
        features = player_features + opponent_features + round_features + [faced_before, count_faced]
        X.append(features)
        y.append(next_opponent)
    
    # Train model
    try:
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        model.fit(X, y)
        model.classes_ = np.array(list(set(y)))
        print(f"✓ Model trained with {len(training_data)} samples")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Global model
ml_model = load_and_train_model()
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
    try:
        data = request.json
        player = data.get('player', '')
        round_name = data.get('round', '')
        last_opponent = data.get('last_opponent', '')
        opponents_faced_before = data.get('opponents_faced_before', [])
        dead_players_list = set(data.get('dead_players', []))
        
        if not ml_model:
            return jsonify({'error': 'Model not trained'}), 500
        
        # Build features
        player_features = [1 if f"Player {i}" == player else 0 for i in range(1, 9)]
        opponent_features = [1 if f"Player {i}" == last_opponent else 0 for i in range(1, 9)]
        
        round_group = round_name.split('-')[0] if '-' in str(round_name) else 'I'
        round_features = [1 if round_group == rg else 0 for rg in ['I', 'II', 'III', 'IV', 'V']]
        
        faced_before = 1 if opponents_faced_before else 0
        count_faced = len(opponents_faced_before)
        
        features = np.array([player_features + opponent_features + round_features + [faced_before, count_faced]])
        
        # Get predictions
        probabilities = ml_model.predict_proba(features)[0]
        predictions = list(zip(ml_model.classes_, probabilities))
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Filter dead and already faced
        all_faced = set(opponents_faced_before) | dead_players_list
        filtered = [(opp, conf) for opp, conf in predictions if opp not in all_faced]
        
        top3 = filtered[:3]
        if not top3:
            top3 = [(opp, conf) for opp, conf in predictions[:3]]
        
        return jsonify({
            'top3': [[str(opp), float(conf * 100)] for opp, conf in top3]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False)
