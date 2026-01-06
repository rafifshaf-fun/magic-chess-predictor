from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
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
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file, index_col=0)
        
        # For each player, build tournament progression
        for player in all_players:
            if player not in df.columns:
                continue
            
            # Get all opponents this player faced in order
            opponents = df[player].tolist()
            opponents_faced = []
            
            for round_idx, (round_name, opponent) in enumerate(df.index, 1):
                if opponent in ["Creep", "Null"] or pd.isna(opponent):
                    continue
                
                # Opponent with "M." prefix is acceptable
                if opponent.startswith("NullM."):
                    continue
                
                # Skip if last opponent (can't predict next)
                if round_idx == len(df):
                    continue
                
                # Find next real opponent
                next_opponent = None
                for future_idx in range(round_idx, len(df)):
                    future_opp = df[player].iloc[future_idx]
                    if future_opp not in ["Creep", "Null"] and not pd.isna(future_opp) and not future_opp.startswith("NullM."):
                        next_opponent = future_opp
                        break
                
                if next_opponent:
                    training_data.append({
                        'player': player,
                        'round': round_name,
                        'last_opponent': opponent,
                        'opponents_faced_before': opponents_faced.copy(),
                        'next_opponent': next_opponent
                    })
                
                opponents_faced.append(opponent)
    
    # Convert to features
    X = []
    y = []
    
    for sample in training_data:
        player = sample['player']
        round_name = sample['round']
        last_opponent = sample['last_opponent']
        opponents_before = sample['opponents_faced_before']
        next_opponent = sample['next_opponent']
        
        # Feature 1-8: One-hot for player (8 dimensions)
        player_features = [1 if f"Player {i}" == player else 0 for i in range(1, 9)]
        
        # Feature 9-26: One-hot for last opponent (8 dimensions)
        opponent_features = [1 if f"Player {i}" == last_opponent else 0 for i in range(1, 9)]
        
        # Feature 27-34: One-hot for round (8 dimensions - rounded to I,II,III,IV,V)
        round_group = round_name.split('-')[0]
        round_features = [1 if round_group == rg else 0 for rg in ['I', 'II', 'III', 'IV', 'V']]
        
        # Feature 35: Have we faced any opponents before?
        faced_before = 1 if opponents_before else 0
        
        # Feature 36: Count of opponents faced before
        count_faced = len(opponents_before)
        
        # Combine all features
        features = player_features + opponent_features + round_features + [faced_before, count_faced]
        
        X.append(features)
        y.append(next_opponent)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Store label encoders for prediction
    model.classes_ = np.array(list(set(y)))
    model.training_data = training_data
    
    print(f"✓ Trained model with {len(training_data)} samples")
    print(f"✓ Feature dimensions: {len(X[0])}")
    
    return model

# Global model
ml_model = load_and_train_model()
dead_players = set()

@app.route('/api/session-stats', methods=['GET'])
def session_stats():
    """Get session stats including dead players"""
    return jsonify({
        'dead_players': list(dead_players),
        'session_logs': 0
    })

@app.route('/api/toggle-death', methods=['POST'])
def toggle_death():
    """Toggle player as dead/alive"""
    data = request.json
    player = data.get('player')
    
    if player in dead_players:
        dead_players.discard(player)
    else:
        dead_players.add(player)
    
    return jsonify({'dead_players': list(dead_players)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict next opponent with opponent history feature"""
    data = request.json
    player = data.get('player')
    round_name = data.get('round')
    last_opponent = data.get('last_opponent')
    opponents_faced_before = data.get('opponents_faced_before', [])
    dead_players_list = set(data.get('dead_players', []))
    
    try:
        # Build feature vector (same as training)
        player_features = [1 if f"Player {i}" == player else 0 for i in range(1, 9)]
        opponent_features = [1 if f"Player {i}" == last_opponent else 0 for i in range(1, 9)]
        
        round_group = round_name.split('-')[0]
        round_features = [1 if round_group == rg else 0 for rg in ['I', 'II', 'III', 'IV', 'V']]
        
        faced_before = 1 if opponents_faced_before else 0
        count_faced = len(opponents_faced_before)
        
        features = np.array([player_features + opponent_features + round_features + [faced_before, count_faced]])
        
        # Get probabilities for all classes
        probabilities = ml_model.predict_proba(features)[0]
        
        # Create prediction pairs
        predictions = list(zip(ml_model.classes_, probabilities))
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out dead players and already faced opponents
        all_faced = set(opponents_faced_before) | dead_players_list
        filtered = [(opp, conf) for opp, conf in predictions if opp not in all_faced]
        
        # Return top 3
        top3 = filtered[:3]
        
        if not top3:
            top3 = [(opp, conf) for opp, conf in predictions[:3]]
        
        return jsonify({
            'top3': [[opp, float(conf * 100)] for opp, conf in top3],
            'debug': {
                'total_candidates': len(predictions),
                'filtered_candidates': len(filtered)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    """Log user feedback (for future model improvements)"""
    data = request.json
    # Just acknowledge, store in memory if needed
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=False)
