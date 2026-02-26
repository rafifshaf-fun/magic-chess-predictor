# Replit Backend - Flask API for Magic Chess Go Go Opponent Prediction
# Deploy on Replit: https://replit.com

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==================== DATA LOADING ====================

def load_training_data():
    """Load and process all match CSV files"""
    matches = []
    match_files = [
        "Match-1.csv", "Match-2.csv", "Match-3.csv", "Match-4.csv",
        "Match-5.csv", "Match-6.csv", "Match-7.csv", "Match-8.csv", "Match-9.csv"
    ]
    
    for file in match_files:
        try:
            df = pd.read_csv(file)
            matches.append(df)
        except FileNotFoundError:
            print(f"Warning: {file} not found")
    
    return matches

def build_transition_model(matches):
    """
    Build a transition probability model
    For each (player, round_number, last_opponent) -> predict next_opponent
    """
    model = defaultdict(lambda: Counter())
    
    for match_df in matches:
        players = match_df.columns[1:]
        
        for player in players:
            opponents = match_df[player].tolist()
            
            for round_idx in range(len(opponents) - 1):
                current_opponent = opponents[round_idx]
                next_opponent = opponents[round_idx + 1]
                
                # Skip Creep (warmup) and Null (eliminated) and matches
                if current_opponent in ['Creep', 'Null'] or next_opponent in ['Creep', 'Null']:
                    continue
                if 'M.Player' in str(current_opponent):
                    continue
                if 'M.Player' in str(next_opponent):
                    next_opponent = next_opponent.replace('M.', '')
                
                # Create feature key: (player, previous_opponent) -> next_opponent
                key = (player, current_opponent)
                model[key][next_opponent] += 1
    
    return model

def build_position_model(matches):
    """
    Build positional model: which opponents appear at which round positions
    """
    position_model = defaultdict(lambda: Counter())
    
    for match_df in matches:
        players = match_df.columns[1:]
        
        for player in players:
            opponents = match_df[player].tolist()
            round_names = match_df.iloc[:, 0].tolist()
            
            for round_pos, opponent in enumerate(opponents):
                if opponent in ['Creep', 'Null']:
                    continue
                if 'M.Player' in str(opponent):
                    opponent = opponent.replace('M.', '')
                
                # Key: (player, round_position) -> opponent
                key = (player, round_pos)
                position_model[key][opponent] += 1
    
    return position_model

# ==================== PREDICTION ENGINE ====================

def predict_next_opponent(player, current_round_idx, last_opponent, 
                         transition_model, position_model, all_players):
    """
    Predict next 3 most likely opponents
    Uses two strategies: transition-based and position-based
    """
    scores = Counter()
    
    # Strategy 1: Transition-based (what comes after this opponent for this player)
    transition_key = (player, last_opponent)
    if transition_key in transition_model:
        for opponent, count in transition_model[transition_key].items():
            scores[opponent] += count * 2  # Higher weight for transitions
    
    # Strategy 2: Position-based (what appears at this round position)
    next_round_idx = current_round_idx + 1
    position_key = (player, next_round_idx)
    if position_key in position_model:
        for opponent, count in position_model[position_key].items():
            if opponent != player:  # Can't play yourself
                scores[opponent] += count
    
    # Fallback: Use general frequency if no specific pattern
    if not scores:
        for p in all_players:
            if p != player:
                scores[p] = 1
    
    # Get top 3
    top_3 = scores.most_common(3)
    
    # Calculate percentages
    total = sum([count for _, count in top_3])
    predictions = [
        {
            'opponent': opponent,
            'probability': round((count / total) * 100, 1)
        }
        for opponent, count in top_3
    ]
    
    return predictions

def extract_round_number(round_str):
    """Extract numeric position from round string like 'I-2', 'II-4', etc."""
    parts = round_str.split('-')
    if len(parts) == 2:
        try:
            return int(parts[1]) - 1  # 0-indexed
        except:
            return 0
    return 0

# ==================== API ROUTES ====================

# Load model on startup
try:
    matches = load_training_data()
    transition_model = build_transition_model(matches)
    position_model = build_position_model(matches)
    all_players = [f'Player {i}' for i in range(1, 9)]
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    transition_model = {}
    position_model = {}
    all_players = [f'Player {i}' for i in range(1, 9)]

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get list of available players"""
    return jsonify({'players': all_players})

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict next opponent
    
    Request body:
    {
        'player': 'Player 1',
        'current_round': 'I-2',
        'last_opponent': 'Player 3'
    }
    """
    try:
        data = request.json
        player = data.get('player')
        current_round = data.get('current_round', 'I-1')
        last_opponent = data.get('last_opponent')
        
        if not player or not last_opponent:
            return jsonify({'error': 'Missing player or last_opponent'}), 400
        
        round_idx = extract_round_number(current_round)
        
        predictions = predict_next_opponent(
            player, round_idx, last_opponent,
            transition_model, position_model, all_players
        )
        
        return jsonify({
            'success': True,
            'player': player,
            'current_round': current_round,
            'last_opponent': last_opponent,
            'next_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple states in sequence
    
    Request body:
    {
        'player': 'Player 1',
        'history': [
            {'round': 'I-1', 'opponent': 'Player 3'},
            {'round': 'I-2', 'opponent': 'Player 8'}
        ]
    }
    """
    try:
        data = request.json
        player = data.get('player')
        history = data.get('history', [])
        
        if not player:
            return jsonify({'error': 'Missing player'}), 400
        
        results = []
        for item in history:
            round_str = item.get('round', 'I-1')
            opponent = item.get('opponent')
            
            if opponent:
                round_idx = extract_round_number(round_str)
                predictions = predict_next_opponent(
                    player, round_idx, opponent,
                    transition_model, position_model, all_players
                )
                
                results.append({
                    'round': round_str,
                    'opponent': opponent,
                    'predictions': predictions
                })
        
        return jsonify({
            'success': True,
            'player': player,
            'predictions_history': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        'name': 'Magic Chess Go Go Opponent Predictor',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/health': 'Health check',
            'GET /api/players': 'Get list of players',
            'POST /api/predict': 'Predict next opponent (single)',
            'POST /api/predict-batch': 'Predict next opponents (sequence)',
            'GET /': 'This documentation'
        },
        'example_request': {
            'url': '/api/predict',
            'method': 'POST',
            'body': {
                'player': 'Player 1',
                'current_round': 'I-2',
                'last_opponent': 'Player 3'
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
