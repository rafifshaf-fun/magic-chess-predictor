from flask import Flask, request, jsonify
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Complete training data from all 4 cleaned matches (376 samples)
training_data = [
    {"player": "Player 1", "current_round": "I-2", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 2", "current_round": "I-2", "last_opponent": "Player 4", "next_opponent": "Player 7"},
    {"player": "Player 3", "current_round": "I-2", "last_opponent": "Player 5", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "I-2", "last_opponent": "Player 2", "next_opponent": "Player 8"},
    {"player": "Player 5", "current_round": "I-2", "last_opponent": "Player 3", "next_opponent": "Player 6"},
    {"player": "Player 6", "current_round": "I-2", "last_opponent": "Player 1", "next_opponent": "Player 5"},
    {"player": "Player 7", "current_round": "I-2", "last_opponent": "Player 8", "next_opponent": "Player 2"},
    {"player": "Player 8", "current_round": "I-2", "last_opponent": "Player 7", "next_opponent": "Player 4"},
    {"player": "Player 1", "current_round": "I-3", "last_opponent": "Player 3", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "I-3", "last_opponent": "Player 7", "next_opponent": "Player 6"},
    {"player": "Player 3", "current_round": "I-3", "last_opponent": "Player 1", "next_opponent": "Player 8"},
    {"player": "Player 4", "current_round": "I-3", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "I-3", "last_opponent": "Player 6", "next_opponent": "Player 5"},
    {"player": "Player 6", "current_round": "I-3", "last_opponent": "Player 5", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "I-3", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "I-3", "last_opponent": "Player 4", "next_opponent": "Player 7"},
    {"player": "Player 1", "current_round": "I-4", "last_opponent": "Player 4", "next_opponent": "Player 7"},
    {"player": "Player 2", "current_round": "I-4", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 3", "current_round": "I-4", "last_opponent": "Player 7", "next_opponent": "Player 2"},
    {"player": "Player 4", "current_round": "I-4", "last_opponent": "Player 1", "next_opponent": "Player 5"},
    {"player": "Player 5", "current_round": "I-4", "last_opponent": "Player 8", "next_opponent": "Player 4"},
    {"player": "Player 6", "current_round": "I-4", "last_opponent": "Player 2", "next_opponent": "Player 8"},
    {"player": "Player 7", "current_round": "I-4", "last_opponent": "Player 3", "next_opponent": "Player 1"},
    {"player": "Player 8", "current_round": "I-4", "last_opponent": "Player 5", "next_opponent": "Player 6"},
    {"player": "Player 1", "current_round": "II-1", "last_opponent": "Player 7", "next_opponent": "Player 8"},
    {"player": "Player 2", "current_round": "II-1", "last_opponent": "Player 3", "next_opponent": "Player 1"},
    {"player": "Player 3", "current_round": "II-1", "last_opponent": "Player 2", "next_opponent": "Player 5"},
    {"player": "Player 4", "current_round": "II-1", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 5", "current_round": "II-1", "last_opponent": "Player 4", "next_opponent": "Player 8"},
    {"player": "Player 6", "current_round": "II-1", "last_opponent": "Player 8", "next_opponent": "Player 5"},
    {"player": "Player 7", "current_round": "II-1", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 8", "current_round": "II-1", "last_opponent": "Player 6", "next_opponent": "Player 2"},
    {"player": "Player 1", "current_round": "II-2", "last_opponent": "Player 8", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "II-2", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "II-2", "last_opponent": "Player 4", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "II-2", "last_opponent": "Player 3", "next_opponent": "Player 7"},
    {"player": "Player 5", "current_round": "II-2", "last_opponent": "Player 2", "next_opponent": "Player 4"},
    {"player": "Player 6", "current_round": "II-2", "last_opponent": "Player 7", "next_opponent": "Player 1"},
    {"player": "Player 7", "current_round": "II-2", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "II-2", "last_opponent": "Player 1", "next_opponent": "Player 2"},
    {"player": "Player 1", "current_round": "II-4", "last_opponent": "Player 5", "next_opponent": "Player 2"},
    {"player": "Player 2", "current_round": "II-4", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 3", "current_round": "II-4", "last_opponent": "Player 6", "next_opponent": "Player 8"},
    {"player": "Player 4", "current_round": "II-4", "last_opponent": "Player 7", "next_opponent": "Player 3"},
    {"player": "Player 5", "current_round": "II-4", "last_opponent": "Player 1", "next_opponent": "Player 7"},
    {"player": "Player 6", "current_round": "II-4", "last_opponent": "Player 3", "next_opponent": "Player 5"},
    {"player": "Player 7", "current_round": "II-4", "last_opponent": "Player 4", "next_opponent": "Player 6"},
    {"player": "Player 8", "current_round": "II-4", "last_opponent": "Player 2", "next_opponent": "Player 4"},
    {"player": "Player 1", "current_round": "II-5", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 2", "current_round": "II-5", "last_opponent": "Player 1", "next_opponent": "Player 6"},
    {"player": "Player 3", "current_round": "II-5", "last_opponent": "Player 8", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "II-5", "last_opponent": "Player 6", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "II-5", "last_opponent": "Player 7", "next_opponent": "Player 3"},
    {"player": "Player 6", "current_round": "II-5", "last_opponent": "Player 4", "next_opponent": "Player 8"},
    {"player": "Player 7", "current_round": "II-5", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 8", "current_round": "II-5", "last_opponent": "Player 3", "next_opponent": "Player 7"},
    {"player": "Player 1", "current_round": "II-6", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 2", "current_round": "II-6", "last_opponent": "Player 4", "next_opponent": "Player 7"},
    {"player": "Player 3", "current_round": "II-6", "last_opponent": "Player 5", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "II-6", "last_opponent": "Player 2", "next_opponent": "Player 8"},
    {"player": "Player 5", "current_round": "II-6", "last_opponent": "Player 3", "next_opponent": "Player 6"},
    {"player": "Player 6", "current_round": "II-6", "last_opponent": "Player 1", "next_opponent": "Player 5"},
    {"player": "Player 7", "current_round": "II-6", "last_opponent": "Player 8", "next_opponent": "Player 2"},
    {"player": "Player 8", "current_round": "II-6", "last_opponent": "Player 7", "next_opponent": "Player 4"},
    {"player": "Player 1", "current_round": "III-1", "last_opponent": "Player 3", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "III-1", "last_opponent": "Player 7", "next_opponent": "Player 6"},
    {"player": "Player 3", "current_round": "III-1", "last_opponent": "Player 1", "next_opponent": "Player 8"},
    {"player": "Player 4", "current_round": "III-1", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "III-1", "last_opponent": "Player 6", "next_opponent": "Player 5"},
    {"player": "Player 6", "current_round": "III-1", "last_opponent": "Player 5", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "III-1", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "III-1", "last_opponent": "Player 4", "next_opponent": "Player 7"},
    {"player": "Player 1", "current_round": "III-2", "last_opponent": "Player 4", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "III-2", "last_opponent": "Player 6", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "III-2", "last_opponent": "Player 7", "next_opponent": "Player 2"},
    {"player": "Player 4", "current_round": "III-2", "last_opponent": "Player 1", "next_opponent": "Player 3"},
    {"player": "Player 5", "current_round": "III-2", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 6", "current_round": "III-2", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 7", "current_round": "III-2", "last_opponent": "Player 3", "next_opponent": "Player 6"},
    {"player": "Player 8", "current_round": "III-2", "last_opponent": "Player 5", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "III-4", "last_opponent": "Player 3", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "III-4", "last_opponent": "Player 2", "next_opponent": "Player 5"},
    {"player": "Player 4", "current_round": "III-4", "last_opponent": "Player 5", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "III-4", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 8", "current_round": "III-4", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 2", "current_round": "III-5", "last_opponent": "Player 5", "next_opponent": "Player 4"},
    {"player": "Player 3", "current_round": "III-5", "last_opponent": "Player 4", "next_opponent": "Player 8"},
    {"player": "Player 4", "current_round": "III-5", "last_opponent": "Player 3", "next_opponent": "Player 2"},
    {"player": "Player 5", "current_round": "III-5", "last_opponent": "Player 2", "next_opponent": "Player 1"},
    {"player": "Player 8", "current_round": "III-5", "last_opponent": "Player 1", "next_opponent": "Player 3"},
    {"player": "Player 2", "current_round": "III-6", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 3", "current_round": "III-6", "last_opponent": "Player 8", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "III-6", "last_opponent": "Player 2", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "III-6", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 6", "current_round": "III-6", "last_opponent": "Player 5", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "III-6", "last_opponent": "Player 3", "next_opponent": "Player 1"},
    {"player": "Player 1", "current_round": "IV-1", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 2", "current_round": "IV-1", "last_opponent": "Player 3", "next_opponent": "Player 1"},
    {"player": "Player 3", "current_round": "IV-1", "last_opponent": "Player 2", "next_opponent": "Player 4"},
    {"player": "Player 4", "current_round": "IV-1", "last_opponent": "Player 1", "next_opponent": "Player 3"},
    {"player": "Player 1", "current_round": "IV-2", "last_opponent": "Player 2", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "IV-2", "last_opponent": "Player 1", "next_opponent": "Player 3"},
    {"player": "Player 3", "current_round": "IV-2", "last_opponent": "Player 4", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "IV-2", "last_opponent": "Player 3", "next_opponent": "Player 2"},
    {"player": "Player 1", "current_round": "IV-4", "last_opponent": "Player 3", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "IV-4", "last_opponent": "Player 4", "next_opponent": "Player 1"},
    {"player": "Player 3", "current_round": "IV-4", "last_opponent": "Player 1", "next_opponent": "Player 2"},
    {"player": "Player 4", "current_round": "IV-4", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 1", "current_round": "IV-5", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 2", "current_round": "IV-5", "last_opponent": "Player 3", "next_opponent": "Player 4"},
    {"player": "Player 3", "current_round": "IV-5", "last_opponent": "Player 2", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "IV-5", "last_opponent": "Player 1", "next_opponent": "Player 2"},
    {"player": "Player 1", "current_round": "IV-6", "last_opponent": "Player 3", "next_opponent": "Player 3"},
    {"player": "Player 3", "current_round": "IV-6", "last_opponent": "Player 1", "next_opponent": "Player 1"},
    {"player": "Player 1", "current_round": "V-1", "last_opponent": "Player 3", "next_opponent": "Player 3"},
    {"player": "Player 3", "current_round": "V-1", "last_opponent": "Player 1", "next_opponent": "Player 1"},
    {"player": "Player 1", "current_round": "V-2", "last_opponent": "Player 3", "next_opponent": "Player 3"},
    {"player": "Player 3", "current_round": "V-2", "last_opponent": "Player 1", "next_opponent": "Player 1"},
    {"player": "Player 1", "current_round": "I-2", "last_opponent": "Player 3", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "I-2", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 3", "current_round": "I-2", "last_opponent": "Player 1", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "I-2", "last_opponent": "Player 6", "next_opponent": "Player 8"},
    {"player": "Player 5", "current_round": "I-2", "last_opponent": "Player 8", "next_opponent": "Player 4"},
    {"player": "Player 6", "current_round": "I-2", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "I-2", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "I-2", "last_opponent": "Player 5", "next_opponent": "Player 7"},
    {"player": "Player 1", "current_round": "I-3", "last_opponent": "Player 4", "next_opponent": "Player 6"},
    {"player": "Player 2", "current_round": "I-3", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "I-3", "last_opponent": "Player 6", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "I-3", "last_opponent": "Player 1", "next_opponent": "Player 5"},
    {"player": "Player 5", "current_round": "I-3", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 6", "current_round": "I-3", "last_opponent": "Player 3", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "I-3", "last_opponent": "Player 8", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "I-3", "last_opponent": "Player 7", "next_opponent": "Player 4"},
    {"player": "Player 1", "current_round": "I-4", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "I-4", "last_opponent": "Player 3", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "I-4", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 4", "current_round": "I-4", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "I-4", "last_opponent": "Player 6", "next_opponent": "Player 2"},
    {"player": "Player 6", "current_round": "I-4", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 7", "current_round": "I-4", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 8", "current_round": "I-4", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 1", "current_round": "II-1", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 2", "current_round": "II-1", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 3", "current_round": "II-1", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 4", "current_round": "II-1", "last_opponent": "Player 3", "next_opponent": "Player 5"},
    {"player": "Player 5", "current_round": "II-1", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 6", "current_round": "II-1", "last_opponent": "Player 7", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "II-1", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "II-1", "last_opponent": "Player 2", "next_opponent": "Player 6"},
    {"player": "Player 1", "current_round": "II-2", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 2", "current_round": "II-2", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 3", "current_round": "II-2", "last_opponent": "Player 7", "next_opponent": "Player 8"},
    {"player": "Player 4", "current_round": "II-2", "last_opponent": "Player 5", "next_opponent": "Player 3"},
    {"player": "Player 5", "current_round": "II-2", "last_opponent": "Player 4", "next_opponent": "Player 6"},
    {"player": "Player 6", "current_round": "II-2", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 7", "current_round": "II-2", "last_opponent": "Player 3", "next_opponent": "Player 5"},
    {"player": "Player 8", "current_round": "II-2", "last_opponent": "Player 6", "next_opponent": "Player 2"},
    {"player": "Player 1", "current_round": "II-4", "last_opponent": "Player 6", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "II-4", "last_opponent": "Player 4", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "II-4", "last_opponent": "Player 8", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "II-4", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 5", "current_round": "II-4", "last_opponent": "Player 7", "next_opponent": "Player 1"},
    {"player": "Player 6", "current_round": "II-4", "last_opponent": "Player 1", "next_opponent": "Player 3"},
    {"player": "Player 7", "current_round": "II-4", "last_opponent": "Player 5", "next_opponent": "Player 4"},
    {"player": "Player 8", "current_round": "II-4", "last_opponent": "Player 3", "next_opponent": "Player 2"},
    {"player": "Player 1", "current_round": "II-5", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 2", "current_round": "II-5", "last_opponent": "Player 6", "next_opponent": "Player 4"},
    {"player": "Player 3", "current_round": "II-5", "last_opponent": "Player 5", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "II-5", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 5", "current_round": "II-5", "last_opponent": "Player 3", "next_opponent": "Player 8"},
    {"player": "Player 6", "current_round": "II-5", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 7", "current_round": "II-5", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 8", "current_round": "II-5", "last_opponent": "Player 1", "next_opponent": "Player 6"},
    {"player": "Player 1", "current_round": "II-6", "last_opponent": "Player 3", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "II-6", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 3", "current_round": "II-6", "last_opponent": "Player 1", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "II-6", "last_opponent": "Player 6", "next_opponent": "Player 8"},
    {"player": "Player 5", "current_round": "II-6", "last_opponent": "Player 8", "next_opponent": "Player 4"},
    {"player": "Player 6", "current_round": "II-6", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "II-6", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "II-6", "last_opponent": "Player 5", "next_opponent": "Player 7"},
    {"player": "Player 1", "current_round": "III-1", "last_opponent": "Player 4", "next_opponent": "Player 6"},
    {"player": "Player 2", "current_round": "III-1", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "III-1", "last_opponent": "Player 6", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "III-1", "last_opponent": "Player 1", "next_opponent": "Player 5"},
    {"player": "Player 5", "current_round": "III-1", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 6", "current_round": "III-1", "last_opponent": "Player 3", "next_opponent": "Player 2"},
    {"player": "Player 7", "current_round": "III-1", "last_opponent": "Player 8", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "III-1", "last_opponent": "Player 7", "next_opponent": "Player 4"},
    {"player": "Player 1", "current_round": "III-2", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "III-2", "last_opponent": "Player 3", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "III-2", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 4", "current_round": "III-2", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 5", "current_round": "III-2", "last_opponent": "Player 6", "next_opponent": "Player 2"},
    {"player": "Player 6", "current_round": "III-2", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 7", "current_round": "III-2", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 8", "current_round": "III-2", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 1", "current_round": "III-4", "last_opponent": "Player 4", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "III-4", "last_opponent": "Player 8", "next_opponent": "Player 1"},
    {"player": "Player 3", "current_round": "III-4", "last_opponent": "Player 4", "next_opponent": "Player 2"},
    {"player": "Player 4", "current_round": "III-4", "last_opponent": "Player 3", "next_opponent": "Player 6"},
    {"player": "Player 5", "current_round": "III-4", "last_opponent": "Player 1", "next_opponent": "Player 3"},
    {"player": "Player 6", "current_round": "III-4", "last_opponent": "Player 5", "next_opponent": "Player 4"},
    {"player": "Player 8", "current_round": "III-4", "last_opponent": "Player 2", "next_opponent": "Player 8"},
    {"player": "Player 1", "current_round": "III-5", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 2", "current_round": "III-5", "last_opponent": "Player 1", "next_opponent": "Player 7"},
    {"player": "Player 3", "current_round": "III-5", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 4", "current_round": "III-5", "last_opponent": "Player 5", "next_opponent": "Player 2"},
    {"player": "Player 5", "current_round": "III-5", "last_opponent": "Player 4", "next_opponent": "Player 1"},
    {"player": "Player 6", "current_round": "III-5", "last_opponent": "Player 8", "next_opponent": "Player 7"},
    {"player": "Player 7", "current_round": "III-5", "last_opponent": "Player 2", "next_opponent": "Player 4"},
    {"player": "Player 8", "current_round": "III-5", "last_opponent": "Player 6", "next_opponent": "Player 3"},
    {"player": "Player 1", "current_round": "III-6", "last_opponent": "Player 8", "next_opponent": "Player 2"},
    {"player": "Player 2", "current_round": "III-6", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 3", "current_round": "III-6", "last_opponent": "Player 5", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "III-6", "last_opponent": "Player 6", "next_opponent": "Player 8"},
    {"player": "Player 5", "current_round": "III-6", "last_opponent": "Player 3", "next_opponent": "Player 7"},
    {"player": "Player 6", "current_round": "III-6", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 7", "current_round": "III-6", "last_opponent": "Player 2", "next_opponent": "Player 5"},
    {"player": "Player 8", "current_round": "III-6", "last_opponent": "Player 7", "next_opponent": "Player 1"},
    {"player": "Player 2", "current_round": "IV-1", "last_opponent": "Player 8", "next_opponent": "Player 4"},
    {"player": "Player 3", "current_round": "IV-1", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 4", "current_round": "IV-1", "last_opponent": "Player 3", "next_opponent": "Player 6"},
    {"player": "Player 5", "current_round": "IV-1", "last_opponent": "Player 6", "next_opponent": "Player 5"},
    {"player": "Player 6", "current_round": "IV-1", "last_opponent": "Player 5", "next_opponent": "Player 8"},
    {"player": "Player 7", "current_round": "IV-1", "last_opponent": "Player 8", "next_opponent": "Player 7"},
    {"player": "Player 8", "current_round": "IV-1", "last_opponent": "Player 7", "next_opponent": "Player 2"},
    {"player": "Player 2", "current_round": "IV-2", "last_opponent": "Player 4", "next_opponent": "Player 8"},
    {"player": "Player 3", "current_round": "IV-2", "last_opponent": "Player 8", "next_opponent": "Player 2"},
    {"player": "Player 4", "current_round": "IV-2", "last_opponent": "Player 2", "next_opponent": "Player 7"},
    {"player": "Player 5", "current_round": "IV-2", "last_opponent": "Player 7", "next_opponent": "Player 1"},
    {"player": "Player 6", "current_round": "IV-2", "last_opponent": "Player 1", "next_opponent": "Player 4"},
    {"player": "Player 7", "current_round": "IV-2", "last_opponent": "Player 5", "next_opponent": "Player 2"},
    {"player": "Player 8", "current_round": "IV-2", "last_opponent": "Player 3", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "IV-4", "last_opponent": "Player 8", "next_opponent": "Player 7"},
    {"player": "Player 3", "current_round": "IV-4", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 4", "current_round": "IV-4", "last_opponent": "Player 5", "next_opponent": "Player 4"},
    {"player": "Player 5", "current_round": "IV-4", "last_opponent": "Player 4", "next_opponent": "Player 3"},
    {"player": "Player 7", "current_round": "IV-4", "last_opponent": "Player 2", "next_opponent": "Player 3"},
    {"player": "Player 8", "current_round": "IV-4", "last_opponent": "Player 3", "next_opponent": "Player 2"},
    {"player": "Player 2", "current_round": "IV-5", "last_opponent": "Player 5", "next_opponent": "Player 6"},
    {"player": "Player 3", "current_round": "IV-5", "last_opponent": "Player 1", "next_opponent": "Player 7"},
    {"player": "Player 4", "current_round": "IV-5", "last_opponent": "Player 7", "next_opponent": "Player 2"},
    {"player": "Player 6", "current_round": "IV-5", "last_opponent": "Player 2", "next_opponent": "Player 1"},
    {"player": "Player 7", "current_round": "IV-5", "last_opponent": "Player 6", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "IV-6", "last_opponent": "Player 1", "next_opponent": "Player 6"},
    {"player": "Player 4", "current_round": "IV-6", "last_opponent": "Player 7", "next_opponent": "Player 5"},
    {"player": "Player 6", "current_round": "IV-6", "last_opponent": "Player 2", "next_opponent": "Player 1"},
    {"player": "Player 7", "current_round": "IV-6", "last_opponent": "Player 4", "next_opponent": "Player 5"},
    {"player": "Player 2", "current_round": "V-1", "last_opponent": "Player 6", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "V-1", "last_opponent": "Player 5", "next_opponent": "Player 7"},
    {"player": "Player 6", "current_round": "V-1", "last_opponent": "Player 1", "next_opponent": "Player 1"},
    {"player": "Player 7", "current_round": "V-1", "last_opponent": "Player 4", "next_opponent": "Player 4"},
    {"player": "Player 2", "current_round": "V-2", "last_opponent": "Player 6", "next_opponent": "Player 1"},
    {"player": "Player 4", "current_round": "V-2", "last_opponent": "Player 5", "next_opponent": "Player 7"},
    {"player": "Player 6", "current_round": "V-2", "last_opponent": "Player 1", "next_opponent": "Player 1"},
    {"player": "Player 7", "current_round": "V-2", "last_opponent": "Player 4", "next_opponent": "Player 4"},
]

# Global model and encoders
model = None
le_player = LabelEncoder()
le_round = LabelEncoder()
le_last_opp = LabelEncoder()
le_next_opp = LabelEncoder()
session_matches = []

def initialize_model():
    global model, le_player, le_round, le_last_opp, le_next_opp
    
    # Get all unique values
    players = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6", "Player 7", "Player 8"]
    rounds = ["I-1", "I-2", "I-3", "I-4", "II-1", "II-2", "II-3", "II-4", "II-5", "II-6", 
              "III-1", "III-2", "III-3", "III-4", "III-5", "III-6", "IV-1", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6",
              "V-1", "V-2", "V-3", "V-4"]
    
    le_player.fit(players)
    le_round.fit(rounds)
    le_last_opp.fit(players)
    le_next_opp.fit(players)
    
    # Prepare training data
    X = []
    y = []
    
    for sample in training_data:
        X.append([
            le_player.transform([sample["player"]])[0],
            le_round.transform([sample["current_round"]])[0],
            le_last_opp.transform([sample["last_opponent"]])[0],
            1  # unique_opponents_faced - simplified for now
        ])
        y.append(le_next_opp.transform([sample["next_opponent"]])[0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

def retrain_model():
    global model, le_player, le_round, le_last_opp, le_next_opp
    
    # Combine original data with session matches
    all_data = training_data.copy()
    
    for match in session_matches:
        all_data.append({
            "player": match["player"],
            "current_round": match["round"],
            "last_opponent": match["previous_opponent"],
            "next_opponent": match["actual_opponent"]
        })
    
    # Retrain
    X = []
    y = []
    
    for sample in all_data:
        try:
            X.append([
                le_player.transform([sample["player"]])[0],
                le_round.transform([sample["current_round"]])[0],
                le_last_opp.transform([sample["last_opponent"]])[0],
                1
            ])
            y.append(le_next_opp.transform([sample["next_opponent"]])[0])
        except:
            pass
    
    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    player = data.get('player')
    round_name = data.get('round')
    last_opponent = data.get('last_opponent')
    
    if not all([player, round_name, last_opponent]):
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        x = np.array([[
            le_player.transform([player])[0],
            le_round.transform([round_name])[0],
            le_last_opp.transform([last_opponent])[0],
            1
        ]])
        
        probs = model.predict_proba(x)[0]
        pred_idx = probs.argmax()
        pred_opp = le_next_opp.inverse_transform([pred_idx])[0]
        
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = list(zip(
            le_next_opp.inverse_transform(top3_idx),
            probs[top3_idx] * 100
        ))
        
        return jsonify({
            "predicted": pred_opp,
            "confidence": float(probs[pred_idx] * 100),
            "top3": top3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/log-result', methods=['POST'])
def log_result():
    global session_matches
    data = request.json
    
    player = data.get('player')
    round_name = data.get('round')
    actual_opponent = data.get('actual_opponent')
    prev_opponent = data.get('previous_opponent')
    
    if not all([player, round_name, actual_opponent, prev_opponent]):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Store match
    session_matches.append({
        "player": player,
        "round": round_name,
        "actual_opponent": actual_opponent,
        "previous_opponent": prev_opponent,
        "timestamp": str(np.datetime64('now'))
    })
    
    # Retrain model
    retrain_model()
    
    return jsonify({
        "message": "Result logged and model retrained",
        "total_matches": len(session_matches),
        "total_training_samples": len(training_data) + len(session_matches)
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        "total_matches_logged": len(session_matches),
        "total_training_samples": len(training_data) + len(session_matches),
        "model_status": "ready",
        "matches": session_matches
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "ready", "training_samples": len(training_data)})

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)