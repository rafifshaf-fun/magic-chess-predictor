import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

# Complete training data from all 9 matches (895 samples - cleaned, no filtering applied)
training_data = [
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "M.Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "M.Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "IV-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "IV-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 2",
        "current_round": "IV-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "IV-5",
        "last_opponent": "Player 2",
        "next_opponent": "M.Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "IV-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "IV-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "V-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "V-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "M.Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "IV-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "IV-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "V-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "V-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "M.Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "M.Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "IV-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "IV-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "IV-4",
        "last_opponent": "M.Player 3",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "M.Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "M.Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "M.Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "M.Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "III-6",
        "last_opponent": "M.Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "M.Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "IV-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "IV-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "IV-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "IV-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "IV-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-5",
        "last_opponent": "Player 7",
        "next_opponent": "M.Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "IV-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "IV-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "IV-6",
        "last_opponent": "M.Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "IV-6",
        "last_opponent": "Player 7",
        "next_opponent": "M.Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "IV-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "V-1",
        "last_opponent": "Player 7",
        "next_opponent": "M.Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "V-1",
        "last_opponent": "M.Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "V-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "III-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "III-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "Player 1",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "IV-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "IV-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "IV-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "IV-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "IV-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "IV-5",
        "last_opponent": "Player 2",
        "next_opponent": "M.Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "IV-6",
        "last_opponent": "M.Player 1",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "V-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "V-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "M.Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "IV-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "IV-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "IV-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "III-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "M.Player 7",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "IV-1",
        "last_opponent": "M.Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "IV-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "IV-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "IV-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "IV-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "IV-4",
        "last_opponent": "Player 1",
        "next_opponent": "M.Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "IV-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "IV-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "IV-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "III-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "III-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "III-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "IV-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "IV-1",
        "last_opponent": "Player 6",
        "next_opponent": "M.Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "IV-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "M.Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "IV-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "IV-5",
        "last_opponent": "Player 4",
        "next_opponent": "M.Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "IV-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "IV-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "IV-6",
        "last_opponent": "M.Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "IV-6",
        "last_opponent": "Player 4",
        "next_opponent": "M.Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "V-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "V-1",
        "last_opponent": "M.Player 2",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 1",
        "current_round": "I-2",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "I-2",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "I-2",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "I-2",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "I-2",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "I-2",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "I-2",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "I-2",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "I-3",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "I-3",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "I-3",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "I-3",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "I-3",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "I-3",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "I-3",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "I-3",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "I-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 2",
        "current_round": "I-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 3",
        "current_round": "I-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "I-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 5",
        "current_round": "I-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "I-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 7",
        "current_round": "I-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 8",
        "current_round": "I-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 1",
        "current_round": "II-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "II-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "II-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "II-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "II-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "II-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "II-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "II-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "II-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 2",
        "current_round": "II-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "II-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "II-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "II-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "II-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "II-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 8",
        "current_round": "II-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 1",
        "current_round": "II-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 2",
        "current_round": "II-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "II-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "II-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 5",
        "current_round": "II-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "II-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 7",
        "current_round": "II-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 8",
        "current_round": "II-5",
        "last_opponent": "Player 1",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "II-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "II-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "II-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 4",
        "current_round": "II-6",
        "last_opponent": "Player 3",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 5",
        "current_round": "II-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 6",
        "current_round": "II-6",
        "last_opponent": "Player 8",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 7",
        "current_round": "II-6",
        "last_opponent": "Player 2",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 8",
        "current_round": "II-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 1",
        "current_round": "III-1",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "III-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "III-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "III-1",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 5",
        "current_round": "III-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 6",
        "current_round": "III-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "III-1",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 8",
        "current_round": "III-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "III-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "III-4",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 3",
        "current_round": "III-4",
        "last_opponent": "Player 7",
        "next_opponent": "Player 8"
    },
    {
        "player": "Player 4",
        "current_round": "III-4",
        "last_opponent": "Player 8",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 5",
        "current_round": "III-4",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 6",
        "current_round": "III-4",
        "last_opponent": "Player 1",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 7",
        "current_round": "III-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 8",
        "current_round": "III-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 1",
        "current_round": "III-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 2",
        "current_round": "III-5",
        "last_opponent": "Player 1",
        "next_opponent": "M.Player 6"
    },
    {
        "player": "Player 3",
        "current_round": "III-5",
        "last_opponent": "Player 8",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 4",
        "current_round": "III-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 5",
        "current_round": "III-5",
        "last_opponent": "Player 6",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "III-5",
        "last_opponent": "Player 5",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 7",
        "current_round": "III-5",
        "last_opponent": "Player 4",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 1",
        "current_round": "III-6",
        "last_opponent": "Player 3",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 2",
        "current_round": "III-6",
        "last_opponent": "M.Player 6",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "III-6",
        "last_opponent": "Player 1",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 4",
        "current_round": "III-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 5"
    },
    {
        "player": "Player 5",
        "current_round": "III-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 6",
        "current_round": "III-6",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "III-6",
        "last_opponent": "Player 5",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 1",
        "current_round": "IV-1",
        "last_opponent": "M.Player 7",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 2",
        "current_round": "IV-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 3",
        "current_round": "IV-1",
        "last_opponent": "Player 2",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "IV-1",
        "last_opponent": "Player 5",
        "next_opponent": "Player 1"
    },
    {
        "player": "Player 6",
        "current_round": "IV-1",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "IV-1",
        "last_opponent": "Player 6",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 2",
        "current_round": "IV-4",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "IV-4",
        "last_opponent": "Player 4",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "IV-4",
        "last_opponent": "Player 3",
        "next_opponent": "Player 2"
    },
    {
        "player": "Player 6",
        "current_round": "IV-4",
        "last_opponent": "Player 2",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "IV-4",
        "last_opponent": "M.Player 4",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "IV-5",
        "last_opponent": "Player 7",
        "next_opponent": "Player 6"
    },
    {
        "player": "Player 4",
        "current_round": "IV-5",
        "last_opponent": "Player 2",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 6",
        "current_round": "IV-5",
        "last_opponent": "M.Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "IV-5",
        "last_opponent": "Player 3",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 3",
        "current_round": "IV-6",
        "last_opponent": "Player 6",
        "next_opponent": "Player 4"
    },
    {
        "player": "Player 4",
        "current_round": "IV-6",
        "last_opponent": "Player 7",
        "next_opponent": "Player 3"
    },
    {
        "player": "Player 7",
        "current_round": "IV-6",
        "last_opponent": "Player 4",
        "next_opponent": "M.Player 3"
    },
    {
        "player": "Player 3",
        "current_round": "V-1",
        "last_opponent": "Player 4",
        "next_opponent": "M.Player 7"
    },
    {
        "player": "Player 4",
        "current_round": "V-1",
        "last_opponent": "Player 3",
        "next_opponent": "Player 7"
    },
    {
        "player": "Player 7",
        "current_round": "V-1",
        "last_opponent": "M.Player 3",
        "next_opponent": "Player 4"
    }
]

# Global model and encoders
model = None
le_player = LabelEncoder()
le_round = LabelEncoder()
le_last_opp = LabelEncoder()
le_next_opp = LabelEncoder()

# SESSION data
session_history = []
dead_players = set()

def initialize_model():
    global model, le_player, le_round, le_last_opp, le_next_opp

    players = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6", "Player 7", "Player 8",
               "M.Player 1", "M.Player 2", "M.Player 3", "M.Player 4", "M.Player 5", "M.Player 6", "M.Player 7", "M.Player 8"]
    rounds = ["I-2", "I-3", "I-4", "II-1", "II-4", "II-5", "II-6",
              "III-1", "III-2", "III-3", "III-4", "III-5", "III-6",
              "IV-1", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6",
              "V-1", "V-2", "V-3"]

    le_player.fit(players)
    le_round.fit(rounds)
    le_last_opp.fit(players)
    le_next_opp.fit(players)

    X = []
    y = []

    for sample in training_data:
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

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

def predict_with_session(player, round_name, last_opponent):
    try:
        x = np.array([[
            le_player.transform([player])[0],
            le_round.transform([round_name])[0],
            le_last_opp.transform([last_opponent])[0],
            1
        ]])

        probs = model.predict_proba(x)[0]
        all_opponents = le_next_opp.classes_
        available_probs = []

        for idx, opponent in enumerate(all_opponents):
            if opponent not in dead_players:
                available_probs.append((opponent, probs[idx] * 100, idx))

        available_probs.sort(key=lambda x: x[1], reverse=True)
        top3 = [(name, prob) for name, prob, _ in available_probs[:3]]

        session_matches = [s for s in session_history if s["player"] == player and s["round"] == round_name and s["last_opponent"] == last_opponent]

        if session_matches:
            correct_opponent = session_matches[-1]["actual_next"]
            if correct_opponent not in dead_players:
                top3_opponents = [t[0] for t in top3]
                if correct_opponent in top3_opponents:
                    idx = top3_opponents.index(correct_opponent)
                    top3.insert(0, top3.pop(idx))

        return {"top3": top3, "session_matches": len(session_matches), "dead_players": list(dead_players)}
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    player = data.get('player')
    round_name = data.get('round')
    last_opponent = data.get('last_opponent')

    if not all([player, round_name, last_opponent]):
        return jsonify({"error": "Missing required fields"}), 400

    result = predict_with_session(player, round_name, last_opponent)
    return jsonify(result) if "error" not in result else (jsonify(result), 400)

@app.route('/api/log-feedback', methods=['POST'])
def log_feedback():
    global session_history
    data = request.json

    player = data.get('player')
    round_name = data.get('round')
    last_opponent = data.get('last_opponent')
    selected_option = data.get('selected_option')
    prediction_options = data.get('prediction_options')

    if not all([player, round_name, last_opponent, selected_option, prediction_options]):
        return jsonify({"error": "Missing required fields"}), 400

    if selected_option in ["A", "B", "C"]:
        idx = ord(selected_option) - ord('A')
        if idx < len(prediction_options):
            actual_opponent = prediction_options[idx][0]
        else:
            return jsonify({"error": "Invalid option"}), 400
    else:
        actual_opponent = selected_option

    session_history.append({"player": player, "round": round_name, "last_opponent": last_opponent, "actual_next": actual_opponent})

    return jsonify({"message": "Logged", "session_size": len(session_history), "logged_opponent": actual_opponent})

@app.route('/api/toggle-death', methods=['POST'])
def toggle_death():
    global dead_players
    data = request.json
    player = data.get('player')

    if not player:
        return jsonify({"error": "Missing player"}), 400

    if player in dead_players:
        dead_players.remove(player)
        status = "alive"
    else:
        dead_players.add(player)
        status = "dead"

    return jsonify({"message": f"{player} is {status}", "dead_players": list(dead_players)})

@app.route('/api/session-stats', methods=['GET'])
def session_stats():
    return jsonify({
        "session_matches_logged": len(session_history),
        "base_model_training_samples": len(training_data),
        "total_effective_samples": len(training_data) + len(session_history),
        "dead_players": list(dead_players)
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "training_samples": len(training_data)})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

initialize_model()
app = app
