"""
Magic Chess Go Go — Opponent Prediction API
============================================
Flask backend that learns opponent assignment patterns from tracked match data
and predicts the most likely next opponent a player will face.
"""

import logging
import os
import pickle
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_CACHE_FILE = "model_cache.pkl"
MATCH_FILE_PATTERN = "Match-*.csv"
NUM_PLAYERS = 8
ALL_PLAYERS = [f"Player {i}" for i in range(1, NUM_PLAYERS + 1)]

# Roman → numeric mapping for round parsing
ROMAN_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

# Rounds per stage: each stage has a different number of rounds
ROUNDS_PER_STAGE = {"I": 4, "II": 6, "III": 6, "IV": 6, "V": 4}

# Players eliminated per round (index 0 = round I-1, etc.) — accumulated
# These are typical values; adjust if your matches differ
ROUND_LIST = [
    "I-1", "I-2", "I-3", "I-4",
    "II-1", "II-2", "II-3", "II-4", "II-5", "II-6",
    "III-1", "III-2", "III-3", "III-4", "III-5", "III-6",
    "IV-1", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6",
    "V-1", "V-2", "V-3", "V-4",
]

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


# ═══════════════════════════════════════════════════════════════════════════
#  ROUND UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def round_to_absolute_index(round_str: str) -> int:
    """Convert a round label like 'III-4' to its 0-indexed position in ROUND_LIST.

    Examples:
        'I-1'   → 0
        'I-4'   → 3
        'II-1'  → 4
        'V-4'   → 25
    """
    if round_str in ROUND_LIST:
        return ROUND_LIST.index(round_str)
    # Fallback: try to parse with regex
    match = re.match(r"([IV]+)-(\d+)", round_str.strip())
    if not match:
        log.warning("Could not parse round: %s", round_str)
        return 0
    roman_part, num_part = match.groups()
    # Count preceding rounds from earlier stages
    offset = 0
    for stage, count in ROUNDS_PER_STAGE.items():
        if stage == roman_part:
            break
        offset += count
    return offset + int(num_part) - 1


def get_next_round(current_round: str) -> Optional[str]:
    """Return the next round label, or None if at the last round."""
    idx = round_to_absolute_index(current_round)
    if idx + 1 < len(ROUND_LIST):
        return ROUND_LIST[idx + 1]
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def clean_opponent_name(raw: str) -> Optional[str]:
    """Normalise an opponent cell value.

    Returns None for rounds that should be skipped (Creep rounds, eliminated players).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if s in ("", "Creep", "Null"):
        return None
    # Normalise "M.Player X" → "Player X"
    s = re.sub(r"^M\.\s*", "", s)
    # Validate it looks like "Player N"
    if re.match(r"^Player\s+\d+$", s):
        return s
    log.debug("Unrecognised opponent value: %r", raw)
    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to every opponent cell in a match DataFrame."""
    df = df.copy()
    for col in df.columns[1:]:  # skip the first (round label) column
        df[col] = df[col].apply(clean_opponent_name)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def discover_match_files(base_dir: str = ".") -> List[str]:
    """Find all match CSV files sorted naturally."""
    from pathlib import Path
    files = sorted(
        Path(base_dir).glob(MATCH_FILE_PATTERN),
        key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.name)],
    )
    return [str(f) for f in files]


def load_training_data(base_dir: str = ".") -> List[pd.DataFrame]:
    """Load and clean all match CSV files."""
    matches: List[pd.DataFrame] = []
    files = discover_match_files(base_dir)
    log.info("Found %d match file(s): %s", len(files), [Path(f).name for f in files])

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            df = clean_dataframe(df)
            matches.append(df)
        except FileNotFoundError:
            log.warning("File not found: %s", filepath)
        except Exception:
            log.exception("Error loading %s", filepath)

    log.info("Loaded %d match(es) successfully.", len(matches))
    return matches


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_models(matches: List[pd.DataFrame]):
    """Build transition, position, and bigram models from cleaned match data.

    Returns:
        transition_model:  (player, last_opponent) → Counter[next_opponent]
        position_model:    (player, round_absolute_idx) → Counter[opponent]
        bigram_model:      (player, prev_opp, curr_opp) → Counter[next_opp]
        player_survival:   dict[round_label] → set of alive players (per match)
    """
    transition_model: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    position_model: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    bigram_model: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    player_survival: List[Dict[str, set]] = []  # per-match survival

    for match_df in matches:
        players = list(match_df.columns[1:])
        round_labels = match_df.iloc[:, 0].tolist()

        # Track which players are alive at each round in this match
        match_alive: Dict[str, set] = {}

        for round_pos, round_label in enumerate(round_labels):
            alive_this_round = set()
            for player in players:
                raw = match_df.at[round_pos, player] if round_pos < len(match_df) else None
                opp = clean_opponent_name(raw)
                if opp is not None:
                    alive_this_round.add(player)
            match_alive[round_label] = alive_this_round

            # Position model: at this round index, who does each player face?
            for player in players:
                raw = match_df.at[round_pos, player]
                opp = clean_opponent_name(raw)
                if opp and opp != player:
                    abs_idx = round_to_absolute_index(round_label)
                    position_model[(player, abs_idx)][opp] += 1

        player_survival.append(match_alive)

        # Transition & bigram models
        for player in players:
            opponents: List[Optional[str]] = []
            for round_pos in range(len(round_labels)):
                raw = match_df.at[round_pos, player]
                opponents.append(clean_opponent_name(raw))

            for i in range(len(opponents) - 1):
                curr = opponents[i]
                nxt = opponents[i + 1]
                if curr is None or nxt is None:
                    continue
                # Single-step transition
                transition_model[(player, curr)][nxt] += 1

                # Bigram (2-step context)
                if i >= 1:
                    prev = opponents[i - 1]
                    if prev is not None:
                        bigram_model[(player, prev, curr)][nxt] += 1

    return transition_model, position_model, bigram_model, player_survival


# ═══════════════════════════════════════════════════════════════════════════
#  ALIVE-PLAYER ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_round_alive_estimates(
    player_survival_data: List[Dict[str, set]],
    threshold: float = 0.5,
) -> Dict[int, set]:
    """For each absolute round index, estimate which players are typically alive.

    A player is considered 'alive' at a given round if they survived to that
    point in more than *threshold* fraction of matches.

    This feeds into the prediction engine so that eliminated players are
    deprioritised and alive-but-unseen opponents get a fair baseline weight.
    """
    if not player_survival_data:
        return {}

    total = len(player_survival_data)
    # round_label -> player -> count of matches where alive
    raw: Dict[str, Counter] = defaultdict(Counter)

    for match_alive in player_survival_data:
        for round_label, alive_set in match_alive.items():
            for player_name in alive_set:
                raw[round_label][player_name] += 1

    estimates: Dict[int, set] = {}
    for round_label, counter in raw.items():
        idx = round_to_absolute_index(round_label)
        alive = {p for p, c in counter.items() if c / total > threshold}
        estimates[idx] = alive

    return estimates


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def save_models(path: str = MODEL_CACHE_FILE) -> None:
    """Pickle the models to disk for fast reload."""
    data = {
        "transition_model": dict(transition_model),
        "position_model": dict(position_model),
        "bigram_model": dict(bigram_model),
        "player_survival": player_survival,
        "round_alive_estimates": round_alive_estimates,
        "saved_at": datetime.now().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Models cached to %s", path)


def load_models(path: str = MODEL_CACHE_FILE) -> Optional[dict]:
    """Load cached models if available and not stale."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Basic validation
        for key in ("transition_model", "position_model", "bigram_model"):
            if key not in data:
                log.warning("Cached model missing key '%s', rebuilding.", key)
                return None
        log.info("Models loaded from cache (%s)", data.get("saved_at", "unknown"))
        return data
    except Exception:
        log.exception("Failed to load cached models, rebuilding.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def predict_next_opponent(
    player: str,
    current_round_idx: int,
    last_opponent: str,
    previous_opponent: Optional[str] = None,
) -> List[dict]:
    """Predict the next opponent with confidence scores.

    Combines five strategies weighted by reliability:

      1. Bigram (prev → curr → next)           — weight 5
      2. Transition (curr → next)               — weight 4
      3. Position (next round index → who)      — weight 3
      4. Alive-but-unseen boost                 — weight 1
         (opponents likely alive but never seen in this context get a small
          baseline score so they aren't excluded just because of sparse data)
      5. General frequency fallback             — weight 1 (empty state)

    Returns a list of up to 3 named opponents plus an "Other Players" entry
    that *also* lists the remaining alive candidates so the user knows who
    else is in the pool.
    """
    scores: Counter = Counter()
    alive_estimate: Optional[set] = round_alive_estimates.get(current_round_idx + 1)

    # ── Strategy 1: Bigram (2-step Markov) ──
    if previous_opponent:
        bigram_key = (player, previous_opponent, last_opponent)
        if bigram_key in bigram_model:
            for opp, count in bigram_model[bigram_key].items():
                scores[opp] += count * 5

    # ── Strategy 2: Single-step transition ──
    trans_key = (player, last_opponent)
    if trans_key in transition_model:
        for opp, count in transition_model[trans_key].items():
            scores[opp] += count * 4

    # ── Strategy 3: Positional ──
    next_idx = current_round_idx + 1
    pos_key = (player, next_idx)
    if pos_key in position_model:
        for opp, count in position_model[pos_key].items():
            if opp != player:
                scores[opp] += count * 3

    # ── Strategy 4: Alive-but-unseen boost ──
    # Magic Chess Go Go's matchmaking tends to avoid re-pairing you with
    # recent opponents. Players who are probably alive but never appeared
    # in this context get a small baseline score so they stay in the pool.
    if alive_estimate:
        for p in alive_estimate:
            if p != player and p not in scores:
                scores[p] += 1  # Small weight — "don't forget me"

    # ── Strategy 5: General frequency fallback ──
    if not scores:
        for p in ALL_PLAYERS:
            if p != player:
                scores[p] = 1

    # ── Build response ──
    total = sum(scores.values())
    top_n = scores.most_common(3)

    predictions = []
    top_sum = 0.0
    top_set: set = set()
    for opp, count in top_n:
        prob = round((count / total) * 100, 1)
        predictions.append({"opponent": opp, "probability": prob})
        top_sum += prob
        top_set.add(opp)

    # ── Other candidates (named, not just a lump) ──
    other_candidates: List[dict] = []
    if alive_estimate:
        for p in sorted(alive_estimate, key=lambda x: scores.get(x, 0), reverse=True):
            if p != player and p not in top_set:
                weight = scores.get(p, 0)
                if weight > 0:
                    share = round((weight / total) * 100, 1)
                    other_candidates.append({"opponent": p, "probability": share})

    other_prob = round(100.0 - top_sum, 1)
    if other_prob > 0.0:
        entry: dict = {"opponent": "Other Players", "probability": other_prob}
        if other_candidates:
            entry["candidates"] = other_candidates
        predictions.append(entry)

    return predictions


def predict_chain(
    player: str,
    history: List[dict],
) -> List[dict]:
    """Predict a chain of opponents given a match history.

    Args:
        player:  e.g. "Player 3"
        history: list of {"round": "III-2", "opponent": "Player 5"} dicts

    Returns:
        list of results with predictions for each step
    """
    results = []
    for i, item in enumerate(history):
        round_str = item.get("round", "I-1")
        opponent = item.get("opponent", "")
        prev_opp = history[i - 1]["opponent"] if i > 0 else None

        if not opponent:
            continue

        round_idx = round_to_absolute_index(round_str)
        preds = predict_next_opponent(
            player, round_idx, opponent, previous_opponent=prev_opp,
        )
        results.append({
            "round": round_str,
            "opponent": opponent,
            "predictions": preds,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "matches_loaded": match_count,
        "models_cached": os.path.exists(MODEL_CACHE_FILE),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/players", methods=["GET"])
def get_players():
    return jsonify({"players": ALL_PLAYERS})


@app.route("/api/rounds", methods=["GET"])
def get_rounds():
    """Return the canonical round list for the frontend."""
    return jsonify({"rounds": ROUND_LIST})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return model statistics for debugging / transparency."""
    return jsonify({
        "matches_loaded": match_count,
        "transition_entries": len(transition_model),
        "position_entries": len(position_model),
        "bigram_entries": len(bigram_model),
        "rounds_tracked": len(ROUND_LIST),
        "round_alive_estimates": len(round_alive_estimates),
        "players": ALL_PLAYERS,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict the next opponent.

    Request body (JSON):
        player         — e.g. "Player 1"
        current_round  — e.g. "III-4"
        last_opponent  — e.g. "Player 5"
        previous_opponent (optional) — e.g. "Player 2"  (for bigram context)
    """
    try:
        data = request.get_json(silent=True) or {}
        player = (data.get("player") or "").strip()
        current_round = (data.get("current_round") or "I-1").strip()
        last_opponent = (data.get("last_opponent") or "").strip()
        previous_opponent = (data.get("previous_opponent") or "").strip() or None

        # ── Validation ──
        errors = []
        if not player:
            errors.append("'player' is required")
        elif player not in ALL_PLAYERS:
            errors.append(f"Invalid player '{player}'. Must be one of {ALL_PLAYERS}")

        if not last_opponent:
            errors.append("'last_opponent' is required")

        if current_round not in ROUND_LIST:
            errors.append(f"Invalid round '{current_round}'. Must be one of {ROUND_LIST}")

        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        round_idx = round_to_absolute_index(current_round)

        preds = predict_next_opponent(
            player, round_idx, last_opponent,
            previous_opponent=previous_opponent,
        )

        next_round = get_next_round(current_round)

        return jsonify({
            "success": True,
            "player": player,
            "current_round": current_round,
            "next_round": next_round,
            "last_opponent": last_opponent,
            "previous_opponent": previous_opponent,
            "next_predictions": preds,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception:
        log.exception("Prediction error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """Predict for a sequence of rounds (full match chain).

    Request body:
        player  — e.g. "Player 1"
        history — [{"round": "I-2", "opponent": "Player 3"}, ...]
    """
    try:
        data = request.get_json(silent=True) or {}
        player = (data.get("player") or "").strip()
        history: List[dict] = data.get("history", [])

        if not player:
            return jsonify({"success": False, "error": "'player' is required"}), 400
        if player not in ALL_PLAYERS:
            return jsonify({"success": False, "error": f"Invalid player '{player}'"}), 400

        results = predict_chain(player, history)
        return jsonify({
            "success": True,
            "player": player,
            "predictions_history": results,
        })

    except Exception:
        log.exception("Batch prediction error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/rebuild-models", methods=["POST"])
def rebuild_models():
    """Admin endpoint: force model rebuild from CSVs and re-cache."""
    global transition_model, position_model, bigram_model, player_survival, round_alive_estimates, match_count
    try:
        matches = load_training_data()
        transition_model, position_model, bigram_model, player_survival = build_models(matches)
        round_alive_estimates = compute_round_alive_estimates(player_survival)
        match_count = len(matches)
        save_models()
        return jsonify({
            "success": True,
            "matches_reloaded": match_count,
            "transition_entries": len(transition_model),
            "position_entries": len(position_model),
            "bigram_entries": len(bigram_model),
        })
    except Exception:
        log.exception("Model rebuild failed")
        return jsonify({"success": False, "error": "Rebuild failed"}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "Magic Chess Go Go — Opponent Predictor API",
        "version": "2.0.0",
        "endpoints": {
            "GET  /api/health": "Health check & model info",
            "GET  /api/players": "List all players",
            "GET  /api/rounds": "List all round labels",
            "GET  /api/stats": "Model statistics",
            "POST /api/predict": "Predict next opponent",
            "POST /api/predict-batch": "Batch predict for a full match",
            "POST /api/rebuild-models": "Force model rebuild from CSVs",
            "GET  /": "This documentation",
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════

def initialize():
    """Try loading cached models; fall back to rebuilding from CSVs."""
    global transition_model, position_model, bigram_model, player_survival, round_alive_estimates, match_count

    cached = load_models()
    if cached:
        transition_model = defaultdict(Counter, {
            tuple(k) if isinstance(k, tuple) else k: Counter(v)
            for k, v in cached["transition_model"].items()
        })
        position_model = defaultdict(Counter, {
            tuple(k) if isinstance(k, tuple) else k: Counter(v)
            for k, v in cached["position_model"].items()
        })
        bigram_model = defaultdict(Counter, {
            tuple(k) if isinstance(k, tuple) else k: Counter(v)
            for k, v in cached["bigram_model"].items()
        })
        player_survival = cached.get("player_survival", [])
        round_alive_estimates = cached.get("round_alive_estimates", {})
        if not round_alive_estimates and player_survival:
            log.info("Computing alive estimates from survival data…")
            round_alive_estimates = compute_round_alive_estimates(player_survival)
        match_count = len(player_survival)
    else:
        log.info("No cache found — building models from CSV files…")
        matches = load_training_data()
        transition_model, position_model, bigram_model, player_survival = build_models(matches)
        round_alive_estimates = compute_round_alive_estimates(player_survival)
        match_count = len(matches)
        save_models()

    log.info(
        "Ready: %d matches | %d transition rules | %d position rules | %d bigram rules | %d round estimates",
        match_count,
        len(transition_model),
        len(position_model),
        len(bigram_model),
        len(round_alive_estimates),
    )


# Module-level globals (populated at startup)
transition_model: Dict[Tuple[str, str], Counter] = {}
position_model: Dict[Tuple[str, int], Counter] = {}
bigram_model: Dict[Tuple[str, str, str], Counter] = {}
player_survival: List[dict] = []
round_alive_estimates: Dict[int, set] = {}
match_count: int = 0

initialize()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
