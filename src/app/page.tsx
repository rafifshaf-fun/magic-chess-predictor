// pages/index.tsx
"use client" 
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PLAYERS = Array.from({ length: 8 }, (_, i) => `Player ${i + 1}`);
const ROUNDS = [
  'I-1', 'I-2', 'I-3', 'I-4',
  'II-1', 'II-2', 'II-3', 'II-4', 'II-5', 'II-6',
  'III-1', 'III-2', 'III-3', 'III-4', 'III-5', 'III-6',
  'IV-1', 'IV-2', 'IV-3', 'IV-4', 'IV-5', 'IV-6',
  'V-1', 'V-2', 'V-3', 'V-4'
];

interface Prediction {
  opponent: string;
  probability: number;
}

interface PredictionResult {
  round: string;
  opponent: string;
  predictions: Prediction[];
}

export default function Home() {
  const [selectedPlayer, setSelectedPlayer] = useState<string>('Player 1');
  const [currentRound, setCurrentRound] = useState<string>('I-2');
  const [lastOpponent, setLastOpponent] = useState<string>('Player 3');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<PredictionResult[]>([]);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

  const handlePredict = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${apiUrl}/api/predict`, {
        player: selectedPlayer,
        current_round: currentRound,
        last_opponent: lastOpponent,
      });

      const nextPredictions = response.data.next_predictions;
      setPredictions(nextPredictions);

      // Add to history
      setHistory([
        ...history,
        {
          round: currentRound,
          opponent: lastOpponent,
          predictions: nextPredictions,
        },
      ]);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to get prediction');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectPrediction = (opponent: string) => {
    // Find next round
    const currentRoundIdx = ROUNDS.indexOf(currentRound);
    if (currentRoundIdx < ROUNDS.length - 1) {
      const nextRound = ROUNDS[currentRoundIdx + 1];
      setCurrentRound(nextRound);
      setLastOpponent(opponent);
      
      // Auto-predict for next round
      setTimeout(() => {
        handlePredictWithValues(selectedPlayer, nextRound, opponent);
      }, 100);
    }
  };

  const handlePredictWithValues = async (
    player: string,
    round: string,
    opponent: string
  ) => {
    try {
      const response = await axios.post(`${apiUrl}/api/predict`, {
        player: player,
        current_round: round,
        last_opponent: opponent,
      });

      const nextPredictions = response.data.next_predictions;
      setPredictions(nextPredictions);

      setHistory([
        ...history,
        {
          round: round,
          opponent: opponent,
          predictions: nextPredictions,
        },
      ]);
    } catch (err: any) {
      console.error('Error:', err);
    }
  };

  const resetGame = () => {
    setSelectedPlayer('Player 1');
    setCurrentRound('I-2');
    setLastOpponent('Player 3');
    setPredictions([]);
    setHistory([]);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-2">
            🎲 Magic Chess Opponent Predictor
          </h1>
          <p className="text-lg text-blue-200">
            Predict your next opponent using advanced machine learning
          </p>
        </div>

        {/* Main Container */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-xl p-6 sticky top-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Game Setup</h2>

              {/* Player Selection */}
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">
                  Which Player Are You?
                </label>
                <select
                  style={{ colorScheme: "light" }}
                  value={selectedPlayer}
                  onChange={(e) => {
                    setSelectedPlayer(e.target.value);
                    resetGame();
                  }}
                  className="w-full rounded-xl border-2 border-purple-500 bg-white px-4 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-purple-300"                >
                  {PLAYERS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              {/* Round Selection */}
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">
                  Current Round
                </label>
                <select
                  style={{ colorScheme: "light" }}
                  value={currentRound}
                  onChange={(e) => setCurrentRound(e.target.value)}
                  className="w-full rounded-xl border-2 border-purple-500 bg-white px-4 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-purple-300"
                >
                  {ROUNDS.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </div>

              {/* Last Opponent Selection */}
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">
                  Last Opponent
                </label>
                <select
                  style={{ colorScheme: "light" }}
                  value={lastOpponent}
                  onChange={(e) => setLastOpponent(e.target.value)}
                  className="w-full rounded-xl border-2 border-purple-500 bg-white px-4 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-purple-300"
                >
                  {PLAYERS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              {/* Predict Button */}
              <button
                onClick={handlePredict}
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold py-3 px-4 rounded-lg hover:from-blue-700 hover:to-purple-700 transition disabled:opacity-50 disabled:cursor-not-allowed mb-4"
              >
                {loading ? 'Predicting...' : '🔮 Predict'}
              </button>

              <button
                onClick={resetGame}
                className="w-full bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-lg hover:bg-gray-400 transition"
              >
                Reset Game
              </button>

              {/* Status Info */}
              <div className="mt-6 p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                <p className="text-sm text-gray-700">
                  <span className="font-semibold">Player:</span> {selectedPlayer}
                </p>
                <p className="text-sm text-gray-700">
                  <span className="font-semibold">Round:</span> {currentRound}
                </p>
                <p className="text-sm text-gray-700">
                  <span className="font-semibold">Last Opponent:</span> {lastOpponent}
                </p>
              </div>
            </div>
          </div>

          {/* Right Panel - Predictions */}
          <div className="lg:col-span-2">
            {error && (
              <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg mb-6">
                <p className="font-bold">Error</p>
                <p>{error}</p>
              </div>
            )}

            {/* Predictions Display */}
            {predictions.length > 0 && (
              <div className="bg-white rounded-lg shadow-xl p-6 mb-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">
                  📊 Next Opponent Predictions
                </h2>
                <div className="space-y-4">
                  {predictions.map((pred, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSelectPrediction(pred.opponent)}
                      className="w-full text-left p-4 bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-200 rounded-lg hover:border-blue-500 hover:shadow-lg transition transform hover:scale-102 cursor-pointer"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-xl font-bold text-gray-800">
                            {pred.opponent}
                          </p>
                          <p className="text-sm text-gray-600">
                            Click to continue prediction
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-3xl font-bold text-transparent bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text">
                            {pred.probability}%
                          </p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-4">
                  💡 Tip: Click on any opponent to continue the prediction chain
                </p>
              </div>
            )}

            {/* Prediction History */}
            {history.length > 0 && (
              <div className="bg-white rounded-lg shadow-xl p-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">
                  📜 Prediction History
                </h2>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {history.map((item, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-gray-50 rounded-lg border-l-4 border-blue-500"
                    >
                      <p className="font-semibold text-gray-800 mb-2">
                        Round {item.round}: {item.opponent}
                      </p>
                      <div className="flex gap-2 flex-wrap">
                        {item.predictions.slice(0, 3).map((pred, pidx) => (
                          <span
                            key={pidx}
                            className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-semibold"
                          >
                            {pred.opponent} ({pred.probability}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Empty State */}
            {predictions.length === 0 && history.length === 0 && (
              <div className="bg-white rounded-lg shadow-xl p-12 text-center">
                <p className="text-4xl mb-4">🎮</p>
                <h3 className="text-2xl font-bold text-gray-800 mb-2">
                  Ready to Predict?
                </h3>
                <p className="text-gray-600 text-lg">
                  Select your player and last opponent, then click "Predict" to see
                  the 3 most likely next opponents!
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-blue-200">
          <p className="text-sm">
            Powered by Machine Learning | Magic Chess Go Go Opponent Predictor v1.0
          </p>
        </div>
      </div>
    </div>
  );
}
