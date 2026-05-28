"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import axios from "axios";

// ── Constants ────────────────────────────────────────────────────────────
const PLAYERS = Array.from({ length: 8 }, (_, i) => `Player ${i + 1}`);
const ROUNDS = [
  "I-1", "I-2", "I-3", "I-4",
  "II-1", "II-2", "II-3", "II-4", "II-5", "II-6",
  "III-1", "III-2", "III-3", "III-4", "III-5", "III-6",
  "IV-1", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6",
  "V-1", "V-2", "V-3", "V-4",
];

const STAGE_LABELS: Record<string, string> = {
  I: "Stage I",
  II: "Stage II",
  III: "Stage III",
  IV: "Stage IV",
  V: "Stage V",
};

const STAGE_COLORS: Record<string, string> = {
  I: "from-emerald-500 to-teal-500",
  II: "from-blue-500 to-indigo-500",
  III: "from-violet-500 to-purple-500",
  IV: "from-orange-500 to-red-500",
  V: "from-pink-500 to-rose-500",
};

// ── Types ────────────────────────────────────────────────────────────────
interface Prediction {
  opponent: string;
  probability: number;
}

interface HistoryEntry {
  round: string;
  opponent: string;
  predictions: Prediction[];
}

interface Stats {
  matches_loaded: number;
  transition_entries: number;
  position_entries: number;
  bigram_entries: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────
const getStageFromRound = (round: string): string => round.split("-")[0];
const roundIndex = (round: string): number => ROUNDS.indexOf(round);
const nextRound = (round: string): string | null => {
  const idx = roundIndex(round);
  return idx >= 0 && idx < ROUNDS.length - 1 ? ROUNDS[idx + 1] : null;
};
const totalRounds = ROUNDS.length;

// ── Sub-components ───────────────────────────────────────────────────────

/** Animated progress bar showing match progression */
function RoundProgress({ currentRound }: { currentRound: string }) {
  const idx = roundIndex(currentRound);
  const pct = Math.round(((idx + 1) / totalRounds) * 100);
  const stage = getStageFromRound(currentRound);

  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span className="font-semibold text-indigo-600">
          {STAGE_LABELS[stage] || stage}
        </span>
        <span>
          Round {currentRound} ({idx + 1}/{totalRounds})
        </span>
      </div>
      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          // eslint-disable-next-line @next/next/no-css-tags
          className={`h-full bg-gradient-to-r ${STAGE_COLORS[stage] || "from-gray-400 to-gray-500"} rounded-full transition-all duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/** Confidence bar for a single prediction */
function ConfidenceBar({ probability, isOther }: { probability: number; isOther: boolean }) {
  return (
    <div className="w-full h-1.5 bg-gray-100 rounded-full mt-1 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${
          isOther
            ? "bg-gray-300"
            : probability > 50
              ? "bg-gradient-to-r from-green-400 to-emerald-500"
              : probability > 25
                ? "bg-gradient-to-r from-blue-400 to-indigo-500"
                : "bg-gradient-to-r from-amber-400 to-orange-500"
        }`}
        style={{ width: `${probability}%` }}
      />
    </div>
  );
}

/** Loading skeleton placeholder */
function PredictionSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      {[1, 2, 3].map((i) => (
        <div key={i} className="p-4 border-2 border-gray-200 rounded-lg">
          <div className="flex justify-between">
            <div className="space-y-2">
              <div className="h-6 w-24 bg-gray-200 rounded" />
              <div className="h-4 w-40 bg-gray-100 rounded" />
            </div>
            <div className="h-8 w-16 bg-gray-200 rounded" />
          </div>
          <div className="mt-2 h-1.5 bg-gray-100 rounded-full" />
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export default function Home() {
  const [selectedPlayer, setSelectedPlayer] = useState<string>("Player 1");
  const [currentRound, setCurrentRound] = useState<string>("I-2");
  const [lastOpponent, setLastOpponent] = useState<string>("Player 3");
  const [previousOpponent, setPreviousOpponent] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [needsAttention, setNeedsAttention] = useState<boolean>(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [historyExpanded, setHistoryExpanded] = useState<boolean>(true);

  const historyEndRef = useRef<HTMLDivElement>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  // Fetch model stats on mount
  useEffect(() => {
    axios
      .get(`${apiUrl}/api/stats`)
      .then((res) => setStats(res.data))
      .catch(() => {});
  }, [apiUrl]);

  // Auto-scroll history
  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  // ── Core prediction function ──
  const fetchPrediction = useCallback(
    async (
      player: string,
      round: string,
      opponent: string,
      prevOpp: string | null,
    ): Promise<Prediction[] | null> => {
      try {
        const payload: Record<string, string> = {
          player,
          current_round: round,
          last_opponent: opponent,
        };
        if (prevOpp) {
          payload.previous_opponent = prevOpp;
        }
        const response = await axios.post(`${apiUrl}/api/predict`, payload);
        return response.data.next_predictions ?? [];
      } catch (err: any) {
        const msg =
          err.response?.data?.errors?.join(", ") ||
          err.response?.data?.error ||
          "Network error — is the backend running?";
        setError(msg);
        console.error("Prediction error:", err);
        return null;
      }
    },
    [apiUrl],
  );

  // ── Handle Predict button ──
  const handlePredict = useCallback(async () => {
    if (!lastOpponent) {
      setError("Please select the Last Opponent before predicting.");
      setNeedsAttention(true);
      return;
    }

    setLoading(true);
    setError(null);

    const result = await fetchPrediction(
      selectedPlayer,
      currentRound,
      lastOpponent,
      previousOpponent,
    );

    if (result) {
      setPredictions(result);
      setHistory((prev) => [
        ...prev,
        { round: currentRound, opponent: lastOpponent, predictions: result },
      ]);
    }
    setLoading(false);
  }, [
    selectedPlayer,
    currentRound,
    lastOpponent,
    previousOpponent,
    fetchPrediction,
  ]);

  // ── Handle clicking a prediction card ──
  const handleSelectPrediction = useCallback(
    async (opponent: string) => {
      const nxt = nextRound(currentRound);
      if (!nxt) return; // game over

      // Advance state
      setPreviousOpponent(lastOpponent || null);
      setCurrentRound(nxt);
      setLastOpponent(opponent);
      setLoading(true);
      setError(null);

      const result = await fetchPrediction(selectedPlayer, nxt, opponent, lastOpponent || null);
      if (result) {
        setPredictions(result);
        setHistory((prev) => [
          ...prev,
          { round: nxt, opponent, predictions: result },
        ]);
      }
      setLoading(false);
    },
    [currentRound, lastOpponent, selectedPlayer, fetchPrediction],
  );

  // ── Handle "Other Players" fallback ──
  const handleOtherSelected = useCallback(() => {
    const nxt = nextRound(currentRound);
    if (!nxt) return;
    setPreviousOpponent(lastOpponent || null);
    setCurrentRound(nxt);
    setLastOpponent("");
    setPredictions([]);
    setNeedsAttention(true);
    setError(
      "None of our predictions matched! Please manually select your real opponent from the highlighted dropdown on the left.",
    );
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentRound, lastOpponent]);

  // ── Reset ──
  const resetGame = useCallback(() => {
    setSelectedPlayer("Player 1");
    setCurrentRound("I-2");
    setLastOpponent("Player 3");
    setPreviousOpponent(null);
    setPredictions([]);
    setHistory([]);
    setError(null);
    setNeedsAttention(false);
  }, []);

  // ── Keyboard support ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !loading) {
        e.preventDefault();
        handlePredict();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handlePredict, loading]);

  // ── Derived values ──
  const stage = getStageFromRound(currentRound);
  const progressPct = Math.round(((roundIndex(currentRound) + 1) / totalRounds) * 100);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-950 to-indigo-950 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* ── Header ── */}
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-2 tracking-tight">
            <span className="inline-block animate-bounce-slow">🎯</span>{" "}
            Magic Chess Opponent Predictor
          </h1>
          <p className="text-lg text-purple-200">
            Predict your next opponent using Markov-chain machine learning
          </p>
          {stats && (
            <p className="text-xs text-purple-400/60 mt-1">
              Trained on {stats.matches_loaded} matches ·{" "}
              {stats.transition_entries.toLocaleString()} transition rules ·{" "}
              {stats.bigram_entries.toLocaleString()} bigram rules
            </p>
          )}
        </header>

        {/* ── Main grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ── Left Panel — Controls ── */}
          <aside className="lg:col-span-1">
            <div className="bg-white/95 backdrop-blur rounded-2xl shadow-2xl p-6 sticky top-8 border border-white/20">
              <h2 className="text-xl font-bold text-gray-800 mb-5 flex items-center gap-2">
                <span>⚙️</span> Game Setup
              </h2>

              <RoundProgress currentRound={currentRound} />

              {/* Player */}
              <div className="mb-5">
                <label className="block text-gray-700 font-semibold mb-1.5 text-sm">
                  Which Player Are You?
                </label>
                <select
                  aria-label="Select your player"
                  value={selectedPlayer}
                  onChange={(e) => {
                    setSelectedPlayer(e.target.value);
                    resetGame();
                  }}
                  className="w-full rounded-xl border-2 border-indigo-300 bg-white px-4 py-2.5 text-gray-900 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition"
                >
                  {PLAYERS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              {/* Round */}
              <div className="mb-5">
                <label className="block text-gray-700 font-semibold mb-1.5 text-sm">
                  Current Round
                </label>
                <select
                  aria-label="Select current round"
                  value={currentRound}
                  onChange={(e) => setCurrentRound(e.target.value)}
                  className="w-full rounded-xl border-2 border-indigo-300 bg-white px-4 py-2.5 text-gray-900 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition"
                >
                  {ROUNDS.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </div>

              {/* Last Opponent */}
              <div className="mb-5">
                <label className="block text-gray-700 font-semibold mb-1.5 text-sm">
                  Last Opponent
                  {needsAttention && (
                    <span className="text-red-500 text-xs ml-2 animate-pulse font-bold">
                      ← SELECT ONE!
                    </span>
                  )}
                </label>
                <select
                  aria-label="Select last opponent"
                  value={lastOpponent}
                  onChange={(e) => {
                    setLastOpponent(e.target.value);
                    setNeedsAttention(false);
                    setError(null);
                  }}
                  className={`w-full rounded-xl border-2 px-4 py-2.5 text-gray-900 focus:outline-none transition-all duration-300 ${
                    needsAttention
                      ? "border-red-500 bg-red-50 ring-4 ring-red-200 animate-pulse"
                      : "border-indigo-300 bg-white focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                  }`}
                >
                  <option value="" disabled>
                    — Select the real opponent —
                  </option>
                  {PLAYERS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              {/* Previous Opponent (for bigram context) — auto-managed, shown for info */}
              {previousOpponent && (
                <div className="mb-5 p-3 bg-amber-50 rounded-lg border border-amber-200 text-sm text-amber-800">
                  <span className="font-semibold">🧠 Context:</span>{" "}
                  Before {lastOpponent || "?"}, you faced{" "}
                  <span className="font-bold">{previousOpponent}</span>
                </div>
              )}

              {/* Buttons */}
              <button
                onClick={handlePredict}
                disabled={loading}
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold py-3 px-4 rounded-xl hover:from-indigo-700 hover:to-purple-700 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-indigo-500/25 mb-3"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Predicting…
                  </span>
                ) : (
                  "🪄 Predict Next Opponent"
                )}
              </button>

              <button
                onClick={resetGame}
                className="w-full bg-gray-200 text-gray-700 font-semibold py-2.5 px-4 rounded-xl hover:bg-gray-300 transition"
              >
                🔄 Reset Game
              </button>

              {/* Status badge */}
              <div className="mt-5 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-indigo-100">
                <p className="text-xs text-gray-600">
                  <span className="font-semibold">Player:</span> {selectedPlayer}
                </p>
                <p className="text-xs text-gray-600">
                  <span className="font-semibold">Round:</span> {currentRound}{" "}
                  <span className="text-indigo-500">({progressPct}% through match)</span>
                </p>
                <p className="text-xs text-gray-600">
                  <span className="font-semibold">Last Opponent:</span>{" "}
                  {lastOpponent || (
                    <span className="text-red-400 italic">Waiting for selection…</span>
                  )}
                </p>
              </div>

              <p className="text-[10px] text-gray-400 mt-3 text-center">
                💡 Press <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">Enter</kbd>{" "}
                to predict
              </p>
            </div>
          </aside>

          {/* ── Right Panel — Results ── */}
          <main className="lg:col-span-2 space-y-6">
            {/* Error banner */}
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 text-red-800 p-4 rounded-r-xl shadow-md animate-shake">
                <p className="font-bold flex items-center text-sm">
                  <span className="mr-2">⚠️</span> Attention Needed
                </p>
                <p className="text-sm mt-1">{error}</p>
              </div>
            )}

            {/* Loading skeleton */}
            {loading && !predictions.length && <PredictionSkeleton />}

            {/* Predictions */}
            {predictions.length > 0 && (
              <section className="bg-white/95 backdrop-blur rounded-2xl shadow-2xl p-6 border border-white/20">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <span>📊</span> Next Opponent Predictions
                  <span className="text-xs font-normal text-gray-400 ml-auto">
                    for {currentRound} → {nextRound(currentRound) || "end"}
                  </span>
                </h2>

                <div className="space-y-3">
                  {predictions.map((pred, idx) => {
                    const isOther = pred.opponent === "Other Players";
                    const isTop = idx === 0 && !isOther;

                    return (
                      <button
                        key={`${pred.opponent}-${idx}`}
                        onClick={() => {
                          if (isOther) {
                            handleOtherSelected();
                          } else {
                            handleSelectPrediction(pred.opponent);
                          }
                        }}
                        disabled={loading}
                        className={`w-full text-left p-4 border-2 rounded-xl transition-all duration-200 transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-wait ${
                          isOther
                            ? "border-dashed border-gray-300 bg-gray-50/80 hover:border-gray-400 hover:bg-gray-100"
                            : isTop
                              ? "border-emerald-300 bg-gradient-to-r from-emerald-50 to-teal-50 hover:border-emerald-400 hover:shadow-lg"
                              : "border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 hover:border-blue-400 hover:shadow-lg"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <p className="text-lg font-bold text-gray-800 truncate">
                                {pred.opponent}
                              </p>
                              {isTop && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-700">
                                  BEST BET
                                </span>
                              )}
                              {isOther && (
                                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-gray-200 text-gray-600">
                                  FALLBACK
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">
                              {isOther
                                ? "Click to manually enter your surprise opponent"
                                : "Click to chain-predict the next round"}
                            </p>
                            <ConfidenceBar probability={pred.probability} isOther={isOther} />

                            {/* Other candidates list — shown when backend returns them */}
                            {isOther && (pred as any).candidates && (pred as any).candidates.length > 0 && (
                              <div className="mt-3 pt-2 border-t border-gray-200">
                                <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1.5">
                                  Also Possible:
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                  {(pred as any).candidates.map((c: any, ci: number) => (
                                    <span
                                      key={ci}
                                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-gray-100 text-gray-600 border border-gray-200"
                                    >
                                      {c.opponent}
                                      <span className="text-gray-400">({c.probability}%)</span>
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                          <div className="text-right flex-shrink-0">
                            <p
                              className={`text-3xl font-extrabold ${
                                isOther
                                  ? "text-gray-400"
                                  : "bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent"
                              }`}
                            >
                              {pred.probability}%
                            </p>
                            <p className="text-[10px] text-gray-400">confidence</p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </section>
            )}

            {/* History */}
            {history.length > 0 && (
              <section className="bg-white/95 backdrop-blur rounded-2xl shadow-2xl p-6 border border-white/20">
                <button
                  onClick={() => setHistoryExpanded(!historyExpanded)}
                  className="w-full flex items-center justify-between text-left"
                >
                  <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                    <span>📜</span> Prediction History
                    <span className="text-sm font-normal text-gray-400">
                      ({history.length} round{history.length !== 1 ? "s" : ""})
                    </span>
                  </h2>
                  <span className={`text-gray-400 text-lg transition-transform duration-200 ${historyExpanded ? "rotate-180" : "rotate-0"}`}>
                    ▼
                  </span>
                </button>

                {historyExpanded && (
                  <div className="mt-4 space-y-3 max-h-80 overflow-y-auto pr-1">
                    {history.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-gray-50 rounded-xl border-l-4 border-indigo-400 hover:bg-gray-100 transition-colors"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <p className="font-semibold text-gray-800 text-sm">
                            Round {item.round}
                          </p>
                          <span className="text-xs font-bold text-indigo-600 bg-indigo-100 px-2 py-0.5 rounded-full">
                            vs {item.opponent}
                          </span>
                        </div>
                        <div className="flex gap-1.5 flex-wrap">
                          {item.predictions.map((pred, pidx) => (
                            <span
                              key={pidx}
                              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
                                pred.opponent === "Other Players"
                                  ? "bg-gray-200 text-gray-500"
                                  : "bg-blue-100 text-blue-700"
                              }`}
                            >
                              {pred.opponent}{" "}
                              <span className="ml-1 opacity-70">{pred.probability}%</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                    <div ref={historyEndRef} />
                  </div>
                )}
              </section>
            )}

            {/* Empty state */}
            {predictions.length === 0 && history.length === 0 && !loading && (
              <div className="bg-white/95 backdrop-blur rounded-2xl shadow-2xl p-12 text-center border border-white/20">
                <p className="text-5xl mb-4">🔮</p>
                <h3 className="text-2xl font-bold text-gray-800 mb-2">
                  Ready to Predict?
                </h3>
                <p className="text-gray-500 text-base max-w-md mx-auto">
                  Select your player and last opponent on the left, then click{" "}
                  <strong>Predict</strong> to see the most likely next opponents
                  powered by machine learning!
                </p>
                {stats && (
                  <div className="mt-6 inline-flex gap-4 text-xs text-gray-400">
                    <span>📁 {stats.matches_loaded} matches</span>
                    <span>🔗 {stats.transition_entries.toLocaleString()} rules</span>
                    <span>🧬 {stats.bigram_entries.toLocaleString()} bigrams</span>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-purple-300/60 text-xs">
          <p>
            Powered by Markov-chain machine learning · Magic Chess Go Go Opponent Predictor v2.0
          </p>
        </footer>
      </div>
    </div>
  );
}

