import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import WaveSurfer from "wavesurfer.js";
import { AnalysisJSON, getAnalysis } from "../lib/api";
import Timeline from "../components/Timeline";
import Transport from "../components/Transport";

export default function Editor() {
  const { songId, analysisId } = useParams<{ songId: string; analysisId: string }>();
  const [analysis, setAnalysis] = useState<AnalysisJSON | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ws, setWs] = useState<WaveSurfer | null>(null);

  useEffect(() => {
    if (!songId || !analysisId) return;
    setAnalysis(null);
    setError(null);
    getAnalysis(songId, analysisId)
      .then(setAnalysis)
      .catch((e) => setError(String(e)));
  }, [songId, analysisId]);

  if (!songId || !analysisId) return <div style={{ padding: 24 }}>Missing route params</div>;
  if (error) return <div style={{ padding: 24, color: "#f88" }}>Error: {error}</div>;
  if (!analysis) return <div style={{ padding: 24 }}>Loading analysis…</div>;

  return (
    <div>
      <div
        style={{
          padding: "8px 16px",
          borderBottom: "1px solid #333",
          display: "flex",
          alignItems: "center",
          gap: 16,
        }}
      >
        <Link to="/library">◀ Library</Link>
        <span style={{ color: "#888" }}>
          {analysis.tempo?.bpm
            ? `${Math.round(analysis.tempo.bpm)} BPM`
            : ""}
        </span>
      </div>
      <Timeline
        songId={songId}
        analysisId={analysisId}
        analysis={analysis}
        onReady={setWs}
      />
      <Transport ws={ws} songId={songId} analysisId={analysisId} />
    </div>
  );
}
