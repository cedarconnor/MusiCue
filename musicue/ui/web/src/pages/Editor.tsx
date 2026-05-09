import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import WaveSurfer from "wavesurfer.js";
import { AnalysisJSON, Song, getAnalysis, getSong } from "../lib/api";
import Timeline from "../components/Timeline";
import Transport from "../components/Transport";
import MetadataCard from "../components/MetadataCard";

export default function Editor() {
  const { songId, analysisId } = useParams<{
    songId: string;
    analysisId: string;
  }>();
  const [song, setSong] = useState<Song | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisJSON | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ws, setWs] = useState<WaveSurfer | null>(null);

  useEffect(() => {
    if (!songId || !analysisId) return;
    setSong(null);
    setAnalysis(null);
    setError(null);
    Promise.all([getSong(songId), getAnalysis(songId, analysisId)])
      .then(([s, a]) => {
        setSong(s);
        setAnalysis(a);
      })
      .catch((e) => setError(String(e)));
  }, [songId, analysisId]);

  if (!songId || !analysisId)
    return <div style={{ padding: 24 }}>Missing route params</div>;
  if (error)
    return <div style={{ padding: 24, color: "#f88" }}>Error: {error}</div>;
  if (!analysis) return <div style={{ padding: 24 }}>Loading analysis…</div>;

  return (
    <div>
      <MetadataCard song={song} analysis={analysis} />
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
