import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import WaveSurfer from "wavesurfer.js";
import type RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { AnalysisJSON, Song, getAnalysis, getSong } from "../lib/api";
import Timeline from "../components/Timeline";
import Transport from "../components/Transport";
import MetadataCard from "../components/MetadataCard";
import LabelChipStrip, {
  SelectedAnnotation,
} from "../components/LabelChipStrip";
import {
  EMPTY_LOOP,
  LoopState,
  applyLoopRegion,
  attachRegions,
  bindLoopKeys,
  bindLoopWraparound,
  loadLoop,
} from "../lib/loop";

export default function Editor() {
  const { songId, analysisId } = useParams<{
    songId: string;
    analysisId: string;
  }>();
  const [song, setSong] = useState<Song | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisJSON | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ws, setWs] = useState<WaveSurfer | null>(null);

  const [loop, setLoop] = useState<LoopState>(EMPTY_LOOP);
  const loopRef = useRef<LoopState>(EMPTY_LOOP);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const [selected, setSelected] = useState<SelectedAnnotation>(null);

  // Keep loopRef in sync so bindLoopKeys/bindLoopWraparound (which read via
  // a stable getter) always see the latest state.
  useEffect(() => {
    loopRef.current = loop;
  }, [loop]);

  useEffect(() => {
    if (!songId || !analysisId) return;
    setSong(null);
    setAnalysis(null);
    setError(null);
    setLoop(loadLoop(songId, analysisId));
    Promise.all([getSong(songId), getAnalysis(songId, analysisId)])
      .then(([s, a]) => {
        setSong(s);
        setAnalysis(a);
      })
      .catch((e) => setError(String(e)));
  }, [songId, analysisId]);

  // Attach the regions plugin + key bindings + wraparound once we have the
  // master WaveSurfer (mix lane) and the route params.
  useEffect(() => {
    if (!ws || !songId || !analysisId) return;
    const regions = attachRegions(ws);
    regionsRef.current = regions;
    applyLoopRegion(regions, loopRef.current);
    const offKeys = bindLoopKeys(
      ws,
      songId,
      analysisId,
      () => loopRef.current,
      setLoop,
    );
    const offWrap = bindLoopWraparound(ws, () => loopRef.current);
    return () => {
      offKeys();
      offWrap();
      regionsRef.current = null;
    };
  }, [ws, songId, analysisId]);

  // Re-render the region whenever loop state changes.
  useEffect(() => {
    if (regionsRef.current) applyLoopRegion(regionsRef.current, loop);
  }, [loop]);

  if (!songId || !analysisId)
    return <div style={{ padding: 24 }}>Missing route params</div>;
  if (error)
    return <div style={{ padding: 24, color: "#f88" }}>Error: {error}</div>;
  if (!analysis) return <div style={{ padding: 24 }}>Loading analysis…</div>;

  return (
    <div>
      <MetadataCard song={song} analysis={analysis} />
      <LabelChipStrip selected={selected} analysis={analysis} />
      <Timeline
        songId={songId}
        analysisId={analysisId}
        analysis={analysis}
        onReady={setWs}
        selected={selected}
        onSelect={setSelected}
      />
      <Transport ws={ws} songId={songId} analysisId={analysisId} />
      <div
        style={{
          padding: "4px 16px",
          color: "#666",
          fontSize: 11,
          fontFamily: "monospace",
        }}
      >
        Space play/pause · I loop-in · O loop-out · L toggle loop · Esc clear
      </div>
    </div>
  );
}
