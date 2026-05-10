import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import WaveSurfer from "wavesurfer.js";
import type RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { AnalysisJSON, Song, getAnalysis, getSong } from "../lib/api";
import Timeline from "../components/Timeline";
import Transport from "../components/Transport";
import MetadataCard from "../components/MetadataCard";
import CurvesPanel from "../components/CurvesPanel";
import ExportModal from "../components/ExportModal";
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
  syncLoopFromServer,
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
  const [curvesCollapsed, setCurvesCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem("curves:collapsed") === "1";
    } catch {
      return false;
    }
  });
  const [cursorTime, setCursorTime] = useState<number>(0);
  const [exportOpen, setExportOpen] = useState<boolean>(false);
  const [clickOn, setClickOn] = useState<boolean>(false);
  const [layout, setLayout] = useState<{ duration: number; pxPerSec: number }>({
    duration: 0,
    pxPerSec: 1,
  });

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
    // Server-canonical refresh: if the DB has a different loop than the
    // localStorage cache, server wins.
    syncLoopFromServer(songId, analysisId).then((srv) => {
      if (srv) setLoop(srv);
    });
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
        showRmsTint={!curvesCollapsed}
        onCursorTime={setCursorTime}
        onLayout={setLayout}
        clickOn={clickOn}
      />
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <Transport
            ws={ws}
            songId={songId}
            analysisId={analysisId}
            clickOn={clickOn}
            onClickOnChange={setClickOn}
          />
        </div>
        <button
          onClick={() => setExportOpen(true)}
          style={{
            background: "#1a1a1a",
            color: "#bbb",
            border: "1px solid #333",
            padding: "6px 14px",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: 13,
            marginRight: 16,
          }}
        >
          Export ▶
        </button>
      </div>
      <ExportModal
        open={exportOpen}
        songId={songId}
        analysisId={analysisId}
        songTitle={song?.title ?? "cuesheet"}
        onClose={() => setExportOpen(false)}
      />
      <CurvesPanel
        analysis={analysis}
        songId={songId}
        analysisId={analysisId}
        duration={layout.duration}
        pxPerSec={layout.pxPerSec}
        cursorTime={cursorTime}
        collapsed={curvesCollapsed}
        onToggleCollapse={() => {
          setCurvesCollapsed((c) => {
            const next = !c;
            try {
              localStorage.setItem("curves:collapsed", next ? "1" : "0");
            } catch {
              // ignore
            }
            return next;
          });
        }}
      />
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
