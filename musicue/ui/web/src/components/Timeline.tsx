import { useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import { AnalysisJSON, getPeaks, sourceAudioUrl } from "../lib/api";
import { drawAnalysisLayer, OVERLAY_HEIGHT } from "./AnalysisOverlay";

interface Props {
  songId: string;
  analysisId: string;
  analysis: AnalysisJSON;
  pxPerSec?: number;
  onReady?: (ws: WaveSurfer) => void;
}

export default function Timeline({
  songId,
  analysisId,
  analysis,
  pxPerSec = 100,
  onReady,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsHostRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    let cancelled = false;
    let currentPps = pxPerSec;

    function resizeOverlay() {
      if (!overlayRef.current || !wsRef.current) return;
      const duration = wsRef.current.getDuration();
      overlayRef.current.width = Math.max(1, Math.ceil(duration * currentPps));
      overlayRef.current.height = OVERLAY_HEIGHT;
    }

    (async () => {
      if (!wsHostRef.current) return;
      let channelData: number[] = [];
      try {
        const peaks = await getPeaks(songId, analysisId, "mix");
        if (cancelled) return;
        // peaks.data is interleaved [min0, max0, min1, max1, ...].
        // WaveSurfer v7 wants per-channel arrays; collapse to single channel
        // of absolute peak values.
        for (let i = 0; i < peaks.data.length; i += 2) {
          channelData.push(
            Math.max(Math.abs(peaks.data[i]), Math.abs(peaks.data[i + 1])),
          );
        }
      } catch {
        // No peaks available; WaveSurfer will decode the audio itself
        // (slower, but still works).
      }

      const ws = WaveSurfer.create({
        container: wsHostRef.current,
        height: 96,
        waveColor: "#666",
        progressColor: "#999",
        cursorColor: "#FFEB3B",
        cursorWidth: 2,
        minPxPerSec: pxPerSec,
        peaks: channelData.length ? [channelData] : undefined,
        url: sourceAudioUrl(songId),
      });
      wsRef.current = ws;

      ws.on("ready", () => {
        resizeOverlay();
        if (overlayRef.current) {
          drawAnalysisLayer(overlayRef.current, analysis, currentPps);
        }
        onReady?.(ws);
      });
      ws.on("zoom", (newPps: number) => {
        currentPps = newPps;
        resizeOverlay();
        if (overlayRef.current) {
          drawAnalysisLayer(overlayRef.current, analysis, currentPps);
        }
      });
    })();
    return () => {
      cancelled = true;
      wsRef.current?.destroy();
      wsRef.current = null;
    };
  }, [songId, analysisId]);

  return (
    <div
      ref={containerRef}
      style={{ position: "relative", overflowX: "auto", padding: "0 8px" }}
    >
      <div ref={wsHostRef} style={{ position: "relative" }} />
      <canvas
        ref={overlayRef}
        style={{
          position: "absolute",
          top: 96,
          left: 8,
          height: OVERLAY_HEIGHT,
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
