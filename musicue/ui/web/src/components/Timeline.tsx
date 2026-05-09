import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { AnalysisJSON, getPeaks, sourceAudioUrl } from "../lib/api";
import { OVERLAY_HEIGHT, drawAnalysisLayer } from "./AnalysisOverlay";

interface Props {
  songId: string;
  analysisId: string;
  analysis: AnalysisJSON;
  onReady?: (ws: WaveSurfer) => void;
}

const WAVE_HEIGHT = 96;

export default function Timeline({
  songId,
  analysisId,
  analysis,
  onReady,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsHostRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const fitPpsRef = useRef<number>(1);
  const [zoom, setZoom] = useState(1);

  // Apply current zoom (relative to fit-to-width baseline) to WaveSurfer +
  // resize the overlay canvas + redraw analysis at the new px/sec.
  function applyZoom(zoomFactor: number) {
    const ws = wsRef.current;
    const host = wsHostRef.current;
    if (!ws || !host) return;
    const duration = ws.getDuration();
    if (!duration) return;
    const fit = Math.max(1, host.clientWidth / duration);
    fitPpsRef.current = fit;
    const pps = fit * zoomFactor;
    ws.zoom(pps);
    if (overlayRef.current) {
      const totalWidth = Math.ceil(duration * pps);
      overlayRef.current.width = totalWidth;
      overlayRef.current.style.width = `${totalWidth}px`;
      overlayRef.current.height = OVERLAY_HEIGHT;
      drawAnalysisLayer(overlayRef.current, analysis, pps);
    }
  }

  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (!wsHostRef.current) return;
      let channelData: number[] = [];
      try {
        const peaks = await getPeaks(songId, analysisId, "mix");
        if (cancelled) return;
        for (let i = 0; i < peaks.data.length; i += 2) {
          channelData.push(
            Math.max(Math.abs(peaks.data[i]), Math.abs(peaks.data[i + 1])),
          );
        }
      } catch {
        // No peaks available; WaveSurfer will decode the audio itself.
      }

      const ws = WaveSurfer.create({
        container: wsHostRef.current,
        height: WAVE_HEIGHT,
        waveColor: "#666",
        progressColor: "#999",
        cursorColor: "#FFEB3B",
        cursorWidth: 2,
        // Start tiny -- we'll zoom to fit-to-width once duration is known.
        minPxPerSec: 1,
        // Don't auto-scroll; we want the whole song visible by default.
        autoScroll: false,
        peaks: channelData.length ? [channelData] : undefined,
        url: sourceAudioUrl(songId),
      });
      wsRef.current = ws;

      ws.on("ready", () => {
        applyZoom(1);
        onReady?.(ws);
      });
    })();

    const onResize = () => applyZoom(zoom);
    window.addEventListener("resize", onResize);

    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      wsRef.current?.destroy();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [songId, analysisId]);

  // Re-apply zoom whenever the slider changes (without re-creating WaveSurfer).
  useEffect(() => {
    if (wsRef.current && wsRef.current.getDuration()) applyZoom(zoom);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoom]);

  return (
    <div ref={containerRef}>
      {/* Inner scroll container: at zoom > 1x the waveform exceeds viewport
          width, but only THIS area scrolls horizontally -- the page (and
          the Transport below) stays put. */}
      <div style={{ position: "relative", overflowX: "auto", overflowY: "hidden" }}>
        <div ref={wsHostRef} />
        <canvas
          ref={overlayRef}
          style={{
            position: "absolute",
            top: WAVE_HEIGHT,
            left: 0,
            height: OVERLAY_HEIGHT,
            pointerEvents: "none",
          }}
        />
        {/* Spacer so the absolutely-positioned overlay reserves vertical
            space in the parent layout. */}
        <div style={{ height: OVERLAY_HEIGHT }} />
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "4px 16px",
          color: "#888",
          fontSize: 12,
        }}
      >
        <label>Zoom</label>
        <input
          type="range"
          min={1}
          max={20}
          step={0.1}
          value={zoom}
          onChange={(e) => setZoom(parseFloat(e.target.value))}
          style={{ flex: 1, maxWidth: 320 }}
        />
        <span style={{ fontFamily: "monospace", minWidth: 56 }}>
          {zoom.toFixed(1)}x
        </span>
      </div>
    </div>
  );
}
