import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import {
  AnalysisJSON,
  getPeaks,
  sourceAudioUrl,
  stemAudioUrl,
} from "../lib/api";
import { OVERLAY_HEIGHT, drawAnalysisLayer } from "./AnalysisOverlay";

interface Props {
  songId: string;
  analysisId: string;
  analysis: AnalysisJSON;
  onReady?: (ws: WaveSurfer) => void;
}

const STEMS = ["drums", "bass", "vocals", "other"] as const;
type Stem = (typeof STEMS)[number];

const MIX_HEIGHT = 96;
const STEM_HEIGHT = 56;

const STEM_COLORS: Record<Stem, { wave: string; progress: string }> = {
  drums: { wave: "#7a4848", progress: "#c87575" },
  bass: { wave: "#4a5e7a", progress: "#7a9ec5" },
  vocals: { wave: "#5a7a4a", progress: "#92c576" },
  other: { wave: "#6e5a7a", progress: "#a98ec0" },
};

async function loadPeaksMono(
  songId: string,
  analysisId: string,
  stem: string,
): Promise<number[][] | undefined> {
  try {
    const peaks = await getPeaks(songId, analysisId, stem);
    const channel: number[] = [];
    // Peaks JSON is min/max pairs; collapse to abs-max for a mono envelope.
    for (let i = 0; i < peaks.data.length; i += 2) {
      channel.push(
        Math.max(Math.abs(peaks.data[i]), Math.abs(peaks.data[i + 1])),
      );
    }
    return channel.length ? [channel] : undefined;
  } catch {
    return undefined;
  }
}

export default function Timeline({
  songId,
  analysisId,
  analysis,
  onReady,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mixHostRef = useRef<HTMLDivElement>(null);
  const stemHostRefs = useRef<Record<Stem, HTMLDivElement | null>>({
    drums: null,
    bass: null,
    vocals: null,
    other: null,
  });
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const mixRef = useRef<WaveSurfer | null>(null);
  const stemsRef = useRef<Record<Stem, WaveSurfer | null>>({
    drums: null,
    bass: null,
    vocals: null,
    other: null,
  });
  const fitPpsRef = useRef<number>(1);
  const [zoom, setZoom] = useState(1);
  const [solo, setSolo] = useState<Stem | null>(null);

  function applyZoom(zoomFactor: number) {
    const ws = mixRef.current;
    const host = mixHostRef.current;
    if (!ws || !host) return;
    const duration = ws.getDuration();
    if (!duration) return;
    const fit = Math.max(1, host.clientWidth / duration);
    fitPpsRef.current = fit;
    const pps = fit * zoomFactor;
    ws.zoom(pps);
    for (const stem of STEMS) {
      stemsRef.current[stem]?.zoom(pps);
    }
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
      if (!mixHostRef.current) return;

      const mixPeaks = await loadPeaksMono(songId, analysisId, "mix");
      if (cancelled) return;

      const mix = WaveSurfer.create({
        container: mixHostRef.current,
        height: MIX_HEIGHT,
        waveColor: "#666",
        progressColor: "#999",
        cursorColor: "#FFEB3B",
        cursorWidth: 2,
        minPxPerSec: 1,
        autoScroll: false,
        peaks: mixPeaks,
        url: sourceAudioUrl(songId),
      });
      mixRef.current = mix;

      // Slave stem lanes: same time domain, no audio playback driven by
      // them. Audio audibility is controlled by mute logic below; visual
      // playhead syncs to the mix's audioprocess.
      await Promise.all(
        STEMS.map(async (stem) => {
          const host = stemHostRefs.current[stem];
          if (!host) return;
          const peaks = await loadPeaksMono(songId, analysisId, stem);
          if (cancelled) return;
          const ws = WaveSurfer.create({
            container: host,
            height: STEM_HEIGHT,
            waveColor: STEM_COLORS[stem].wave,
            progressColor: STEM_COLORS[stem].progress,
            cursorColor: "#FFEB3B",
            cursorWidth: 1,
            minPxPerSec: 1,
            autoScroll: false,
            interact: false,
            peaks,
            url: stemAudioUrl(songId, analysisId, stem),
          });
          // Mute on creation: stems are visualisation slaves; only the mix
          // is audible by default. A solo toggle below flips the audibility.
          // Without this, all four stems play unmuted on top of the mix and
          // the user hears every part doubled.
          ws.setMuted(true);
          stemsRef.current[stem] = ws;
        }),
      );

      const syncSlaves = () => {
        const t = mix.getCurrentTime();
        for (const stem of STEMS) {
          const ws = stemsRef.current[stem];
          if (!ws) continue;
          // Drift-correct only on visible drift; setTime on every frame
          // would be wasteful.
          if (Math.abs(ws.getCurrentTime() - t) > 0.04) ws.setTime(t);
        }
      };
      const syncOnPlay = () => {
        for (const stem of STEMS) stemsRef.current[stem]?.play();
      };
      const syncOnPause = () => {
        for (const stem of STEMS) stemsRef.current[stem]?.pause();
      };
      const syncOnSeek = () => {
        const t = mix.getCurrentTime();
        for (const stem of STEMS) stemsRef.current[stem]?.setTime(t);
      };
      mix.on("audioprocess", syncSlaves);
      mix.on("play", syncOnPlay);
      mix.on("pause", syncOnPause);
      mix.on("seeking", syncOnSeek);

      mix.on("ready", () => {
        applyZoom(1);
        onReady?.(mix);
      });
    })();

    const onResize = () => applyZoom(zoom);
    window.addEventListener("resize", onResize);

    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      mixRef.current?.destroy();
      mixRef.current = null;
      for (const stem of STEMS) {
        stemsRef.current[stem]?.destroy();
        stemsRef.current[stem] = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [songId, analysisId]);

  // Re-apply zoom whenever the slider changes (without re-creating WaveSurfer).
  useEffect(() => {
    if (mixRef.current && mixRef.current.getDuration()) applyZoom(zoom);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoom]);

  // Solo logic: when no solo, mix is audible and stems are muted.
  // When a stem is soloed, that stem is audible and mix + other stems mute.
  useEffect(() => {
    const mix = mixRef.current;
    if (mix) mix.setMuted(solo !== null);
    for (const stem of STEMS) {
      const ws = stemsRef.current[stem];
      if (!ws) continue;
      ws.setMuted(solo !== stem);
    }
  }, [solo]);

  return (
    <div ref={containerRef}>
      <div
        style={{ position: "relative", overflowX: "auto", overflowY: "hidden" }}
      >
        <div ref={mixHostRef} />
        <canvas
          ref={overlayRef}
          style={{
            position: "absolute",
            top: MIX_HEIGHT,
            left: 0,
            height: OVERLAY_HEIGHT,
            pointerEvents: "none",
          }}
        />
        <div style={{ height: OVERLAY_HEIGHT }} />
        {STEMS.map((stem) => (
          <div
            key={stem}
            style={{
              display: "flex",
              alignItems: "stretch",
              borderTop: "1px solid #222",
            }}
          >
            <div
              style={{
                width: 80,
                padding: "4px 8px",
                background: "#181818",
                color: "#bbb",
                fontSize: 12,
                fontFamily: "monospace",
                display: "flex",
                flexDirection: "column",
                gap: 4,
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <div>{stem}</div>
              <button
                onClick={() => setSolo(solo === stem ? null : stem)}
                style={{
                  fontSize: 10,
                  padding: "2px 6px",
                  background: solo === stem ? "#5a7a4a" : "#2a2a2a",
                  color: solo === stem ? "#fff" : "#bbb",
                  border: "1px solid #333",
                  cursor: "pointer",
                }}
              >
                {solo === stem ? "soloed" : "solo"}
              </button>
            </div>
            <div
              ref={(el) => {
                stemHostRefs.current[stem] = el;
              }}
              style={{ flex: 1, minWidth: 0 }}
            />
          </div>
        ))}
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
