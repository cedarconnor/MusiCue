import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import {
  AnalysisJSON,
  getPeaks,
  sourceAudioUrl,
  stemAudioUrl,
} from "../lib/api";
import { OVERLAY_HEIGHT, drawAnalysisLayer } from "./AnalysisOverlay";
import OnsetMarkers, { Stem as OverlayStem } from "./OnsetMarkers";
import PhraseBlocks from "./PhraseBlocks";
import { SelectedAnnotation } from "./LabelChipStrip";

interface Props {
  songId: string;
  analysisId: string;
  analysis: AnalysisJSON;
  onReady?: (ws: WaveSurfer) => void;
  selected?: SelectedAnnotation;
  onSelect?: (sel: SelectedAnnotation) => void;
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
  selected,
  onSelect,
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
  const [pps, setPps] = useState(1);
  const [duration, setDuration] = useState(0);

  function applyZoom(zoomFactor: number) {
    const ws = mixRef.current;
    const host = mixHostRef.current;
    if (!ws || !host) return;
    const dur = ws.getDuration();
    if (!dur) return;
    const fit = Math.max(1, host.clientWidth / dur);
    fitPpsRef.current = fit;
    const nextPps = fit * zoomFactor;
    // Set overlay state first so the OnsetMarkers/PhraseBlocks canvases
    // mount even if a stem WaveSurfer's zoom call below throws (which
    // happens when a stem audio request 404'd: WaveSurfer's zoom raises
    // "No audio loaded" and would otherwise abort the rest of this fn).
    setPps(nextPps);
    setDuration(dur);
    try {
      ws.zoom(nextPps);
    } catch {
      // mix not yet decoded; next ready-fire will retry.
    }
    for (const stem of STEMS) {
      try {
        stemsRef.current[stem]?.zoom(nextPps);
      } catch {
        // stem audio missing or not loaded; skip — overlays still render.
      }
    }
    if (overlayRef.current) {
      const totalWidth = Math.ceil(dur * nextPps);
      overlayRef.current.width = totalWidth;
      overlayRef.current.style.width = `${totalWidth}px`;
      overlayRef.current.height = OVERLAY_HEIGHT;
      drawAnalysisLayer(overlayRef.current, analysis, nextPps);
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

      // Bind the master events synchronously, BEFORE any awaits, so we
      // don't miss the "ready" fire on a fast-decoding mix. (Stems load
      // in parallel below; if we awaited them first, the mix could be
      // ready-and-fired before the listener attached, leaving onReady
      // never called -- which the Transport's play button depends on.)
      let mixReadyFired = false;
      mix.on("ready", () => {
        mixReadyFired = true;
        applyZoom(1);
        onReady?.(mix);
      });

      // Slave-sync handlers. Stems may not exist yet when these fire on
      // the very first play; that's fine -- the optional chaining skips
      // them and the user just hears the mix until the stems show up.
      const syncSlaves = () => {
        const t = mix.getCurrentTime();
        for (const stem of STEMS) {
          const ws = stemsRef.current[stem];
          if (!ws) continue;
          if (Math.abs(ws.getCurrentTime() - t) > 0.04) ws.setTime(t);
        }
      };
      const syncOnPlay = () => {
        for (const stem of STEMS) {
          // play() returns a Promise; swallow rejections so a stem that
          // hasn't decoded yet doesn't surface as an unhandled rejection.
          stemsRef.current[stem]?.play()?.catch?.(() => {});
        }
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

      // Stems load in the background; failures don't block the mix.
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
          // Mute on creation: stems are visualisation slaves; only the
          // mix is audible by default. Without this, all four stems play
          // on top of the mix and every part is doubled.
          ws.setMuted(true);
          stemsRef.current[stem] = ws;
        }),
      );

      // Belt-and-suspenders: if the mix already fired "ready" before our
      // listener attached (only possible if WaveSurfer's listener semantics
      // change in a future version), call onReady manually. Cheap to check.
      if (!mixReadyFired && mix.getDuration()) {
        mixReadyFired = true;
        applyZoom(1);
        onReady?.(mix);
      }
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
            <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
              <div
                ref={(el) => {
                  stemHostRefs.current[stem] = el;
                }}
                style={{ minWidth: 0 }}
              />
              {duration > 0 && analysis.onsets?.[stem] && (
                <OnsetMarkers
                  stem={stem as OverlayStem}
                  onsets={analysis.onsets[stem]}
                  duration={duration}
                  pxPerSec={pps}
                  height={STEM_HEIGHT}
                  drumClasses={stem === "drums"}
                  selectedIdx={
                    selected?.kind === "onset" && selected.stem === stem
                      ? selected.idx
                      : null
                  }
                  onSelect={(idx) =>
                    onSelect?.({
                      kind: "onset",
                      stem: stem as OverlayStem,
                      idx,
                    })
                  }
                />
              )}
              {duration > 0 &&
                (stem === "vocals" || stem === "other") &&
                analysis.phrases?.[stem] && (
                  <PhraseBlocks
                    stem={stem as OverlayStem}
                    phrases={analysis.phrases[stem]}
                    duration={duration}
                    pxPerSec={pps}
                    height={STEM_HEIGHT}
                    selectedIdx={
                      selected?.kind === "phrase" && selected.stem === stem
                        ? selected.idx
                        : null
                    }
                    onSelect={(idx) =>
                      onSelect?.({
                        kind: "phrase",
                        stem: stem as OverlayStem,
                        idx,
                      })
                    }
                  />
                )}
            </div>
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
