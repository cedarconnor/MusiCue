import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import type WaveSurfer from "wavesurfer.js";

export type LoopState = {
  in: number | null;
  out: number | null;
  enabled: boolean;
};

export const EMPTY_LOOP: LoopState = { in: null, out: null, enabled: false };

export function loadLoop(songId: string, analysisId: string): LoopState {
  try {
    const raw = localStorage.getItem(`loop:${songId}:${analysisId}`);
    if (!raw) return { ...EMPTY_LOOP };
    return { ...EMPTY_LOOP, ...(JSON.parse(raw) as LoopState) };
  } catch {
    return { ...EMPTY_LOOP };
  }
}

export function saveLoop(
  songId: string,
  analysisId: string,
  state: LoopState,
): void {
  try {
    localStorage.setItem(
      `loop:${songId}:${analysisId}`,
      JSON.stringify(state),
    );
  } catch {
    // Quota exceeded or storage disabled. Loop is best-effort.
  }
}

export function attachRegions(ws: WaveSurfer): RegionsPlugin {
  return ws.registerPlugin(RegionsPlugin.create());
}

export function applyLoopRegion(
  regions: RegionsPlugin,
  state: LoopState,
): void {
  regions.clearRegions();
  if (state.in == null || state.out == null || state.in >= state.out) return;
  regions.addRegion({
    id: "loop",
    start: state.in,
    end: state.out,
    color: state.enabled
      ? "rgba(80, 160, 255, 0.18)"
      : "rgba(120, 120, 120, 0.10)",
    drag: false,
    resize: false,
  });
}

export function bindLoopKeys(
  ws: WaveSurfer,
  songId: string,
  analysisId: string,
  getState: () => LoopState,
  setState: (next: LoopState) => void,
): () => void {
  const isTextField = (t: EventTarget | null) =>
    t instanceof HTMLElement &&
    (t.matches("input, textarea") || t.isContentEditable);

  const on = (e: KeyboardEvent) => {
    if (isTextField(e.target)) return;
    const t = ws.getCurrentTime();
    const cur = getState();
    let next: LoopState | null = null;
    if (e.key === "i" || e.key === "I") {
      next = { ...cur, in: t, out: cur.out && cur.out > t ? cur.out : null };
    } else if (e.key === "o" || e.key === "O") {
      next = { ...cur, out: t, in: cur.in != null && cur.in < t ? cur.in : null };
    } else if (e.key === "l" || e.key === "L") {
      next = { ...cur, enabled: !cur.enabled };
    } else if (e.key === "Escape") {
      next = { ...EMPTY_LOOP };
    }
    if (next === null) return;
    setState(next);
    saveLoop(songId, analysisId, next);
    e.preventDefault();
  };

  document.addEventListener("keydown", on);
  return () => document.removeEventListener("keydown", on);
}

export function bindLoopWraparound(
  ws: WaveSurfer,
  getState: () => LoopState,
): () => void {
  const onTime = () => {
    const s = getState();
    if (!s.enabled || s.in == null || s.out == null) return;
    if (ws.getCurrentTime() >= s.out) {
      ws.setTime(s.in);
    }
  };
  ws.on("audioprocess", onTime);
  return () => ws.un("audioprocess", onTime);
}
