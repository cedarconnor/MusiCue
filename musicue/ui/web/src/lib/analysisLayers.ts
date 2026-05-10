import { AnalysisJSON } from "./api";

export const SECTION_H = 16;
export const ONSET_TOP = SECTION_H + 4;
export const ONSET_H = 28;
export const BEAT_TOP = ONSET_TOP + ONSET_H + 4;
export const BEAT_H = 24;
export const OVERLAY_HEIGHT = 80;

const DRUM_CLASS_COLORS: Record<string, string> = {
  kick: "#FF5722",
  snare: "#2196F3",
  hihat: "#9C27B0",
  hat: "#9C27B0",
  tom: "#4CAF50",
  cymbal: "#FFC107",
  ride: "#00BCD4",
  percussion: "#888",
};

const STEM_COLORS: Record<string, string> = {
  drums: "#FF5722",
  bass: "#2196F3",
  vocals: "#4CAF50",
  other: "#9C27B0",
};

export function drawSections(
  ctx: CanvasRenderingContext2D,
  analysis: AnalysisJSON,
  pxPerSec: number,
): void {
  for (const sec of analysis.sections ?? []) {
    const x0 = sec.start * pxPerSec;
    const x1 = sec.end * pxPerSec;
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(x0, 0, x1 - x0, SECTION_H);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, SECTION_H);
    ctx.stroke();
    ctx.fillStyle = "#ddd";
    ctx.font = "10px sans-serif";
    ctx.fillText(sec.label ?? "", x0 + 4, SECTION_H - 4);
  }
}

export function drawMixOnsets(
  ctx: CanvasRenderingContext2D,
  analysis: AnalysisJSON,
  pxPerSec: number,
): void {
  for (const [stem, onsets] of Object.entries(analysis.onsets ?? {})) {
    const stemColor = STEM_COLORS[stem] ?? "#aaa";
    for (const o of onsets) {
      const x = o.t * pxPerSec;
      const cls = o.drum_class as string | undefined;
      ctx.strokeStyle = cls ? (DRUM_CLASS_COLORS[cls] ?? stemColor) : stemColor;
      ctx.lineWidth = 1.5;
      const v = o.strength ?? 0.5;
      const h = ONSET_H * Math.max(0.2, Math.min(1, v));
      ctx.beginPath();
      ctx.moveTo(x, ONSET_TOP + (ONSET_H - h));
      ctx.lineTo(x, ONSET_TOP + ONSET_H);
      ctx.stroke();
    }
  }
}

export function drawBeats(
  ctx: CanvasRenderingContext2D,
  analysis: AnalysisJSON,
  pxPerSec: number,
): void {
  for (const b of analysis.beats ?? []) {
    const x = b.t * pxPerSec;
    ctx.strokeStyle = b.downbeat
      ? "rgba(255,235,59,0.85)"
      : "rgba(255,255,255,0.3)";
    ctx.lineWidth = b.downbeat ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(x, BEAT_TOP);
    ctx.lineTo(x, BEAT_TOP + BEAT_H);
    ctx.stroke();
  }
}

const SHAPE_FN: Record<string, (p: number) => number> = {
  linear: (p) => p,
  ease_in: (p) => p * p,
  ease_out: (p) => 1 - (1 - p) * (1 - p),
  ease_in_out: (p) =>
    p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2,
};

const SECTION_LABEL_TINT: Record<string, string> = {
  intro: "rgba(150, 200, 255, 0.7)",
  verse: "rgba(200, 200, 200, 0.7)",
  chorus: "rgba(255, 200, 100, 0.7)",
  bridge: "rgba(200, 150, 220, 0.7)",
  solo: "rgba(255, 150, 150, 0.7)",
  outro: "rgba(120, 120, 120, 0.7)",
  end: "rgba(120, 120, 120, 0.7)",
};

const RAMP_H = 14;
const RAMP_BASELINE_Y = SECTION_H;

export function drawTransitionRamps(
  ctx: CanvasRenderingContext2D,
  analysis: AnalysisJSON,
  pxPerSec: number,
): void {
  for (const tr of analysis.section_transitions ?? []) {
    const fn = SHAPE_FN[tr.ramp.shape] ?? SHAPE_FN.linear;
    const xStart = tr.ramp.t_start * pxPerSec;
    const xEnd = tr.ramp.t_end * pxPerSec;
    const w = Math.max(1, xEnd - xStart);
    const color = SECTION_LABEL_TINT[tr.to] ?? "rgba(255, 235, 59, 0.7)";

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(xStart, RAMP_BASELINE_Y);
    const N = 16;
    for (let i = 0; i <= N; i++) {
      const p = i / N;
      const x = xStart + w * p;
      const y = RAMP_BASELINE_Y - fn(p) * RAMP_H;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(xEnd, RAMP_BASELINE_Y);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

/** Composite paint: sections, ramps, mix-lane onsets, beats. */
export function drawAllMixLayers(
  canvas: HTMLCanvasElement,
  analysis: AnalysisJSON,
  pxPerSec: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawSections(ctx, analysis, pxPerSec);
  drawTransitionRamps(ctx, analysis, pxPerSec);
  drawMixOnsets(ctx, analysis, pxPerSec);
  drawBeats(ctx, analysis, pxPerSec);
}
