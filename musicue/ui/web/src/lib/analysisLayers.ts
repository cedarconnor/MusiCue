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

/** Composite paint: sections, mix-lane onsets, beats. v0.1c adds drawTransitionRamps in Task 4.1. */
export function drawAllMixLayers(
  canvas: HTMLCanvasElement,
  analysis: AnalysisJSON,
  pxPerSec: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawSections(ctx, analysis, pxPerSec);
  drawMixOnsets(ctx, analysis, pxPerSec);
  drawBeats(ctx, analysis, pxPerSec);
}
