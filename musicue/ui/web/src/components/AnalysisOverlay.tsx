import { AnalysisJSON } from "../lib/api";

const DRUM_COLORS: Record<string, string> = {
  kick: "#FF5722",
  snare: "#2196F3",
  hihat: "#9C27B0",
  hat: "#9C27B0",
  tom: "#4CAF50",
  cymbal: "#FFC107",
  ride: "#00BCD4",
  percussion: "#888",
};

export const OVERLAY_HEIGHT = 80;

export function drawAnalysisLayer(
  canvas: HTMLCanvasElement,
  analysis: AnalysisJSON,
  pxPerSec: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const SECTION_H = 16;
  const ONSET_TOP = SECTION_H + 4;
  const ONSET_H = 28;
  const BEAT_TOP = ONSET_TOP + ONSET_H + 4;
  const BEAT_H = 24;

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

  const drumOnsets = analysis.onsets?.drums ?? [];
  for (const o of drumOnsets) {
    const x = o.t * pxPerSec;
    const cls = (o.drum_class as string) ?? "percussion";
    ctx.strokeStyle = DRUM_COLORS[cls] ?? "#aaa";
    ctx.lineWidth = 1.5;
    const v = o.strength ?? 0.5;
    const h = ONSET_H * Math.max(0.2, Math.min(1, v));
    ctx.beginPath();
    ctx.moveTo(x, ONSET_TOP + (ONSET_H - h));
    ctx.lineTo(x, ONSET_TOP + ONSET_H);
    ctx.stroke();
  }

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
