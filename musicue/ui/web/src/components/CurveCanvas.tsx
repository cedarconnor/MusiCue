import { useEffect, useRef } from "react";

interface Props {
  values: number[];
  hopSec: number;
  duration: number;
  pxPerSec: number;
  height: number;
  yRange?: [number, number];
  cursorTime?: number;
  color?: string;
}

export default function CurveCanvas({
  values,
  hopSec,
  duration,
  pxPerSec,
  height,
  yRange,
  cursorTime,
  color = "#9cf",
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = Math.max(1, Math.ceil(duration * pxPerSec));
    canvas.width = w;
    canvas.style.width = `${w}px`;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, height);
    if (values.length === 0 || hopSec <= 0) return;

    let lo: number;
    let hi: number;
    if (yRange) {
      [lo, hi] = yRange;
    } else {
      lo = Infinity;
      hi = -Infinity;
      for (const v of values) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (lo === hi) {
        hi = lo + 1;
      }
    }
    const span = hi - lo;

    if (lo < 0 && hi > 0) {
      const yZero = height - ((0 - lo) / span) * height;
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, yZero);
      ctx.lineTo(w, yZero);
      ctx.stroke();
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < values.length; i++) {
      const t = i * hopSec;
      if (t > duration) break;
      const x = t * pxPerSec;
      const v = values[i];
      const norm = (v - lo) / span;
      const y = height - Math.max(0, Math.min(1, norm)) * height;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    if (cursorTime != null && cursorTime >= 0 && cursorTime <= duration) {
      const cx = cursorTime * pxPerSec;
      ctx.strokeStyle = "rgba(255,235,59,0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, height);
      ctx.stroke();
    }
  }, [values, hopSec, duration, pxPerSec, height, yRange, cursorTime, color]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: "block",
        width: Math.ceil(duration * pxPerSec),
        height,
      }}
    />
  );
}
