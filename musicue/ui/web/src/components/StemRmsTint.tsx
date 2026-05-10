import { useEffect, useRef } from "react";
import { Stem } from "./OnsetMarkers";

const STEM_TINT: Record<Stem, string> = {
  drums: "#c87575",
  bass: "#7a9ec5",
  vocals: "#92c576",
  other: "#a98ec0",
};

interface Props {
  stem: Stem;
  values: number[];
  hopSec: number;
  duration: number;
  pxPerSec: number;
  height: number;
  alphaCap?: number;
}

function hexToRgb(hex: string): [number, number, number] {
  const m = hex.replace("#", "");
  return [
    parseInt(m.slice(0, 2), 16),
    parseInt(m.slice(2, 4), 16),
    parseInt(m.slice(4, 6), 16),
  ];
}

export default function StemRmsTint({
  stem,
  values,
  hopSec,
  duration,
  pxPerSec,
  height,
  alphaCap = 0.15,
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

    let max = 0;
    for (const v of values) if (v > max) max = v;
    if (max === 0) return;

    const [r, g, b] = hexToRgb(STEM_TINT[stem]);
    for (let x = 0; x < w; x++) {
      const t = x / pxPerSec;
      const idx = Math.min(values.length - 1, Math.max(0, Math.floor(t / hopSec)));
      const norm = values[idx] / max;
      const alpha = Math.min(alphaCap, norm * alphaCap);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.fillRect(x, 0, 1, height);
    }
  }, [stem, values, hopSec, duration, pxPerSec, height, alphaCap]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        height,
        width: Math.ceil(duration * pxPerSec),
        pointerEvents: "none",
      }}
    />
  );
}
