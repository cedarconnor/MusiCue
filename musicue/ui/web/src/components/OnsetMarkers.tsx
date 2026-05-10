import { useEffect, useRef } from "react";
import { OnsetItem } from "../lib/api";

const DRUM_CLASS_COLORS: Record<string, string> = {
  kick: "#ef4444",
  snare: "#3b82f6",
  hihat: "#22c55e",
  hat: "#22c55e",
  tom: "#a855f7",
  cymbal: "#fbbf24",
  ride: "#06b6d4",
  percussion: "#888",
};

const STEM_TINT: Record<string, string> = {
  drums: "#c87575",
  bass: "#7a9ec5",
  vocals: "#92c576",
  other: "#a98ec0",
};

export type Stem = "drums" | "bass" | "vocals" | "other";

interface Props {
  stem: Stem;
  onsets: OnsetItem[];
  duration: number;
  pxPerSec: number;
  height: number;
  drumClasses?: boolean;
  selectedIdx?: number | null;
  onSelect?: (idx: number) => void;
}

export default function OnsetMarkers({
  stem,
  onsets,
  duration,
  pxPerSec,
  height,
  drumClasses,
  selectedIdx,
  onSelect,
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
    onsets.forEach((o, i) => {
      const x = o.t * pxPerSec;
      const v = Math.max(0.2, Math.min(1, o.strength ?? 0.5));
      const tickH = height * v;
      const cls = drumClasses ? (o.drum_class ?? null) : null;
      const baseColor = cls
        ? (DRUM_CLASS_COLORS[cls] ?? STEM_TINT[stem])
        : STEM_TINT[stem];
      ctx.strokeStyle = i === selectedIdx ? "#FFEB3B" : baseColor;
      ctx.lineWidth = i === selectedIdx ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(x, height - tickH);
      ctx.lineTo(x, height);
      ctx.stroke();
    });
  }, [stem, onsets, duration, pxPerSec, height, drumClasses, selectedIdx]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onSelect || onsets.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = x / pxPerSec;
    let bestI = 0;
    let bestD = Infinity;
    for (let i = 0; i < onsets.length; i++) {
      const d = Math.abs(onsets[i].t - t);
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }
    if (bestD * pxPerSec < 6) onSelect(bestI);
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        height,
        width: Math.ceil(duration * pxPerSec),
        cursor: "pointer",
        pointerEvents: "auto",
      }}
    />
  );
}
