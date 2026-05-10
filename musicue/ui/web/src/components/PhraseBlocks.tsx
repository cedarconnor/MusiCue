import { useEffect, useRef } from "react";
import { PhraseItem } from "../lib/api";
import { Stem } from "./OnsetMarkers";

const STEM_TINT: Record<Stem, string> = {
  drums: "#c87575",
  bass: "#7a9ec5",
  vocals: "#92c576",
  other: "#a98ec0",
};

interface Props {
  stem: Stem;
  phrases: PhraseItem[];
  duration: number;
  pxPerSec: number;
  height: number;
  selectedIdx?: number | null;
  onSelect?: (idx: number) => void;
}

export default function PhraseBlocks({
  stem,
  phrases,
  duration,
  pxPerSec,
  height,
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
    const tint = STEM_TINT[stem];
    phrases.forEach((p, i) => {
      const x0 = p.t_start * pxPerSec;
      const x1 = p.t_end * pxPerSec;
      const isSel = i === selectedIdx;
      ctx.fillStyle = isSel ? `${tint}66` : `${tint}2e`;
      ctx.fillRect(x0, 0, x1 - x0, height);
      ctx.strokeStyle = isSel ? "#FFEB3B" : tint;
      ctx.lineWidth = isSel ? 2 : 1;
      ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, height - 1);
    });
  }, [stem, phrases, duration, pxPerSec, height, selectedIdx]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onSelect || phrases.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = x / pxPerSec;
    const idx = phrases.findIndex((p) => p.t_start <= t && t <= p.t_end);
    if (idx >= 0) onSelect(idx);
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
