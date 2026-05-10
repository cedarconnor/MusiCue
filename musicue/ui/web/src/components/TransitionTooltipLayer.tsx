import { useState } from "react";
import { SectionTransition } from "../lib/api";
import { SECTION_H } from "../lib/analysisLayers";

interface Props {
  transitions: SectionTransition[];
  duration: number;
  pxPerSec: number;
  forwardClickTo?: HTMLElement | null;
}

interface HoverState {
  idx: number;
  x: number;
  y: number;
}

export default function TransitionTooltipLayer({
  transitions,
  duration,
  pxPerSec,
  forwardClickTo,
}: Props) {
  const [hover, setHover] = useState<HoverState | null>(null);

  const handleClick = (e: React.MouseEvent) => {
    if (!forwardClickTo) return;
    const evt = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
      clientX: e.clientX,
      clientY: e.clientY,
    });
    forwardClickTo.dispatchEvent(evt);
  };

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        width: Math.ceil(duration * pxPerSec),
        height: SECTION_H,
        pointerEvents: "none",
      }}
    >
      {transitions.map((tr, i) => {
        const x = tr.ramp.t_start * pxPerSec;
        const w = Math.max(4, (tr.ramp.t_end - tr.ramp.t_start) * pxPerSec);
        return (
          <div
            key={i}
            onMouseEnter={(e) =>
              setHover({ idx: i, x: e.clientX, y: e.clientY })
            }
            onMouseMove={(e) =>
              setHover({ idx: i, x: e.clientX, y: e.clientY })
            }
            onMouseLeave={() => setHover(null)}
            onClick={handleClick}
            style={{
              position: "absolute",
              left: x,
              top: 0,
              width: w,
              height: SECTION_H,
              pointerEvents: "auto",
              cursor: "pointer",
            }}
          />
        );
      })}
      {hover && (
        <div
          style={{
            position: "fixed",
            left: hover.x + 12,
            top: hover.y + 12,
            background: "#1a1a1a",
            border: "1px solid #444",
            color: "#ddd",
            padding: "6px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontFamily: "monospace",
            zIndex: 100,
            pointerEvents: "none",
          }}
        >
          <div style={{ color: "#fff", fontWeight: 600 }}>
            {transitions[hover.idx].ramp.shape}
          </div>
          {transitions[hover.idx].ramp_evidence?.spectral_flux_rise !== undefined && (
            <div>
              spectral_flux_rise:{" "}
              {transitions[hover.idx].ramp_evidence!.spectral_flux_rise!.toFixed(2)}
            </div>
          )}
          {transitions[hover.idx].ramp_evidence?.lufs_rise_db !== undefined && (
            <div>
              lufs_rise_db:{" "}
              {transitions[hover.idx].ramp_evidence!.lufs_rise_db!.toFixed(2)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
