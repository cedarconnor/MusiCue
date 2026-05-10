import { useEffect, useState } from "react";
import { AnalysisJSON } from "../lib/api";
import CurveCanvas from "./CurveCanvas";

const CURVE_OPTIONS: Array<{ key: string; label: string; color: string }> = [
  { key: "lufs", label: "LUFS", color: "#ffd166" },
  { key: "spectral_centroid", label: "Spectral Centroid", color: "#06d6a0" },
  { key: "spectral_flux", label: "Spectral Flux", color: "#118ab2" },
  { key: "stereo_width", label: "Stereo Width", color: "#ef476f" },
  { key: "stereo_pan", label: "Stereo Pan", color: "#bda4ff" },
];

const FIXED_RANGE: Record<string, [number, number]> = {
  lufs: [-40, 0],
  spectral_centroid: [0, 8000],
  spectral_flux: [0, 1],
  stereo_width: [0, 1],
  stereo_pan: [-1, 1],
};

const CURVE_PANEL_H = 80;

interface Props {
  analysis: AnalysisJSON;
  songId: string;
  analysisId: string;
  duration: number;
  pxPerSec: number;
  cursorTime: number;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function CurvesPanel({
  analysis,
  songId,
  analysisId,
  duration,
  pxPerSec,
  cursorTime,
  collapsed,
  onToggleCollapse,
}: Props) {
  const storageKey = `curve:${songId}:${analysisId}`;
  const [curveKey, setCurveKey] = useState<string>(() => {
    try {
      return localStorage.getItem(storageKey) || "lufs";
    } catch {
      return "lufs";
    }
  });
  const [autoscale, setAutoscale] = useState<boolean>(false);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, curveKey);
    } catch {
      // localStorage disabled / quota; in-memory state still works.
    }
  }, [storageKey, curveKey]);

  const curve = analysis.curves?.[curveKey];
  const opt = CURVE_OPTIONS.find((o) => o.key === curveKey);

  return (
    <div
      style={{
        borderTop: "1px solid #222",
        background: "#0d0d0d",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "6px 12px",
          fontSize: 12,
          color: "#bbb",
        }}
      >
        <button
          onClick={onToggleCollapse}
          style={{
            background: "transparent",
            color: "#bbb",
            border: "none",
            cursor: "pointer",
            fontSize: 12,
            padding: 0,
          }}
        >
          {collapsed ? "▸" : "▾"} Curves
        </button>
        {!collapsed && (
          <>
            <select
              value={curveKey}
              onChange={(e) => setCurveKey(e.target.value)}
              style={{
                background: "#1a1a1a",
                color: "#eee",
                border: "1px solid #333",
                padding: "4px 8px",
                borderRadius: 4,
                fontSize: 12,
              }}
            >
              {CURVE_OPTIONS.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>
            <button
              onClick={() => setAutoscale((a) => !a)}
              style={{
                background: autoscale ? "#3a5a8c" : "#1a1a1a",
                color: autoscale ? "#fff" : "#bbb",
                border: `1px solid ${autoscale ? "#5a7ab0" : "#333"}`,
                borderRadius: 999,
                padding: "2px 10px",
                fontSize: 11,
                cursor: "pointer",
                marginLeft: "auto",
              }}
            >
              {autoscale ? "autoscale" : "fixed range"}
            </button>
          </>
        )}
      </div>
      {!collapsed && (
        <div
          style={{
            overflowX: "auto",
            overflowY: "hidden",
            position: "relative",
          }}
        >
          {curve && curve.values?.length > 0 ? (
            <CurveCanvas
              values={curve.values}
              hopSec={curve.hop_sec}
              duration={duration}
              pxPerSec={pxPerSec}
              height={CURVE_PANEL_H}
              yRange={autoscale ? undefined : FIXED_RANGE[curveKey]}
              cursorTime={cursorTime}
              color={opt?.color}
            />
          ) : (
            <div
              style={{
                height: CURVE_PANEL_H,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#666",
                fontSize: 12,
              }}
            >
              No data for this curve.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
