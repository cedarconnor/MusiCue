import { useState } from "react";
import { useReadiness } from "../hooks/useReadiness";
import { chipSummary } from "../lib/readiness";
import ReadinessPopover from "./ReadinessPopover";

const BG: Record<"green" | "amber" | "red", string> = {
  green: "#1e8e3e",
  amber: "#c97a00",
  red: "#c5221f",
};

export default function ReadinessChip() {
  const { report, error, loading, refresh } = useReadiness();
  const [open, setOpen] = useState(false);

  let label: string;
  let color: "green" | "amber" | "red";
  if (error) {
    label = "status unavailable";
    color = "red";
  } else if (!report) {
    label = loading ? "checking…" : "—";
    color = "amber";
  } else {
    const s = chipSummary(report);
    label = s.label;
    color = s.color;
  }

  return (
    <div style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title="System readiness"
        style={{
          background: BG[color],
          color: "white",
          border: "none",
          borderRadius: 12,
          padding: "4px 10px",
          fontSize: 12,
          cursor: "pointer",
          fontWeight: 500,
        }}
      >
        ● {label}
      </button>
      {open && report && (
        <ReadinessPopover
          report={report}
          loading={loading}
          onClose={() => setOpen(false)}
          onRefresh={refresh}
        />
      )}
    </div>
  );
}
