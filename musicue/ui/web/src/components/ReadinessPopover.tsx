import type { ComponentStatus, ReadinessReport } from "../lib/readiness";

const STATE_DOT: Record<ComponentStatus["state"], string> = {
  ready: "#1e8e3e",
  degraded: "#c97a00",
  missing: "#c5221f",
  error: "#c5221f",
};

const STATE_LABEL: Record<ComponentStatus["state"], string> = {
  ready: "Ready",
  degraded: "Degraded",
  missing: "Missing",
  error: "Error",
};

export default function ReadinessPopover(props: {
  report: ReadinessReport;
  loading: boolean;
  onClose: () => void;
  onRefresh: () => Promise<void>;
}) {
  const { report, loading, onClose, onRefresh } = props;
  return (
    <div
      role="dialog"
      style={{
        position: "absolute",
        top: 32,
        right: 0,
        zIndex: 1000,
        background: "white",
        color: "#222",
        border: "1px solid #ccc",
        borderRadius: 8,
        boxShadow: "0 6px 24px rgba(0,0,0,0.18)",
        width: 420,
        maxHeight: 480,
        overflowY: "auto",
        padding: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 8,
        }}
      >
        <strong>System readiness</strong>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          style={{ border: "none", background: "transparent", cursor: "pointer" }}
        >
          ✕
        </button>
      </div>

      <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {report.components.map((c) => (
          <li
            key={c.name}
            style={{
              padding: "6px 0",
              borderBottom: "1px solid #eee",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  background: STATE_DOT[c.state],
                  display: "inline-block",
                }}
              />
              <span style={{ fontWeight: 600 }}>{c.name}</span>
              <span style={{ color: "#666", fontSize: 12 }}>
                {STATE_LABEL[c.state]}
                {c.version ? ` · v${c.version}` : ""}
                {c.required ? "" : " · optional"}
              </span>
            </div>
            {c.detail && (
              <div style={{ fontSize: 12, color: "#444", marginTop: 2 }}>
                {c.detail}
              </div>
            )}
            {c.remediation && (
              <div
                style={{
                  fontSize: 12,
                  marginTop: 4,
                  fontFamily: "monospace",
                  background: "#f4f4f4",
                  padding: "3px 6px",
                  borderRadius: 4,
                }}
              >
                {c.remediation}
              </div>
            )}
          </li>
        ))}
      </ul>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 8,
        }}
      >
        <span style={{ fontSize: 11, color: "#888" }}>
          Re-run install.bat to fix missing items.
        </span>
        <button
          type="button"
          onClick={() => void onRefresh()}
          disabled={loading}
          style={{
            border: "1px solid #ccc",
            background: "white",
            borderRadius: 4,
            padding: "3px 10px",
            cursor: loading ? "wait" : "pointer",
            fontSize: 12,
          }}
        >
          {loading ? "Checking…" : "Recheck"}
        </button>
      </div>
    </div>
  );
}
