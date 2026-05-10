import { CSSProperties, useState } from "react";
import {
  ExportFormat,
  ExportGrammar,
  exportCuesheet,
} from "../lib/exportApi";

const FORMATS: Array<{ key: ExportFormat; label: string }> = [
  { key: "csv", label: "CSV" },
  { key: "json", label: "JSON" },
  { key: "midi", label: "MIDI (.mid)" },
  { key: "after_effects", label: "After Effects (.jsx)" },
  { key: "touchdesigner", label: "TouchDesigner (CHOP CSV)" },
  { key: "osc", label: "OSC (JSON bundle)" },
  { key: "houdini", label: "Houdini CHOP CSV" },
  { key: "disguise", label: "disguise cue list" },
  { key: "unreal", label: "Unreal Sequencer (JSON)" },
];

const GRAMMARS: Array<{ key: ExportGrammar; label: string }> = [
  { key: "concert_visuals", label: "Concert visuals" },
  { key: "character_animation", label: "Character animation" },
  { key: "lighting", label: "Lighting" },
  { key: "camera_edit", label: "Camera edit" },
];

const FPS_NEEDED: ReadonlySet<ExportFormat> = new Set([
  "after_effects",
  "disguise",
]);

interface Props {
  open: boolean;
  songId: string;
  analysisId: string;
  songTitle: string;
  onClose: () => void;
}

export default function ExportModal({
  open,
  songId,
  analysisId,
  songTitle,
  onClose,
}: Props) {
  const [format, setFormat] = useState<ExportFormat>("csv");
  const [grammar, setGrammar] = useState<ExportGrammar>("concert_visuals");
  const [filename, setFilename] = useState<string>(songTitle);
  const [fps, setFps] = useState<number>(24);
  const [ticks, setTicks] = useState<number>(480);
  const [oscHost, setOscHost] = useState<string>("127.0.0.1");
  const [oscPort, setOscPort] = useState<number>(9000);
  const [busy, setBusy] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);

  if (!open) return null;

  const handleExport = async () => {
    setBusy(true);
    setErr(null);
    try {
      await exportCuesheet(songId, analysisId, {
        format,
        grammar,
        filename,
        fps: FPS_NEEDED.has(format) ? fps : undefined,
        ticks_per_beat: format === "midi" ? ticks : undefined,
        osc_host: format === "osc" ? oscHost : undefined,
        osc_port: format === "osc" ? oscPort : undefined,
      });
      onClose();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div onClick={onClose} style={overlayStyle}>
      <div onClick={(e) => e.stopPropagation()} style={panelStyle}>
        <div style={{ fontSize: 16, marginBottom: 14, color: "#fff" }}>
          Export cuesheet
        </div>
        <div style={gridStyle}>
          <label style={labelStyle}>Format</label>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as ExportFormat)}
            style={inputStyle}
          >
            {FORMATS.map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>

          <label style={labelStyle}>Grammar</label>
          <select
            value={grammar}
            onChange={(e) => setGrammar(e.target.value as ExportGrammar)}
            style={inputStyle}
          >
            {GRAMMARS.map((g) => (
              <option key={g.key} value={g.key}>
                {g.label}
              </option>
            ))}
          </select>

          <label style={labelStyle}>Filename</label>
          <input
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            style={inputStyle}
          />

          {FPS_NEEDED.has(format) && (
            <>
              <label style={labelStyle}>FPS</label>
              <input
                type="number"
                min={1}
                max={120}
                step={1}
                value={fps}
                onChange={(e) => setFps(parseFloat(e.target.value) || 24)}
                style={inputStyle}
              />
            </>
          )}

          {format === "midi" && (
            <>
              <label style={labelStyle}>Ticks/beat</label>
              <select
                value={ticks}
                onChange={(e) => setTicks(parseInt(e.target.value, 10))}
                style={inputStyle}
              >
                <option value={480}>480</option>
                <option value={960}>960</option>
                <option value={1920}>1920</option>
              </select>
            </>
          )}

          {format === "osc" && (
            <>
              <label style={labelStyle}>OSC host</label>
              <input
                value={oscHost}
                onChange={(e) => setOscHost(e.target.value)}
                style={inputStyle}
              />
              <label style={labelStyle}>OSC port</label>
              <input
                type="number"
                min={1}
                max={65535}
                value={oscPort}
                onChange={(e) =>
                  setOscPort(parseInt(e.target.value, 10) || 9000)
                }
                style={inputStyle}
              />
            </>
          )}
        </div>

        {err && (
          <div style={{ marginTop: 14, color: "#f88", fontSize: 12 }}>
            {err}
          </div>
        )}

        <div style={actionsStyle}>
          <button onClick={onClose} disabled={busy} style={btnSecondary}>
            Cancel
          </button>
          <button onClick={handleExport} disabled={busy} style={btnPrimary}>
            {busy ? "Exporting…" : "Export ▶"}
          </button>
        </div>
      </div>
    </div>
  );
}

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.6)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
};

const panelStyle: CSSProperties = {
  background: "#161616",
  border: "1px solid #333",
  borderRadius: 6,
  padding: 20,
  minWidth: 480,
  color: "#ddd",
  fontSize: 13,
};

const gridStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "120px 1fr",
  gap: 10,
  alignItems: "center",
};

const labelStyle: CSSProperties = { color: "#bbb" };

const inputStyle: CSSProperties = {
  background: "#1a1a1a",
  color: "#eee",
  border: "1px solid #333",
  padding: "5px 8px",
  borderRadius: 4,
  fontSize: 13,
};

const actionsStyle: CSSProperties = {
  marginTop: 18,
  display: "flex",
  justifyContent: "flex-end",
  gap: 8,
};

const btnPrimary: CSSProperties = {
  background: "#3a5a8c",
  color: "#fff",
  border: "1px solid #5a7ab0",
  padding: "6px 16px",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 13,
};

const btnSecondary: CSSProperties = {
  background: "#1a1a1a",
  color: "#bbb",
  border: "1px solid #333",
  padding: "6px 16px",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 13,
};
