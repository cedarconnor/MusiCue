import { CSSProperties, useState } from "react";
import {
  ExportFormat,
  ExportGrammar,
  exportCuesheet,
} from "../lib/exportApi";

const FORMATS: Array<{ key: ExportFormat; label: string; group?: string }> = [
  { key: "csv", label: "CSV", group: "Data" },
  { key: "json", label: "JSON", group: "Data" },
  { key: "midi", label: "MIDI (.mid)", group: "Music" },
  { key: "after_effects", label: "After Effects (.jsx)", group: "Motion graphics" },
  { key: "touchdesigner", label: "TouchDesigner (CHOP CSV)", group: "Real-time" },
  { key: "osc", label: "OSC (JSON bundle)", group: "Real-time" },
  { key: "houdini", label: "Houdini CHOP CSV", group: "VFX" },
  { key: "disguise", label: "disguise cue list", group: "Show control" },
  { key: "unreal", label: "Unreal Sequencer (JSON)", group: "Real-time" },
  { key: "edl", label: "EDL (CMX 3600)", group: "Editorial" },
  { key: "fcpxml", label: "FCPXML (.fcpxml)", group: "Editorial" },
  { key: "premiere_markers", label: "Premiere markers (CSV)", group: "Editorial" },
  { key: "resolve_markers", label: "Resolve markers (CSV)", group: "Editorial" },
];

const EDITORIAL_FORMATS: ReadonlySet<ExportFormat> = new Set([
  "edl",
  "fcpxml",
  "premiere_markers",
  "resolve_markers",
]);

type MarkerSource = "section" | "transition" | "impulse" | "envelope";

const ALL_MARKER_SOURCES: ReadonlyArray<{ key: MarkerSource; label: string }> = [
  { key: "section", label: "Sections" },
  { key: "transition", label: "Transitions" },
  { key: "impulse", label: "Impulses (drum hits, etc.)" },
  { key: "envelope", label: "Envelopes (phrases)" },
];

const GRAMMARS: Array<{ key: ExportGrammar; label: string }> = [
  { key: "concert_visuals", label: "Concert visuals" },
  { key: "character_animation", label: "Character animation" },
  { key: "lighting", label: "Lighting" },
  { key: "camera_edit", label: "Camera edit" },
];

// FPS now applies to every export (it's the cuesheet animation rate). The
// drop-frame toggle is meaningful only at 29.97 / 59.94.
const DROP_FRAME_FPS_TOLERANCE = 0.05;
function dropFrameSupported(fps: number): boolean {
  return (
    Math.abs(fps - 29.97) < DROP_FRAME_FPS_TOLERANCE ||
    Math.abs(fps - 59.94) < DROP_FRAME_FPS_TOLERANCE
  );
}

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
  const [dropFrame, setDropFrame] = useState<boolean>(false);
  const [ticks, setTicks] = useState<number>(480);
  const [oscHost, setOscHost] = useState<string>("127.0.0.1");
  const [oscPort, setOscPort] = useState<number>(9000);
  const [markerSources, setMarkerSources] = useState<Set<MarkerSource>>(
    new Set(["section", "transition"]),
  );
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
        fps,
        drop_frame: dropFrameSupported(fps) && dropFrame,
        ticks_per_beat: format === "midi" ? ticks : undefined,
        osc_host: format === "osc" ? oscHost : undefined,
        osc_port: format === "osc" ? oscPort : undefined,
        marker_sources: EDITORIAL_FORMATS.has(format)
          ? Array.from(markerSources)
          : undefined,
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
            {Array.from(
              FORMATS.reduce((acc, f) => {
                const g = f.group ?? "Other";
                if (!acc.has(g)) acc.set(g, []);
                acc.get(g)!.push(f);
                return acc;
              }, new Map<string, typeof FORMATS>()),
            ).map(([groupName, opts]) => (
              <optgroup key={groupName} label={groupName}>
                {opts.map((f) => (
                  <option key={f.key} value={f.key}>
                    {f.label}
                  </option>
                ))}
              </optgroup>
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

          <label style={labelStyle}>FPS</label>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <select
              value={String(fps)}
              onChange={(e) => setFps(parseFloat(e.target.value))}
              style={{ ...inputStyle, flex: "0 0 auto" }}
            >
              <option value="23.976">23.976</option>
              <option value="24">24</option>
              <option value="25">25</option>
              <option value="29.97">29.97</option>
              <option value="30">30</option>
              <option value="48">48</option>
              <option value="50">50</option>
              <option value="59.94">59.94</option>
              <option value="60">60</option>
            </select>
            {dropFrameSupported(fps) && (
              <label
                style={{
                  ...labelStyle,
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={dropFrame}
                  onChange={(e) => setDropFrame(e.target.checked)}
                />
                drop-frame
              </label>
            )}
          </div>

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

          {EDITORIAL_FORMATS.has(format) && (
            <>
              <label style={labelStyle}>Markers</label>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {ALL_MARKER_SOURCES.map((s) => (
                  <label
                    key={s.key}
                    style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}
                  >
                    <input
                      type="checkbox"
                      checked={markerSources.has(s.key)}
                      onChange={(e) => {
                        const next = new Set(markerSources);
                        if (e.target.checked) next.add(s.key);
                        else next.delete(s.key);
                        setMarkerSources(next);
                      }}
                    />
                    {s.label}
                  </label>
                ))}
              </div>
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
