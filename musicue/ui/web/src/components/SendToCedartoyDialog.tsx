import { CSSProperties, useState } from "react";
import {
  CedarToyGrammar,
  sendToCedarToy,
} from "../lib/cedartoyApi";

interface Props {
  open: boolean;
  songId: string;
  analysisId: string;
  songTitle: string;
  onClose: () => void;
}

const GRAMMARS: Array<{ key: CedarToyGrammar; label: string }> = [
  { key: "concert_visuals", label: "Concert visuals" },
  { key: "character_animation", label: "Character animation" },
  { key: "lighting", label: "Lighting" },
  { key: "camera_edit", label: "Camera edit" },
];

export default function SendToCedartoyDialog({
  open,
  songId,
  analysisId,
  songTitle,
  onClose,
}: Props) {
  const safeName = songTitle.replace(/[\\/:*?"<>|]/g, "_");
  const [outputFolder, setOutputFolder] = useState<string>(
    `exports/${safeName}`,
  );
  const [grammar, setGrammar] = useState<CedarToyGrammar>("concert_visuals");
  const [includeStems, setIncludeStems] = useState<boolean>(true);
  const [forceAnalyze, setForceAnalyze] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);
  const [okMsg, setOkMsg] = useState<string | null>(null);

  if (!open) return null;

  const handleSend = async () => {
    setBusy(true);
    setErr(null);
    setOkMsg(null);
    try {
      const res = await sendToCedarToy(songId, analysisId, {
        output_folder: outputFolder,
        grammar,
        include_stems: includeStems,
        force_analyze: forceAnalyze,
      });
      const stemsLine = res.stems_included
        ? "stems included"
        : res.stems_omitted_reason ?? "stems not included";
      setOkMsg(`Wrote ${res.output_folder} — ${stemsLine}.`);
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
          Send to CedarToy
        </div>

        <div style={gridStyle}>
          <label style={labelStyle}>Output folder</label>
          <input
            value={outputFolder}
            onChange={(e) => setOutputFolder(e.target.value)}
            style={inputStyle}
            placeholder="exports/<song>/"
          />

          <label style={labelStyle}>Grammar</label>
          <select
            value={grammar}
            onChange={(e) => setGrammar(e.target.value as CedarToyGrammar)}
            style={inputStyle}
          >
            {GRAMMARS.map((g) => (
              <option key={g.key} value={g.key}>{g.label}</option>
            ))}
          </select>

          <label style={labelStyle}>Stems</label>
          <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={includeStems}
              onChange={(e) => setIncludeStems(e.target.checked)}
            />
            Include stems (drums / bass / vocals / other)
          </label>

          <label style={labelStyle}>Analysis</label>
          <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={forceAnalyze}
              onChange={(e) => setForceAnalyze(e.target.checked)}
            />
            Force re-analyze (ignore cache, ~2 min)
          </label>
        </div>

        {err && <div style={{ marginTop: 14, color: "#f88", fontSize: 12 }}>{err}</div>}
        {okMsg && <div style={{ marginTop: 14, color: "#7ec97e", fontSize: 12 }}>{okMsg}</div>}

        <div style={actionsStyle}>
          <button onClick={onClose} disabled={busy} style={btnSecondary}>
            {okMsg ? "Close" : "Cancel"}
          </button>
          <button
            onClick={handleSend}
            disabled={busy || !outputFolder.trim()}
            style={btnPrimary}
          >
            {busy ? "Sending…" : "Export ▶"}
          </button>
        </div>
      </div>
    </div>
  );
}

const overlayStyle: CSSProperties = {
  position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
};
const panelStyle: CSSProperties = {
  background: "#161616", border: "1px solid #333", borderRadius: 6,
  padding: 20, minWidth: 480, color: "#ddd", fontSize: 13,
};
const gridStyle: CSSProperties = {
  display: "grid", gridTemplateColumns: "120px 1fr", gap: 10, alignItems: "center",
};
const labelStyle: CSSProperties = { color: "#bbb" };
const inputStyle: CSSProperties = {
  background: "#1a1a1a", color: "#eee", border: "1px solid #333",
  padding: "5px 8px", borderRadius: 4, fontSize: 13,
};
const actionsStyle: CSSProperties = {
  marginTop: 18, display: "flex", justifyContent: "flex-end", gap: 8,
};
const btnPrimary: CSSProperties = {
  background: "#3a5a8c", color: "#fff", border: "1px solid #5a7ab0",
  padding: "6px 16px", borderRadius: 4, cursor: "pointer", fontSize: 13,
};
const btnSecondary: CSSProperties = {
  background: "#1a1a1a", color: "#bbb", border: "1px solid #333",
  padding: "6px 16px", borderRadius: 4, cursor: "pointer", fontSize: 13,
};
