import { Link } from "react-router-dom";
import type { AnalysisJSON, Song } from "../lib/api";

function formatBpm(
  curve: Array<{ t: number; bpm: number }> | undefined,
  global: number | undefined,
): string {
  if (!global || !Number.isFinite(global)) return "— BPM";
  const points = curve ?? [];
  const distinct = new Set(points.map((p) => Math.round(p.bpm)));
  if (distinct.size <= 1) return `${Math.round(global)} BPM`;
  const lo = Math.min(...distinct);
  const hi = Math.max(...distinct);
  if (hi - lo <= 1) return `${Math.round(global)} BPM`;
  return `${lo}–${hi} BPM`;
}

function fmtDuration(sec: number | null | undefined): string {
  if (!sec || !isFinite(sec)) return "";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtLUFS(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "";
  return `${v.toFixed(1)} LUFS`;
}

function urlHostname(u: string): string {
  try {
    return new URL(u).hostname;
  } catch {
    return u;
  }
}

export default function MetadataCard({
  song,
  analysis,
}: {
  song: Song | null;
  analysis: AnalysisJSON;
}) {
  const bpmText = formatBpm(
    analysis.tempo?.bpm_curve,
    analysis.tempo?.bpm_global ?? analysis.tempo?.bpm,
  );
  const lufs = fmtLUFS(analysis.lufs_integrated);
  const dur = fmtDuration(analysis.source?.duration_sec);
  const url = song?.source_url ?? null;

  return (
    <div
      style={{
        padding: "8px 16px",
        borderBottom: "1px solid #333",
        display: "flex",
        alignItems: "center",
        gap: 16,
        flexWrap: "wrap",
      }}
    >
      <Link to="/library">◀ Library</Link>
      {dur && <span style={{ color: "#888" }}>⏱ {dur}</span>}
      {bpmText && <span style={{ color: "#888" }}>♪ {bpmText}</span>}
      {lufs && <span style={{ color: "#888" }}>{lufs}</span>}
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            marginLeft: "auto",
            color: "#9af",
            fontSize: 12,
            textDecoration: "none",
            border: "1px solid #335",
            padding: "2px 8px",
            borderRadius: 4,
          }}
        >
          ↗ {urlHostname(url)}
        </a>
      )}
    </div>
  );
}
