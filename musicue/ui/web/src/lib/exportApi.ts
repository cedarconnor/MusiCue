export type ExportFormat =
  | "csv"
  | "json"
  | "midi"
  | "after_effects"
  | "touchdesigner"
  | "osc"
  | "houdini"
  | "disguise"
  | "unreal"
  | "edl"
  | "fcpxml"
  | "premiere_markers"
  | "resolve_markers";

export type ExportGrammar =
  | "concert_visuals"
  | "character_animation"
  | "lighting"
  | "camera_edit";

export interface ExportRequest {
  format: ExportFormat;
  grammar: ExportGrammar;
  filename?: string;
  fps?: number;
  drop_frame?: boolean;
  ticks_per_beat?: number;
  osc_host?: string;
  osc_port?: number;
  marker_sources?: string[];
}

/**
 * Pull a filename out of a Content-Disposition header. Handles both forms:
 *   - `filename="cuesheet.jsx"` (legacy)
 *   - `filename*=utf-8''cuesheet.jsx` (RFC 5987, URL-encoded)
 *
 * Starlette emits only the extended form by default — when our previous
 * implementation matched only `filename=`, every download landed as `.bin`.
 */
export function parseFilenameFromContentDisposition(disp: string): string | null {
  // Prefer the extended form (RFC 5987) — it's UTF-8 safe.
  const ext = /filename\*=(?:utf-8|UTF-8)''([^;]+)/.exec(disp);
  if (ext) {
    try {
      return decodeURIComponent(ext[1].trim());
    } catch {
      return ext[1].trim();
    }
  }
  const legacy = /filename="?([^";]+)"?/.exec(disp);
  return legacy ? legacy[1] : null;
}

/** POSTs the export request and triggers a browser download. */
export async function exportCuesheet(
  songId: string,
  analysisId: string,
  req: ExportRequest,
): Promise<void> {
  const r = await fetch(
    `/api/songs/${songId}/analyses/${analysisId}/export`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    },
  );
  if (!r.ok) {
    let detail = "";
    try {
      const j = await r.json();
      detail = j.detail ?? JSON.stringify(j);
    } catch {
      detail = await r.text().catch(() => "");
    }
    throw new Error(`export failed (${r.status}): ${detail}`);
  }
  const blob = await r.blob();
  const disp = r.headers.get("Content-Disposition") ?? "";
  const fname = parseFilenameFromContentDisposition(disp)
    ?? `${req.filename ?? "cuesheet"}`;

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fname;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
