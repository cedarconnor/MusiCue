export type ExportFormat =
  | "csv"
  | "json"
  | "midi"
  | "after_effects"
  | "touchdesigner"
  | "osc"
  | "houdini"
  | "disguise"
  | "unreal";

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
  ticks_per_beat?: number;
  osc_host?: string;
  osc_port?: number;
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
  const m = /filename="?([^";]+)"?/.exec(disp);
  const fname = m?.[1] ?? `${req.filename ?? "cuesheet"}.bin`;

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fname;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
