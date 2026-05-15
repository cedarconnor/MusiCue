export type CedarToyGrammar =
  | "concert_visuals"
  | "character_animation"
  | "lighting"
  | "camera_edit";

export interface SendToCedarToyRequest {
  output_folder: string;
  grammar: CedarToyGrammar;
  include_stems: boolean;
  force_analyze?: boolean;
}

export interface SendToCedarToyResponse {
  ok: boolean;
  output_folder: string;
  stems_included: boolean;
  stems_omitted_reason: string | null;
}

export async function sendToCedarToy(
  songId: string,
  analysisId: string,
  req: SendToCedarToyRequest,
): Promise<SendToCedarToyResponse> {
  const r = await fetch(
    `/api/songs/${songId}/analyses/${analysisId}/send-to-cedartoy`,
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
    throw new Error(`send-to-cedartoy failed (${r.status}): ${detail}`);
  }
  return (await r.json()) as SendToCedarToyResponse;
}
