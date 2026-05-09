export interface Song {
  id: string;
  title: string;
  has_analysis: boolean;
  analysis_ids: string[];
  source_url?: string | null;
}

export async function listSongs(): Promise<Song[]> {
  const r = await fetch("/api/songs");
  if (!r.ok) throw new Error(`listSongs: ${r.status}`);
  return (await r.json()).songs;
}

export async function uploadSong(file: File, title?: string): Promise<Song> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("title", title ?? file.name.replace(/\.[^/.]+$/, ""));
  const r = await fetch("/api/songs", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`uploadSong: ${r.status}`);
  return r.json();
}

export async function startAnalyze(songId: string): Promise<{ job_id: string }> {
  const r = await fetch(`/api/songs/${songId}/analyze`, { method: "POST" });
  if (!r.ok) throw new Error(`startAnalyze: ${r.status}`);
  return r.json();
}

export interface AnalysisJSON {
  tempo?: { bpm_global?: number; bpm?: number };
  source?: { duration_sec?: number; sample_rate?: number };
  lufs_integrated?: number | null;
  beats?: Array<{ t: number; downbeat: boolean }>;
  sections?: Array<{ start: number; end: number; label: string }>;
  onsets?: Record<
    string,
    Array<{ t: number; strength?: number; drum_class?: string }>
  >;
  curves?: Record<string, { hop_sec: number; values: number[] }>;
}

export async function getSong(songId: string): Promise<Song> {
  const r = await fetch(`/api/songs/${songId}`);
  if (!r.ok) throw new Error(`getSong: ${r.status}`);
  return r.json();
}

export async function getAnalysis(
  songId: string,
  analysisId: string,
): Promise<AnalysisJSON> {
  const r = await fetch(`/api/songs/${songId}/analyses/${analysisId}`);
  if (!r.ok) throw new Error(`getAnalysis: ${r.status}`);
  return r.json();
}

export interface PeaksJSON {
  version: number;
  channels: number;
  sample_rate: number;
  samples_per_pixel: number;
  length: number;
  data: number[];
}

export async function getPeaks(
  songId: string,
  analysisId: string,
  stem: string,
): Promise<PeaksJSON> {
  const r = await fetch(
    `/api/songs/${songId}/analyses/${analysisId}/peaks/${stem}`,
  );
  if (!r.ok) throw new Error(`getPeaks: ${r.status}`);
  return r.json();
}

export async function ensureClick(
  songId: string,
  analysisId: string,
): Promise<void> {
  const r = await fetch(
    `/api/songs/${songId}/analyses/${analysisId}/click`,
    { method: "POST" },
  );
  if (!r.ok) throw new Error(`ensureClick: ${r.status}`);
}

export function clickWavUrl(songId: string, analysisId: string): string {
  // Cache-bust: regenerated WAVs share the same URL but have different
  // content. Append a timestamp so the browser refetches each time the
  // user toggles the click track on.
  return `/api/songs/${songId}/analyses/${analysisId}/click.wav?t=${Date.now()}`;
}

export function sourceAudioUrl(songId: string): string {
  return `/api/songs/${songId}/source`;
}

export function stemAudioUrl(
  songId: string,
  analysisId: string,
  stem: string,
): string {
  return `/api/songs/${songId}/analyses/${analysisId}/stems/${stem}`;
}
