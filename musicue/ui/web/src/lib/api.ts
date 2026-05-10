export interface Song {
  id: string;
  title: string;
  source_url?: string | null;
  source_ext: string;
  duration_sec?: number | null;
  bpm_global?: number | null;
  lufs_integrated?: number | null;
  added_at: string;
  trashed_at?: string | null;
  has_thumbnail: boolean;
  // legacy compatibility for code paths that still expect these
  has_analysis?: boolean;
  analysis_ids?: string[];
}

export interface ListSongsParams {
  q?: string;
  filters?: string[];
  sort?: "added_at" | "title" | "duration_sec" | "bpm_global";
  trashed?: boolean;
  limit?: number;
  offset?: number;
}

export async function listSongs(
  params: ListSongsParams = {},
): Promise<Song[]> {
  const sp = new URLSearchParams();
  if (params.q) sp.set("q", params.q);
  for (const f of params.filters ?? []) sp.append("filter", f);
  if (params.sort) sp.set("sort", params.sort);
  if (params.trashed) sp.set("trashed", "1");
  if (params.limit !== undefined) sp.set("limit", String(params.limit));
  if (params.offset !== undefined) sp.set("offset", String(params.offset));
  const qs = sp.toString();
  const r = await fetch(`/api/songs${qs ? `?${qs}` : ""}`);
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

export async function analyzeUrl(url: string): Promise<{ job_id: string }> {
  const r = await fetch("/api/songs/from_url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!r.ok) {
    const detail = (await r.json().catch(() => ({}))).detail ?? r.statusText;
    throw new Error(`analyzeUrl: ${detail}`);
  }
  return r.json();
}

export interface OnsetItem {
  t: number;
  strength?: number;
  drum_class?: string | null;
  drum_class_conf?: number | null;
  labels?: string[];
}

export interface PhraseItem {
  t_start: number;
  t_end: number;
  note_count?: number;
  pitch_low?: number;
  pitch_peak?: number;
  pitch_contour?: number[];
  labels?: string[];
}

export interface AnalysisJSON {
  schema_version?: string;
  tempo?: { bpm_global?: number; bpm?: number };
  source?: { duration_sec?: number; sample_rate?: number };
  lufs_integrated?: number | null;
  beats?: Array<{ t: number; downbeat: boolean }>;
  sections?: Array<{ start: number; end: number; label: string }>;
  onsets?: Record<string, OnsetItem[]>;
  phrases?: Record<string, PhraseItem[]>;
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

export function thumbnailUrl(songId: string): string {
  return `/api/songs/${songId}/thumbnail`;
}

export async function trashSong(songId: string): Promise<void> {
  const r = await fetch(`/api/songs/${songId}/trash`, { method: "POST" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `trashSong: ${r.status}`);
  }
}

export async function untrashSong(songId: string): Promise<void> {
  const r = await fetch(`/api/songs/${songId}/untrash`, { method: "POST" });
  if (!r.ok) throw new Error(`untrashSong: ${r.status}`);
}

export async function deleteSong(songId: string): Promise<void> {
  const r = await fetch(`/api/songs/${songId}`, { method: "DELETE" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `deleteSong: ${r.status}`);
  }
}

export async function emptyTrash(): Promise<{
  deleted: number;
  skipped: string[];
}> {
  const r = await fetch("/api/library/empty-trash", { method: "POST" });
  if (!r.ok) throw new Error(`emptyTrash: ${r.status}`);
  return r.json();
}

export interface LoopState {
  loop_in: number;
  loop_out: number;
  enabled: boolean;
  updated_at?: string;
}

export async function getLoop(
  songId: string,
  analysisId: string,
): Promise<LoopState | null> {
  const r = await fetch(`/api/songs/${songId}/analyses/${analysisId}/loop`);
  if (r.status === 204) return null;
  if (!r.ok) throw new Error(`getLoop: ${r.status}`);
  return r.json();
}

export async function putLoop(
  songId: string,
  analysisId: string,
  loop: Omit<LoopState, "updated_at">,
): Promise<LoopState> {
  const r = await fetch(`/api/songs/${songId}/analyses/${analysisId}/loop`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(loop),
  });
  if (!r.ok) throw new Error(`putLoop: ${r.status}`);
  return r.json();
}
