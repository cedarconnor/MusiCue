import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Song, listSongs, startAnalyze, uploadSong } from "../lib/api";
import { useJob } from "../lib/jobs";
import URLDropZone from "../components/URLDropZone";

interface ActiveJob {
  jobId: string;
  // Set for file uploads; absent for URL-ingest until the complete event
  // delivers the song_id (URL-ingested songs are SHA-keyed off the
  // downloaded WAV, which only exists after stage 1 finishes).
  songId?: string;
}

export default function Library() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [activeJob, setActiveJob] = useState<ActiveJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { events, done } = useJob(activeJob?.jobId ?? null);
  const nav = useNavigate();

  const refresh = async () => setSongs(await listSongs());
  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!done || !activeJob) return;
    if (done.type === "complete") {
      const songId = done.result.song_id ?? activeJob.songId;
      const analysisId = done.result.analysis_id;
      refresh().then(() => {
        if (songId && analysisId) nav(`/editor/${songId}/${analysisId}`);
      });
    } else if (done.type === "error") {
      setError(done.error);
    }
    // Either way, drop the active job so the in-progress card unmounts.
    setActiveJob(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    setError(null);
    try {
      const song = await uploadSong(file);
      const { job_id } = await startAnalyze(song.id);
      setActiveJob({ jobId: job_id, songId: song.id });
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  const lastProgress = [...events].reverse().find((e) => e.type === "progress") as
    | { type: "progress"; fraction: number; stage?: string }
    | undefined;

  return (
    <div style={{ padding: 24, maxWidth: 720, margin: "0 auto" }}>
      <h1>Library</h1>
      <URLDropZone onJobStarted={(jobId) => setActiveJob({ jobId })} />
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        style={{
          border: "2px dashed #555",
          padding: 32,
          margin: "16px 0",
          textAlign: "center",
          background: "#222",
        }}
      >
        Drop an audio file here to analyze
      </div>
      {error && <div style={{ color: "#f88" }}>Error: {error}</div>}
      {activeJob && (
        <div style={{ margin: "8px 0" }}>
          Analyzing…{" "}
          {lastProgress
            ? `${Math.round(lastProgress.fraction * 100)}% (${lastProgress.stage ?? ""})`
            : "queued"}
        </div>
      )}
      {songs.length === 0 && !activeJob && (
        <div style={{ color: "#888" }}>
          No songs yet. Paste a URL above or drop an audio file.
        </div>
      )}
      <ul style={{ listStyle: "none", padding: 0 }}>
        {songs.map((s) => (
          <li
            key={s.id}
            style={{
              padding: 12,
              borderBottom: "1px solid #333",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span>
              {s.has_analysis ? (
                <Link to={`/editor/${s.id}/${s.analysis_ids[0]}`}>
                  {s.title}
                </Link>
              ) : (
                <span>{s.title}</span>
              )}
            </span>
            <span style={{ color: "#888", fontSize: 12 }}>
              {s.has_analysis
                ? `${s.analysis_ids.length} analysis`
                : "no analysis"}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
