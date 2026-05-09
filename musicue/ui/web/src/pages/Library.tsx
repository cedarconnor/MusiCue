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

const ACTIVE_JOB_KEY = "musicue:activeJob";

function loadActiveJob(): ActiveJob | null {
  try {
    const raw = localStorage.getItem(ACTIVE_JOB_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ActiveJob;
    return parsed.jobId ? parsed : null;
  } catch {
    return null;
  }
}

function persistActiveJob(job: ActiveJob | null): void {
  try {
    if (job) localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(job));
    else localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch {
    // Storage disabled / quota; in-memory state still works.
  }
}

export default function Library() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [activeJob, setActiveJobState] = useState<ActiveJob | null>(loadActiveJob);
  const [error, setError] = useState<string | null>(null);
  const { events, done } = useJob(activeJob?.jobId ?? null);
  const nav = useNavigate();

  const setActiveJob = (next: ActiveJob | null) => {
    persistActiveJob(next);
    setActiveJobState(next);
  };

  const refresh = async () => setSongs(await listSongs());
  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
  }, []);

  // Validate any persisted activeJob from a previous tab/session: if the
  // server doesn't recognize it (process restarted, job manager wiped),
  // drop the phantom so the UI doesn't get stuck showing a progress card
  // for a job that will never emit events.
  useEffect(() => {
    if (!activeJob) return;
    let cancelled = false;
    fetch(`/api/jobs/${activeJob.jobId}`).then((r) => {
      if (cancelled) return;
      if (!r.ok) setActiveJob(null);
      else if (r.ok) {
        // If the server already considers the job done, drop it -- the
        // SSE replay will just emit a status + terminal event and close,
        // which is fine, but clearing now also refreshes the song list.
        r.json().then((snap) => {
          if (
            !cancelled &&
            ["complete", "failed", "cancelled"].includes(snap.status)
          ) {
            refresh().then(() => setActiveJob(null));
          }
        });
      }
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        <div
          style={{
            margin: "12px 0",
            padding: "12px 16px",
            background: "#1d2a3a",
            border: "1px solid #2d4a6a",
            borderRadius: 6,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginBottom: 6,
            }}
          >
            <span style={{ color: "#cde", fontWeight: 600 }}>
              {lastProgress?.stage
                ? `Analyzing — ${lastProgress.stage}`
                : "Analyzing…"}
            </span>
            <span
              style={{
                color: "#9af",
                fontFamily: "monospace",
                fontSize: 13,
              }}
            >
              {lastProgress
                ? `${Math.round(lastProgress.fraction * 100)}%`
                : "queued"}
            </span>
          </div>
          <div
            style={{
              height: 6,
              background: "#0f1822",
              borderRadius: 3,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${(lastProgress?.fraction ?? 0) * 100}%`,
                background: "#5a8ec5",
                transition: "width 0.2s ease-out",
              }}
            />
          </div>
        </div>
      )}
      {songs.length === 0 && !activeJob && (
        <div style={{ color: "#888" }}>
          No songs yet. Paste a URL above or drop an audio file.
        </div>
      )}
      <ul style={{ listStyle: "none", padding: 0 }}>
        {songs.map((s) => {
          const isAnalyzing = activeJob?.songId === s.id;
          const pct = isAnalyzing && lastProgress
            ? `${Math.round(lastProgress.fraction * 100)}%`
            : null;
          return (
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
              <span style={{ color: isAnalyzing ? "#9af" : "#888", fontSize: 12 }}>
                {isAnalyzing
                  ? `Analyzing… ${pct ?? "queued"}`
                  : s.has_analysis
                  ? `${s.analysis_ids.length} analysis`
                  : "no analysis"}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
