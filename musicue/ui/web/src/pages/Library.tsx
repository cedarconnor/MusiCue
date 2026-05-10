import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Song,
  deleteSong,
  emptyTrash,
  listSongs,
  startAnalyze,
  trashSong,
  untrashSong,
  uploadSong,
} from "../lib/api";
import { useJob } from "../lib/jobs";
import URLDropZone from "../components/URLDropZone";
import SongRow from "../components/SongRow";
import SearchBox from "../components/SearchBox";
import FilterChipBar from "../components/FilterChipBar";
import IndexBanner from "../components/IndexBanner";

interface ActiveJob {
  jobId: string;
  songId?: string;
}
const ACTIVE_JOB_KEY = "musicue:activeJob";

function loadActiveJob(): ActiveJob | null {
  try {
    const raw = localStorage.getItem(ACTIVE_JOB_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as ActiveJob;
    return p.jobId ? p : null;
  } catch {
    return null;
  }
}
function persistActiveJob(j: ActiveJob | null): void {
  try {
    if (j) localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(j));
    else localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch {
    // Storage disabled / quota; in-memory state still works.
  }
}

type SortKey = "added_at" | "title" | "duration_sec" | "bpm_global";

export default function Library() {
  const nav = useNavigate();
  const [params, setParams] = useSearchParams();

  const trashed = params.get("trashed") === "1";
  const q = params.get("q") ?? "";
  const filters = params.getAll("filter");
  const sort = (params.get("sort") as SortKey | null) ?? "added_at";

  const [songs, setSongs] = useState<Song[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [trashCount, setTrashCount] = useState<number>(0);
  const [activeJob, setActiveJobState] = useState<ActiveJob | null>(loadActiveJob);
  const { events, done } = useJob(activeJob?.jobId ?? null);

  const setActiveJob = (j: ActiveJob | null) => {
    persistActiveJob(j);
    setActiveJobState(j);
  };

  const setParam = (k: string, v: string | null) => {
    const next = new URLSearchParams(params);
    if (v === null || v === "") next.delete(k);
    else next.set(k, v);
    setParams(next);
  };
  const toggleFilter = (id: string) => {
    const set = new Set(filters);
    if (set.has(id)) set.delete(id);
    else set.add(id);
    const next = new URLSearchParams(params);
    next.delete("filter");
    for (const f of set) next.append("filter", f);
    setParams(next);
  };

  const refresh = async () => {
    const list = await listSongs({
      q: q || undefined,
      filters: filters.length ? filters : undefined,
      sort,
      trashed,
    });
    setSongs(list);
    if (!trashed) {
      const tlist = await listSongs({ trashed: true });
      setTrashCount(tlist.length);
    } else {
      setTrashCount(list.length);
    }
  };

  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, filters.join("|"), sort, trashed]);

  // Validate any persisted activeJob (same logic as v0.1a).
  useEffect(() => {
    if (!activeJob) return;
    let cancelled = false;
    fetch(`/api/jobs/${activeJob.jobId}`).then((r) => {
      if (cancelled) return;
      if (!r.ok) setActiveJob(null);
      else
        r.json().then((snap) => {
          if (
            !cancelled &&
            ["complete", "failed", "cancelled"].includes(snap.status)
          )
            refresh().then(() => setActiveJob(null));
        });
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

  async function handleAction(
    s: Song,
    a: "open" | "trash" | "untrash" | "delete" | "copy_url",
  ) {
    try {
      if (a === "open") nav(`/editor/${s.id}/${(s.analysis_ids ?? [])[0]}`);
      else if (a === "trash") {
        await trashSong(s.id);
        refresh();
      } else if (a === "untrash") {
        await untrashSong(s.id);
        refresh();
      } else if (a === "delete") {
        await deleteSong(s.id);
        refresh();
      } else if (a === "copy_url" && s.source_url) {
        await navigator.clipboard.writeText(s.source_url);
      }
    } catch (err) {
      setError(String(err));
    }
  }

  const lastProgress = useMemo(
    () =>
      [...events].reverse().find((e) => e.type === "progress") as
        | { type: "progress"; fraction: number; stage?: string }
        | undefined,
    [events],
  );

  return (
    <div style={{ padding: 24, maxWidth: 960, margin: "0 auto" }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <h1 style={{ margin: 0 }}>{trashed ? "Trash" : "Library"}</h1>
        {trashed ? (
          <button
            onClick={() => setParam("trashed", null)}
            style={{
              background: "transparent",
              color: "#9cf",
              border: "none",
              cursor: "pointer",
            }}
          >
            ← Back to Library
          </button>
        ) : null}
      </div>

      <IndexBanner />

      {!trashed && (
        <>
          <URLDropZone onJobStarted={(jobId) => setActiveJob({ jobId })} />
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            style={{
              border: "2px dashed #555",
              padding: 24,
              margin: "12px 0",
              textAlign: "center",
              background: "#222",
            }}
          >
            Drop an audio file here to analyze
          </div>

          <div
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              margin: "16px 0 8px",
            }}
          >
            <SearchBox
              value={q}
              onChange={(v) => setParam("q", v || null)}
            />
            <select
              value={sort}
              onChange={(e) => setParam("sort", e.target.value)}
              style={{
                background: "#1a1a1a",
                color: "#eee",
                border: "1px solid #333",
                padding: "8px",
                borderRadius: 4,
              }}
            >
              <option value="added_at">Newest</option>
              <option value="title">Title (A–Z)</option>
              <option value="duration_sec">Duration</option>
              <option value="bpm_global">BPM</option>
            </select>
          </div>
          <FilterChipBar active={filters} onToggle={toggleFilter} />
        </>
      )}

      {trashed && songs.length > 0 && (
        <div style={{ margin: "12px 0" }}>
          <button
            onClick={async () => {
              if (!window.confirm(`Permanently delete ${songs.length} songs?`))
                return;
              await emptyTrash();
              refresh();
            }}
            style={{
              padding: "8px 14px",
              background: "#7a3a3a",
              color: "#fff",
              border: "1px solid #a55",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            Empty Trash ({songs.length})
          </button>
        </div>
      )}

      {error && (
        <div style={{ color: "#f88", margin: "8px 0" }}>Error: {error}</div>
      )}

      {activeJob && !trashed && <ProgressCard lastProgress={lastProgress} />}

      {songs.length === 0 ? (
        <div style={{ color: "#888", padding: 24, textAlign: "center" }}>
          {trashed
            ? "Trash is empty."
            : q || filters.length
              ? "No matches."
              : "No songs yet. Paste a URL above or drop an audio file."}
        </div>
      ) : (
        <div>
          {songs.map((s) => (
            <SongRow
              key={s.id}
              song={s}
              analysisId={(s.analysis_ids ?? [])[0]}
              trashed={trashed}
              onAction={(a) => handleAction(s, a)}
            />
          ))}
        </div>
      )}

      {!trashed && (
        <div
          style={{
            marginTop: 24,
            paddingTop: 16,
            borderTop: "1px solid #333",
          }}
        >
          <button
            onClick={() => setParam("trashed", "1")}
            style={{
              background: "transparent",
              color: "#888",
              border: "none",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            🗑 Trash ({trashCount})
          </button>
        </div>
      )}
    </div>
  );
}

function ProgressCard({
  lastProgress,
}: {
  lastProgress: { type: "progress"; fraction: number; stage?: string } | undefined;
}) {
  return (
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
  );
}
