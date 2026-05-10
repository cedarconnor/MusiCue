import { useState } from "react";
import { Link } from "react-router-dom";
import { Song, thumbnailUrl } from "../lib/api";

type Action = "open" | "trash" | "untrash" | "delete" | "copy_url";

interface Props {
  song: Song;
  /** Provided when the song has at least one analysis. */
  analysisId?: string;
  trashed?: boolean;
  onAction: (a: Action) => void;
}

function fallbackChip(title: string): string {
  return title.trim().slice(0, 1).toUpperCase() || "?";
}

function formatDuration(s: number | null | undefined): string {
  if (!s) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function hostnameSafe(url: string): string {
  try {
    return new URL(url).hostname.replace("www.", "");
  } catch {
    return url;
  }
}

export default function SongRow({ song, analysisId, trashed, onAction }: Props) {
  const [menuOpen, setMenuOpen] = useState(false);
  const linkable = !trashed && analysisId;

  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        padding: 12,
        borderBottom: "1px solid #2a2a2a",
        alignItems: "center",
      }}
    >
      <div
        style={{
          width: 96,
          height: 96,
          flexShrink: 0,
          background: "#1a1a1a",
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#888",
          fontSize: 36,
          fontFamily: "serif",
          overflow: "hidden",
        }}
      >
        {song.has_thumbnail ? (
          <img
            src={thumbnailUrl(song.id)}
            alt=""
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          fallbackChip(song.title)
        )}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 16, color: "#eee", marginBottom: 4 }}>
          {linkable ? (
            <Link
              to={`/editor/${song.id}/${analysisId}`}
              style={{ color: "#9cf", textDecoration: "none" }}
            >
              {song.title}
            </Link>
          ) : (
            <span style={{ color: trashed ? "#888" : "#eee" }}>{song.title}</span>
          )}
        </div>
        <div style={{ fontSize: 12, color: "#888", display: "flex", gap: 12 }}>
          <span>{formatDuration(song.duration_sec)}</span>
          {song.bpm_global != null && (
            <span>{Math.round(song.bpm_global)} BPM</span>
          )}
          {song.lufs_integrated != null && (
            <span>{song.lufs_integrated.toFixed(1)} LUFS</span>
          )}
          {song.source_url && (
            <span style={{ color: "#9af" }}>{hostnameSafe(song.source_url)}</span>
          )}
        </div>
      </div>
      <div style={{ position: "relative" }}>
        <button
          onClick={() => setMenuOpen((o) => !o)}
          style={{
            background: "transparent",
            color: "#888",
            border: "none",
            fontSize: 18,
            cursor: "pointer",
            padding: "4px 8px",
          }}
          aria-label="More"
        >
          ⋯
        </button>
        {menuOpen && (
          <div
            style={{
              position: "absolute",
              right: 0,
              top: "100%",
              background: "#1a1a1a",
              border: "1px solid #333",
              borderRadius: 4,
              minWidth: 140,
              zIndex: 10,
            }}
            onMouseLeave={() => setMenuOpen(false)}
          >
            {!trashed && linkable && (
              <MenuItem
                label="Open"
                onClick={() => {
                  onAction("open");
                  setMenuOpen(false);
                }}
              />
            )}
            {!trashed && (
              <MenuItem
                label="Move to Trash"
                onClick={() => {
                  onAction("trash");
                  setMenuOpen(false);
                }}
              />
            )}
            {trashed && (
              <MenuItem
                label="Restore"
                onClick={() => {
                  onAction("untrash");
                  setMenuOpen(false);
                }}
              />
            )}
            {trashed && (
              <MenuItem
                label="Delete forever"
                danger
                onClick={() => {
                  onAction("delete");
                  setMenuOpen(false);
                }}
              />
            )}
            {song.source_url && (
              <MenuItem
                label="Copy URL"
                onClick={() => {
                  onAction("copy_url");
                  setMenuOpen(false);
                }}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MenuItem({
  label,
  onClick,
  danger,
}: {
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "8px 12px",
        background: "transparent",
        color: danger ? "#f88" : "#eee",
        border: "none",
        cursor: "pointer",
        fontSize: 13,
      }}
    >
      {label}
    </button>
  );
}
