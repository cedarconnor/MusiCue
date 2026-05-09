import { useState } from "react";
import { analyzeUrl } from "../lib/api";

export default function URLDropZone({
  onJobStarted,
}: {
  onJobStarted: (jobId: string) => void;
}) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    if (!url.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const { job_id } = await analyzeUrl(url.trim());
      setUrl("");
      onJobStarted(job_id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        border: "1px dashed #444",
        padding: 16,
        borderRadius: 6,
        marginBottom: 12,
      }}
    >
      <div style={{ marginBottom: 6, color: "#888", fontSize: 12 }}>
        Paste a URL (YouTube, SoundCloud, Bandcamp, …)
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="https://www.youtube.com/watch?v=…"
          disabled={busy}
          style={{
            flex: 1,
            padding: "6px 8px",
            background: "#181818",
            border: "1px solid #333",
            color: "#ddd",
            borderRadius: 4,
          }}
        />
        <button onClick={submit} disabled={busy || !url.trim()}>
          {busy ? "Submitting…" : "Analyze URL"}
        </button>
      </div>
      {err && (
        <div style={{ color: "#f88", marginTop: 8, fontSize: 12 }}>{err}</div>
      )}
    </div>
  );
}
