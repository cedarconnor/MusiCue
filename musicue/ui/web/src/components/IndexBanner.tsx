import { useEffect, useState } from "react";

interface IndexEvent {
  type: "idle" | "rebuilding" | "done";
  count?: number;
  total?: number;
}

export default function IndexBanner() {
  const [evt, setEvt] = useState<IndexEvent | null>(null);

  useEffect(() => {
    const es = new EventSource("/api/library/index_events");
    es.onmessage = (m) => {
      try {
        setEvt(JSON.parse(m.data));
      } catch {
        // malformed event; ignore
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, []);

  if (!evt || evt.type === "idle" || evt.type === "done") return null;

  return (
    <div
      style={{
        padding: "8px 12px",
        margin: "12px 0",
        background: "#2a2a1a",
        border: "1px solid #5a5a3a",
        borderRadius: 4,
        color: "#dca",
        fontSize: 13,
      }}
    >
      ⏳ Indexing library…
      {evt.count != null && evt.total != null
        ? ` ${evt.count} / ${evt.total} songs`
        : ""}
    </div>
  );
}
