import { useEffect, useState } from "react";

export type JobEvent =
  | { type: "status"; status: string; progress?: number }
  | { type: "progress"; fraction: number; stage?: string }
  | { type: "complete"; result: { analysis_id: string } }
  | { type: "error"; error: string }
  | { type: "cancelled" };

export function useJob(jobId: string | null): {
  events: JobEvent[];
  done: JobEvent | null;
} {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [done, setDone] = useState<JobEvent | null>(null);

  useEffect(() => {
    if (!jobId) return;
    setEvents([]);
    setDone(null);
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${proto}//${location.host}/api/jobs/${jobId}/stream`,
    );
    ws.onmessage = (m) => {
      const evt = JSON.parse(m.data) as JobEvent;
      setEvents((prev) => [...prev, evt]);
      if (evt.type === "complete" || evt.type === "error" || evt.type === "cancelled") {
        setDone(evt);
        ws.close();
      }
    };
    return () => ws.close();
  }, [jobId]);

  return { events, done };
}
