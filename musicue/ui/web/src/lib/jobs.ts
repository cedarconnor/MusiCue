import { useEffect, useState } from "react";

export type JobEvent =
  | { type: "status"; status: string; progress?: number }
  | { type: "progress"; fraction: number; stage?: string }
  | { type: "complete"; result: { analysis_id: string; song_id?: string } }
  | { type: "error"; error: string }
  | { type: "cancelled" };

/** Subscribe to a job's event stream via SSE. Returns events seen so far
 * and the terminal event once it lands. */
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
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.onmessage = (m) => {
      let evt: JobEvent;
      try {
        evt = JSON.parse(m.data) as JobEvent;
      } catch {
        return;
      }
      setEvents((prev) => [...prev, evt]);
      if (
        evt.type === "complete" ||
        evt.type === "error" ||
        evt.type === "cancelled"
      ) {
        setDone(evt);
        es.close();
      }
    };
    es.onerror = () => {
      // EventSource auto-reconnects on transient errors. Only the server
      // emitting an explicit error event is treated as terminal.
    };
    return () => es.close();
  }, [jobId]);

  return { events, done };
}

/** Request cancellation. The server emits a `cancelled` (or `error`/
 * `complete`) event on the same job's SSE stream as a side effect. */
export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}
