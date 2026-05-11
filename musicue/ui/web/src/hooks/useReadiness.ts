import { useCallback, useEffect, useState } from "react";
import {
  fetchReadiness,
  refreshReadiness,
  type ReadinessReport,
} from "../lib/readiness";

const POLL_MS = 5 * 60 * 1000;

export function useReadiness(): {
  report: ReadinessReport | null;
  error: string | null;
  loading: boolean;
  refresh: () => Promise<void>;
} {
  const [report, setReport] = useState<ReadinessReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setReport(await refreshReadiness());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      try {
        const r = await fetchReadiness();
        if (!cancelled) setReport(r);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();

    const id = window.setInterval(() => {
      void load();
    }, POLL_MS);

    function onFocus() {
      void load();
    }
    window.addEventListener("focus", onFocus);

    return () => {
      cancelled = true;
      window.clearInterval(id);
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  return { report, error, loading, refresh };
}
