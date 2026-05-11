export type ComponentState = "ready" | "degraded" | "missing" | "error";
export type Overall = "green" | "amber" | "red";

export interface ComponentStatus {
  name: string;
  state: ComponentState;
  required: boolean;
  version: string | null;
  detail: string | null;
  cache_path: string | null;
  remediation: string | null;
}

export interface ReadinessReport {
  components: ComponentStatus[];
  overall: Overall;
  checked_at: string;
}

export async function fetchReadiness(): Promise<ReadinessReport> {
  const r = await fetch("/api/health/readiness");
  if (!r.ok) throw new Error(`readiness fetch failed: ${r.status}`);
  return (await r.json()) as ReadinessReport;
}

export async function refreshReadiness(): Promise<ReadinessReport> {
  const r = await fetch("/api/health/readiness/refresh", { method: "POST" });
  if (!r.ok) throw new Error(`readiness refresh failed: ${r.status}`);
  return (await r.json()) as ReadinessReport;
}

export function chipSummary(report: ReadinessReport): {
  label: string;
  color: Overall;
} {
  if (report.overall === "green") return { label: "All systems go", color: "green" };
  const bad = report.components.filter(
    (c) =>
      c.required && (c.state === "missing" || c.state === "error"),
  );
  if (bad.length > 0) {
    return {
      label: `${bad.length} critical issue${bad.length === 1 ? "" : "s"}`,
      color: "red",
    };
  }
  const warn = report.components.filter(
    (c) =>
      c.state === "missing" || c.state === "error" || c.state === "degraded",
  );
  return {
    label: `${warn.length} optional missing`,
    color: "amber",
  };
}
