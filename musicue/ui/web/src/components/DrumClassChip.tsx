const COLORS: Record<string, string> = {
  kick: "#ef4444",
  snare: "#3b82f6",
  hihat: "#22c55e",
  hat: "#22c55e",
  tom: "#a855f7",
  cymbal: "#fbbf24",
};

interface Props {
  drumClass?: string | null;
  conf?: number | null;
}

export default function DrumClassChip({ drumClass, conf }: Props) {
  if (!drumClass) return null;
  const c = COLORS[drumClass] ?? "#888";
  const pct = conf != null ? `${Math.round(conf * 100)}%` : null;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "2px 8px",
        background: c,
        color: "#fff",
        borderRadius: 999,
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {drumClass}
      {pct && <span style={{ opacity: 0.8 }}>{pct}</span>}
    </span>
  );
}
