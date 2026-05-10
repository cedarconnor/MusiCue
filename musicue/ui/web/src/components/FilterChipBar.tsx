interface ChipDef {
  id: string;
  label: string;
}

const CHIPS: ChipDef[] = [
  { id: "has_stems", label: "has stems" },
  { id: "has_clap", label: "has CLAP" },
  { id: "has_url", label: "has URL" },
  { id: "bpm_80_120", label: "80–120 BPM" },
  { id: "bpm_120_140", label: "120–140 BPM" },
  { id: "bpm_140_plus", label: "140+ BPM" },
  { id: "recent_24h", label: "last 24h" },
  { id: "recent_7d", label: "last 7d" },
];

interface Props {
  active: string[];
  onToggle: (id: string) => void;
}

export default function FilterChipBar({ active, onToggle }: Props) {
  const set = new Set(active);
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {CHIPS.map((c) => {
        const on = set.has(c.id);
        return (
          <button
            key={c.id}
            onClick={() => onToggle(c.id)}
            style={{
              padding: "4px 10px",
              background: on ? "#3a5a8c" : "#222",
              color: on ? "#fff" : "#bbb",
              border: `1px solid ${on ? "#5a7ab0" : "#333"}`,
              borderRadius: 999,
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            {c.label}
          </button>
        );
      })}
    </div>
  );
}
