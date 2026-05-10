import { AnalysisJSON, OnsetItem, PhraseItem } from "../lib/api";
import { Stem } from "./OnsetMarkers";
import DrumClassChip from "./DrumClassChip";

export type SelectedAnnotation =
  | { kind: "onset"; stem: Stem; idx: number }
  | { kind: "phrase"; stem: Stem; idx: number }
  | null;

interface Props {
  selected: SelectedAnnotation;
  analysis: AnalysisJSON;
}

const stripStyle: React.CSSProperties = {
  display: "flex",
  gap: 6,
  alignItems: "center",
  flexWrap: "wrap",
  padding: "8px 16px",
  borderTop: "1px solid #222",
  borderBottom: "1px solid #222",
  background: "#0f0f0f",
  minHeight: 32,
};
const chipStyle: React.CSSProperties = {
  padding: "2px 8px",
  borderRadius: 999,
  fontSize: 11,
  fontWeight: 500,
};

export default function LabelChipStrip({ selected, analysis }: Props) {
  if (!selected) {
    return (
      <div style={stripStyle}>
        <span style={{ color: "#666", fontSize: 12 }}>
          Click an onset or phrase to see its CLAP labels.
        </span>
      </div>
    );
  }
  const stemTag = (
    <span style={{ ...chipStyle, background: "#333", color: "#bbb" }}>
      {selected.stem}
    </span>
  );

  let item: OnsetItem | PhraseItem | undefined;
  let extra: React.ReactNode = null;
  if (selected.kind === "onset") {
    item = (analysis.onsets?.[selected.stem] ?? [])[selected.idx];
    if (item && selected.stem === "drums" && (item as OnsetItem).drum_class) {
      const o = item as OnsetItem;
      extra = <DrumClassChip drumClass={o.drum_class} conf={o.drum_class_conf} />;
    }
  } else {
    item = (analysis.phrases?.[selected.stem] ?? [])[selected.idx];
    if (item) {
      const p = item as PhraseItem;
      extra = (
        <span style={{ ...chipStyle, background: "#222", color: "#999" }}>
          {p.note_count} notes · MIDI {p.pitch_low}–{p.pitch_peak}
        </span>
      );
    }
  }
  if (!item) return <div style={stripStyle}>{stemTag}</div>;

  const labels = item.labels ?? [];
  return (
    <div style={stripStyle}>
      {stemTag}
      {extra}
      {labels.length === 0 ? (
        <span style={{ color: "#666", fontSize: 12 }}>
          No labels for this {selected.kind}.
        </span>
      ) : (
        labels.map((l, i) => (
          <span
            key={i}
            style={{ ...chipStyle, background: "#1a3a5a", color: "#cde" }}
          >
            {l}
          </span>
        ))
      )}
    </div>
  );
}
