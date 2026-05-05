import { useParams } from "react-router-dom";

export default function Editor() {
  const { songId, analysisId } = useParams();
  return (
    <div style={{ padding: 24 }}>
      Editor for {songId} / {analysisId} (Task 14 will wire this up)
    </div>
  );
}
