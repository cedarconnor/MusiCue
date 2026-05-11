import { Route, Routes, Navigate, Link } from "react-router-dom";
import Library from "./pages/Library";
import Editor from "./pages/Editor";
import ReadinessChip from "./components/ReadinessChip";

function Header() {
  return (
    <header
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "6px 14px",
        borderBottom: "1px solid #2a2a2a",
        background: "#111",
        color: "white",
      }}
    >
      <Link
        to="/library"
        style={{ color: "white", textDecoration: "none", fontWeight: 600 }}
      >
        MusiCue
      </Link>
      <ReadinessChip />
    </header>
  );
}

export default function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Navigate to="/library" replace />} />
        <Route path="/library" element={<Library />} />
        <Route path="/editor/:songId/:analysisId" element={<Editor />} />
      </Routes>
    </>
  );
}
