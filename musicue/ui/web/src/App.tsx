import { Route, Routes, Navigate } from "react-router-dom";
import Library from "./pages/Library";
import Editor from "./pages/Editor";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/library" replace />} />
      <Route path="/library" element={<Library />} />
      <Route path="/editor/:songId/:analysisId" element={<Editor />} />
    </Routes>
  );
}
