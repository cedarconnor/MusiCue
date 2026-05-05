import { Route, Routes, Link } from "react-router-dom";

function Home() {
  return (
    <div style={{ padding: 24 }}>
      <h1>MusiCue</h1>
      <Link to="/library">Library</Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
    </Routes>
  );
}
