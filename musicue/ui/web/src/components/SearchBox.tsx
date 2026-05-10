import { useEffect, useRef, useState } from "react";

interface Props {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  debounceMs?: number;
}

export default function SearchBox({
  value,
  onChange,
  placeholder = "Search title / url…",
  debounceMs = 150,
}: Props) {
  const [local, setLocal] = useState(value);
  const tRef = useRef<number | null>(null);

  useEffect(() => {
    setLocal(value);
  }, [value]);

  function handle(next: string) {
    setLocal(next);
    if (tRef.current) window.clearTimeout(tRef.current);
    tRef.current = window.setTimeout(() => onChange(next), debounceMs);
  }

  return (
    <input
      type="search"
      value={local}
      onChange={(e) => handle(e.target.value)}
      placeholder={placeholder}
      style={{
        flex: 1,
        padding: "8px 12px",
        background: "#1a1a1a",
        color: "#eee",
        border: "1px solid #333",
        borderRadius: 4,
        fontSize: 14,
      }}
    />
  );
}
