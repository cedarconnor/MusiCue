import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { clickWavUrl, ensureClick } from "../lib/api";

interface Props {
  ws: WaveSurfer | null;
  songId: string;
  analysisId: string;
  clickOn: boolean;
  onClickOnChange: (v: boolean) => void;
}

export default function Transport({
  ws,
  songId,
  analysisId,
  clickOn,
  onClickOnChange,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [clickLoading, setClickLoading] = useState(false);
  const clickAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (!ws) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setCurrentTime(ws.getCurrentTime());
    const onReady = () => setDuration(ws.getDuration());
    ws.on("play", onPlay);
    ws.on("pause", onPause);
    ws.on("audioprocess", onTime);
    ws.on("seeking", onTime);
    ws.on("ready", onReady);
    if (ws.getDuration()) setDuration(ws.getDuration());
    return () => {
      ws.un("play", onPlay);
      ws.un("pause", onPause);
      ws.un("audioprocess", onTime);
      ws.un("seeking", onTime);
      ws.un("ready", onReady);
    };
  }, [ws]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
        return;
      }
      if (e.code === "Space" && ws) {
        e.preventDefault();
        ws.playPause();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ws]);

  // Mix-mute decisions live in Timeline (which combines clickOn + solo into
  // one source of truth). Transport just owns clickOn and the click <audio>
  // playback; Timeline reads clickOn via a prop and mutes the mix
  // accordingly. This prevents a Timeline solo-state-change from overriding
  // Transport's "mute mix while click plays" intent.

  // Sync click <audio> element with WaveSurfer's transport.
  useEffect(() => {
    if (!ws) return;
    const click = clickAudioRef.current;
    if (!click) return;

    const onPlay = () => {
      click.currentTime = ws.getCurrentTime();
      if (clickOn) click.play().catch(() => {});
    };
    const onPause = () => click.pause();
    const onSeek = () => {
      click.currentTime = ws.getCurrentTime();
    };
    const onTime = () => {
      // Drift correction: snap if drift > 50ms.
      if (clickOn && Math.abs(click.currentTime - ws.getCurrentTime()) > 0.05) {
        click.currentTime = ws.getCurrentTime();
      }
    };
    ws.on("play", onPlay);
    ws.on("pause", onPause);
    ws.on("seeking", onSeek);
    ws.on("audioprocess", onTime);
    return () => {
      ws.un("play", onPlay);
      ws.un("pause", onPause);
      ws.un("seeking", onSeek);
      ws.un("audioprocess", onTime);
    };
  }, [ws, clickOn]);

  async function toggleClick() {
    if (!clickOn) {
      setClickLoading(true);
      try {
        await ensureClick(songId, analysisId);
        if (clickAudioRef.current) {
          clickAudioRef.current.src = clickWavUrl(songId, analysisId);
          clickAudioRef.current.load();
          if (ws) {
            clickAudioRef.current.currentTime = ws.getCurrentTime();
            if (ws.isPlaying()) {
              await clickAudioRef.current.play().catch(() => {});
            }
          }
        }
        onClickOnChange(true);
      } catch (err) {
        console.error("click toggle failed", err);
      } finally {
        setClickLoading(false);
      }
    } else {
      clickAudioRef.current?.pause();
      onClickOnChange(false);
    }
  }

  function fmtTime(t: number): string {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  return (
    <div
      style={{
        display: "flex",
        gap: 16,
        alignItems: "center",
        padding: "8px 16px",
        borderTop: "1px solid #333",
        background: "#1d1d1d",
      }}
    >
      <button onClick={() => ws?.playPause()} disabled={!ws}>
        {playing ? "Pause" : "Play"}
      </button>
      <span style={{ fontFamily: "monospace", color: "#bbb" }}>
        {fmtTime(currentTime)} / {fmtTime(duration)}
      </span>
      <button onClick={toggleClick} disabled={!ws || clickLoading}>
        {clickLoading ? "Generating click…" : `Click track: ${clickOn ? "ON" : "off"}`}
      </button>
      <audio ref={clickAudioRef} preload="auto" />
    </div>
  );
}
