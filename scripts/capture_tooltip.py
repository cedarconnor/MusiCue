"""Capture the section-transition tooltip hover state."""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/screenshots")
SONG = "a0639697796de0fe79f8f78707da867bfcb45ee7790dcff2ef49d1f0ce88aa94"
ANALYSIS = "bac98e988355"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(f"http://localhost:8765/editor/{SONG}/{ANALYSIS}")
        # Wait for decode
        for _ in range(60):
            dur = page.evaluate(
                """() => {
                  const h = Array.from(document.querySelectorAll('div')).filter(d=>d.shadowRoot);
                  const a = h[0]?.shadowRoot.querySelector('audio');
                  return a && Number.isFinite(a.duration) ? a.duration : 0;
                }"""
            )
            if dur > 0:
                break
            time.sleep(0.5)
        time.sleep(2.0)

        # Find a transition rect by walking the DOM: it lives in the div that
        # wraps mixHostRef + MixLaneOverlay + TransitionTooltipLayer.
        rect = page.evaluate(
            """() => {
              // Tooltip rects sit inside a parent that has pointerEvents:'none'
              // and contains children with cursor:'pointer'. Find them.
              const candidates = Array.from(document.querySelectorAll('div'))
                .filter(d => {
                  const cs = d.style;
                  return cs.position === 'absolute'
                      && cs.cursor === 'pointer'
                      && cs.pointerEvents === 'auto'
                      && d.parentElement?.style?.pointerEvents === 'none';
                });
              if (!candidates.length) return null;
              // Pick one near the middle of the timeline (transitions are often there).
              const target = candidates[Math.floor(candidates.length / 2)];
              const r = target.getBoundingClientRect();
              return {x: r.left + r.width / 2, y: r.top + r.height / 2, count: candidates.length};
            }"""
        )
        print("rect:", rect)
        if not rect:
            print("FAIL: no transition rects found")
            browser.close()
            return

        page.mouse.move(rect["x"], rect["y"])
        time.sleep(0.8)
        page.screenshot(path=str(OUT / "editor_transition_tooltip.png"))
        print("wrote tooltip screenshot")
        browser.close()


if __name__ == "__main__":
    main()
