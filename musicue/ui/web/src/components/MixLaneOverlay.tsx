import { forwardRef } from "react";
import { OVERLAY_HEIGHT } from "../lib/analysisLayers";

interface Props {
  topPx: number;
}

const MixLaneOverlay = forwardRef<HTMLCanvasElement, Props>(({ topPx }, ref) => (
  <canvas
    ref={ref}
    style={{
      position: "absolute",
      top: topPx,
      left: 0,
      height: OVERLAY_HEIGHT,
      pointerEvents: "none",
    }}
  />
));

export default MixLaneOverlay;
export { OVERLAY_HEIGHT };
