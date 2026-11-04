import { useId } from "react";
import { assetUrl } from "../api";
import {
  HEAD_HOVER_RATIO,
  TINTS,
  arrowGeometry,
  type Point,
  type ArrowTint,
} from "./arrowGeometry";

const deg = (r: number) => (r * 180) / Math.PI;

/** A greyscale sprite times a colour — Godot's `Modulate`, as an SVG filter. */
function tintMatrix(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return `${r} 0 0 0 0  0 ${g} 0 0 0  0 0 ${b} 0 0  0 0 0 1 0`;
}

/**
 * Draw one targeting arrow. Renders into an existing `<svg>` whose user units
 * are board px.
 *
 * `hovering` is the game's `SetHighlightingOn` pop — the head grows while the
 * cursor is over a valid target — and `tint` is that same call's colour.
 */
export default function TargetingArrow({
  from, tip, boardHeight, tint = "none", hovering = false, testId,
}: {
  from: Point;
  tip: Point;
  boardHeight: number;
  tint?: ArrowTint;
  hovering?: boolean;
  testId?: string;
}) {
  const uid = useId().replace(/:/g, "");
  const g = arrowGeometry(from, tip, boardHeight);
  const segmentSrc = assetUrl("ui/combat/targeting_arrow_segment.png");
  const headSrc = assetUrl("ui/combat/targeting_arrow_head.png");
  const filterId = `ta-${uid}-${tint}`;

  return (
    <g data-testid={testId} data-tint={tint} filter={tint === "none" ? undefined : `url(#${filterId})`}>
      {tint !== "none" && (
        <defs>
          <filter id={filterId} colorInterpolationFilters="sRGB">
            <feColorMatrix type="matrix" values={tintMatrix(TINTS[tint])} />
          </filter>
        </defs>
      )}
      {g.segments.map((s, i) => (
        <image
          key={i} href={segmentSrc} width={s.size} height={s.size}
          x={-s.size / 2} y={-s.size / 2}
          transform={`translate(${s.x} ${s.y}) rotate(${deg(s.rotation)})`}
        />
      ))}
      <g transform={`translate(${g.head.x} ${g.head.y}) rotate(${deg(g.head.rotation)})`}>
        <image
          data-testid={testId ? `${testId}-head` : undefined}
          href={headSrc} width={g.headSize.w} height={g.headSize.h}
          x={-g.headSize.w / 2} y={-g.headSize.h / 2}
          style={{
            transformBox: "fill-box", transformOrigin: "center",
            transform: `scale(${hovering ? HEAD_HOVER_RATIO : 1})`,
            transition: "transform 1s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        />
      </g>
    </g>
  );
}
