import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Orbs from "../Orbs";
import { orbArc } from "../orbArc";
import Board from "../Board";
import type { Orb, Step } from "../../api";

function orb(index: number, type: string, passive: number, evoke: number): Orb {
  return { index, type, name: type, passive, evoke, image: `/assets/orb_atlas/${type.toLowerCase()}.png` };
}

describe("Orbs", () => {
  it("renders one icon per channelled orb with its passive value", () => {
    render(<Orbs orbs={[orb(0, "Lightning", 3, 8), orb(1, "Frost", 2, 5)]} slots={3} />);
    expect(screen.getByAltText("Lightning")).toBeInTheDocument();
    expect(screen.getByAltText("Frost")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("pads to the slot capacity with empty orb slots", () => {
    render(<Orbs orbs={[orb(0, "Dark", 6, 6)]} slots={3} />);
    expect(screen.getAllByLabelText("empty orb slot")).toHaveLength(2);
  });

  it("renders nothing when there are no slots and no orbs", () => {
    const { container } = render(<Orbs orbs={[]} slots={0} />);
    expect(container.firstChild).toBeNull();
  });

  it("draws a ring when the empty-slot art is missing, never its caption", () => {
    // The art tree deploys separately from the app, so the file can be absent
    // on a bucket that is a deploy behind. A broken <img> paints its ALT TEXT,
    // which is how "empty orb slot" ended up written across the belt.
    render(<Orbs orbs={[]} slots={1} />);
    const art = screen.getByLabelText("empty orb slot");
    expect(art.tagName).toBe("IMG");
    expect(art).toHaveAttribute("alt", "");
    fireEvent.error(art);
    const ring = screen.getByLabelText("empty orb slot");
    expect(ring.tagName).not.toBe("IMG");
    expect(ring).toHaveClass("gs-orb-empty");
  });

  it("hangs the belt on the arc the game sweeps around the creature", () => {
    // NOrbManager.TweenLayout: slot 0 (next to evoke) out on the creature's
    // RIGHT, the last slot on its LEFT, the middle of the queue over the crown —
    // and a bigger queue rides a WIDER ring, not a more crowded one. The arc is
    // centred 0.6 of a body ABOVE the creature's root, which is where the game
    // hangs NOrbManager (the `%OrbPos` marker) — see ORB_CENTRE_TO_BODY.
    const radius = (p: { x: number; y: number }, bodyH: number) =>
      Math.hypot(p.x, p.y + 0.6 * bodyH);
    const arc = orbArc(3, 245);
    expect(arc[0].x).toBeGreaterThan(0);
    expect(arc[2].x).toBeLessThan(0);
    expect(arc[1].y).toBeLessThan(Math.min(arc[0].y, arc[2].y)); // -y is up
    expect(radius(arc[1], 245)).toBeCloseTo(225, 0);       // _minRadius at capacity 3
    expect(radius(orbArc(10, 245)[0], 245)).toBeCloseTo(300, 0); // _maxRadius at 10
    // …and the whole belt scales with the creature it belongs to.
    expect(radius(orbArc(3, 490)[1], 490)).toBeCloseTo(450, 0);
  });

  it("keeps the whole queue up around the creature's head, never at its hips", () => {
    // The bug this pins: with the arc centred on the creature's ROOT, the ends
    // of the queue (slot 0 and the last slot, the two lowest points of a 125°
    // sweep that stops 25° above the horizontal) hung at 0.4 of a body — beside
    // the character's hips — and only the middle of the queue reached its head.
    for (const n of [1, 3, 5, 10]) {
      for (const p of orbArc(n, 245)) {
        expect(p.y).toBeLessThan(-0.9 * 245); // every slot at crown height or above
      }
    }
  });

  it("draws an orb at the size the game's orb scene gives it", () => {
    // `scenes/orbs/orb.tscn`: %Outline / Bounds are a 60x60 rect and the 100px
    // art is scaled DOWN into it; NOrb._Ready then takes the local player's
    // orbs to 0.85. Reading the 100px art as the sprite size drew them 1.67x
    // too big — big enough to cover the character they orbit.
    render(<Orbs orbs={[orb(0, "Lightning", 3, 8)]} slots={1} bodyH={245} />);
    expect(screen.getByAltText("Lightning")).toHaveStyle({ width: "51px", height: "51px" });
  });

  it("never under-counts capacity below the number of filled orbs", () => {
    // slots omitted but orbs present → capacity falls back to orb count.
    render(<Orbs orbs={[orb(0, "Plasma", 1, 1)]} />);
    expect(screen.getByAltText("Plasma")).toBeInTheDocument();
    expect(screen.queryByLabelText("empty orb slot")).toBeNull();
  });
});

function defectStep(): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: 0 },
    state: {
      decision: "combat_play", round: 1, energy: 3, max_energy: 3,
      draw_pile_count: 0, discard_pile_count: 0, exhaust_pile_count: 0,
      draw_pile: [], discard_pile: [], exhaust_pile: [],
      player: { name: "Defect", hp: 70, max_hp: 75, block: 0 },
      enemies: [{ index: 0, name: "Cultist", hp: 50, max_hp: 50, block: 0, image: null }],
      hand: [],
      orbs: [orb(0, "Lightning", 3, 8)],
      orb_slots: 3,
    },
  };
}

describe("Board orb queue", () => {
  it("surfaces the Defect orb queue on the combat board", () => {
    render(<Board step={defectStep()} />);
    expect(screen.getByAltText("Lightning")).toBeInTheDocument();
    // 1 filled + 2 empty slots = capacity 3.
    expect(screen.getAllByLabelText("empty orb slot")).toHaveLength(2);
  });

  it("anchors the belt to the player's sprite, not to the column above it", () => {
    // The belt is placed around the creature, so it has to live inside the
    // sprite wrap. Stacked above the player column it anchored to the top of
    // the RESERVED sprite box — sized for the tallest rig the arena makes room
    // for — which parked the orbs in the empty sky above a short character.
    const { container } = render(<Board step={defectStep()} />);
    expect(container.querySelector('[data-combatant="player"] .gs-sprite-wrap .gs-orbs')).not.toBeNull();
  });
});
