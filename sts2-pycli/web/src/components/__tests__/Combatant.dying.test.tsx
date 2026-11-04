import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Combatant from "../Combatant";
import { dyingChromeFadeSeconds, DEFAULT_DYING_FADE_SECONDS } from "../combatantMetrics";
import type { DissolveEffect } from "../../effects/manifest";

// A rigged combatant mounts `SpineSprite`, whose WebGL load has nothing to
// run against in jsdom — same stand-in `Board.effects.test.tsx` uses.
vi.mock("../SpineSprite", () => ({
  default: () => <div data-testid="spine-mock" />,
}));

/**
 * Task 14 fix round 1: the WebGL dissolve shader now owns the SPRITE's own
 * fade (`SpineSprite`'s `dissolve` prop), so `.gs-dying`'s CSS animation was
 * re-scoped to the surrounding DOM chrome only (HP bar / name / powers) —
 * see `game-skin.css`'s own doc on the rule. That rule reads its duration
 * off `--gs-dying-ms`, which `dyingChromeFadeSeconds` computes and
 * `Combatant` sets inline; these tests pin that computation and its wiring,
 * not the CSS animation itself (jsdom does not apply stylesheets — see
 * `gameSkinDying.test.ts` for the CSS-level check).
 */
const enemy = { name: "Test Subject", hp: 40, max_hp: 100, block: 0 };

const dissolveEffect = (dissolveSeconds: number) => ({ dissolveSeconds }) as DissolveEffect;

describe("dyingChromeFadeSeconds", () => {
  it("matches the dissolve's own duration when one is playing", () => {
    expect(dyingChromeFadeSeconds({ effect: dissolveEffect(1.4) })).toBe(1.4);
  });

  it("falls back to the pre-Task-14 CSS timing with no dissolve yet (manifest not loaded, or not dying at all)", () => {
    expect(dyingChromeFadeSeconds(null)).toBe(DEFAULT_DYING_FADE_SECONDS);
    expect(dyingChromeFadeSeconds(undefined)).toBe(DEFAULT_DYING_FADE_SECONDS);
  });
});

describe("a dying enemy's --gs-dying-ms wiring", () => {
  it("is fed the dissolve's own duration so the chrome and the sprite finish together", () => {
    const { container } = render(
      <Combatant c={enemy} isEnemy dying
        dissolve={{ effect: dissolveEffect(1.4), rev: "r1", startedAt: 0 }} />,
    );
    const wrap = container.querySelector(".gs-combatant") as HTMLElement;
    expect(wrap.style.getPropertyValue("--gs-dying-ms")).toBe("1.4s");
  });

  it("falls back to the default timing while dying but before a dissolve has resolved", () => {
    const { container } = render(<Combatant c={enemy} isEnemy dying />);
    const wrap = container.querySelector(".gs-combatant") as HTMLElement;
    expect(wrap.style.getPropertyValue("--gs-dying-ms")).toBe(`${DEFAULT_DYING_FADE_SECONDS}s`);
  });
});

/**
 * Task 14 review round 3: `.gs-dying` alone does not mean the sprite is being
 * dissolved, and `game-skin.css` skips the sprite in its fade only for a
 * sprite the shader really owns. `.gs-dissolving` is the marker that says so;
 * get it wrong in the permissive direction and a corpse double-fades, in the
 * strict direction and it pops out at full opacity with no fade at all.
 */
describe("the .gs-dissolving marker", () => {
  const dying = { effect: dissolveEffect(2.5), rev: "r1", startedAt: 0 };
  const cls = (ui: React.ReactElement) =>
    (render(ui).container.querySelector(".gs-combatant") as HTMLElement).className;

  it("is set when a dying rig has a dissolve to run", () => {
    expect(cls(<Combatant c={{ ...enemy, rig: "monsters/twig_slime" }} isEnemy dying dissolve={dying} />))
      .toMatch(/\bgs-dissolving\b/);
  });

  it("is NOT set before the effect manifest resolves — `dissolve` is null and SpineSprite draws the plain rig", () => {
    expect(cls(<Combatant c={{ ...enemy, rig: "monsters/twig_slime" }} isEnemy dying dissolve={null} />))
      .not.toMatch(/\bgs-dissolving\b/);
  });

  it("is NOT set for an enemy with no rig — that is a plain <img>, which no shader touches", () => {
    expect(cls(<Combatant c={{ ...enemy, image: "/img/slime.png" }} isEnemy dying dissolve={dying} />))
      .not.toMatch(/\bgs-dissolving\b/);
  });

  it("is NOT set on a noSprite spacer — its body is drawn by another combatant", () => {
    expect(cls(
      <Combatant c={{ ...enemy, rig: "monsters/kaiser_crab" }} isEnemy dying dissolve={dying}
        noSprite={{ w: 40, h: 40 }} />,
    )).not.toMatch(/\bgs-dissolving\b/);
  });

  it("is not set at all when the creature is not dying", () => {
    expect(cls(<Combatant c={{ ...enemy, rig: "monsters/twig_slime" }} isEnemy dissolve={dying} />))
      .not.toMatch(/\bgs-dissolving\b/);
  });
});
