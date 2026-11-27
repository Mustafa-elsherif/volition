# Volition design system

The product reads as a precision electrophysiology instrument: a cool graphite
ground, an amber signal accent, an azure counter-pole, crisp hairlines, and no
shadows or glass. The binary nature of the task is carried by color, with the decoded
probability living on an amber-to-azure axis. The source of truth is `web/styles.css`;
this document records the decisions behind it.

## Color

Dark is the primary scene (an instrument read in a dim lab). Both themes ship and meet
WCAG AA.

| Role | Dark | Light |
|---|---|---|
| Background | `#0b0f14` | `#f4f6f8` |
| Panel | `#11161d` | `#ffffff` |
| Hairline | `#1e2731` | `#e0e6ec` |
| Ink | `#e7edf3` | `#0e1621` |
| Muted ink | `#8392a1` | `#5b6774` |
| Amber (brand / active, right pole) | `#ffb24c` | `#b4701a` |
| Azure (left pole) | `#74adff` | `#2f6fd6` |

Color is committed, not decorative: amber marks the interactive and the right choice,
azure the left choice, and the probability track is an azure-to-amber gradient so the
decoded value reads as a position on that axis. Secondary text is tinted from the ink,
never a flat gray.

## Typography

- **Archivo** (400 to 800) for display and UI. The display is heavy and tightly
  tracked; headings use `text-wrap: balance`.
- **JetBrains Mono** (400 to 600) for every numeric readout and for instrument labels
  (11px, uppercase, tracked). All changing numbers use tabular figures.

Monospace is used for data and measurement, not as decoration.

## Components

- **Panels**: 1px hairline borders, small radii (5 to 8px), flat. No drop shadows, no glass, no one-sided accent bars.
- **Probability axis**: a gradient track with a needle; the needle position is the decoded probability.
- **Raster**: a canvas field of thin amber ticks whose density encodes each neuron's z-scored firing rate, the signature material of the hero.
- **Region bars**: horizontal attention bars with tabular-figure percentages.
- **Buttons**: solid amber primary, hairline ghost secondary; both rebuilt in the instrument vocabulary.
- **Metric wall**: a hairline-divided grid, not floating cards.

## Browser surfaces

Selection, scrollbars, focus rings, caret, and tabular figures are all themed from the
palette. Focus is a 2px amber outline with offset.

## Motion

One authored moment: the hero raster reveals left to right with an exponential
ease-out on load, and region bars grow from zero on decode. Everything respects
`prefers-reduced-motion`. Transitions use `cubic-bezier(0.16, 1, 0.3, 1)`.

## Responsiveness

Fluid from 320px. The hero and workbench collapse from two columns to one below 900px;
the top nav collapses on small screens; gutters use `clamp(16px, 5vw, 64px)`.
