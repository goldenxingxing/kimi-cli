import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  scrollCorrection,
  wasScrollApplied,
} from "./collapse-scroll-anchor.ts";

describe("scrollCorrection", () => {
  it("gives back the pixels the block drifted, so it can be put back", () => {
    // The virtual list compensated for the collapse by pulling the content
    // down: the block that was 200px into the viewport is now at 1400px, and
    // the reader is looking at messages from long before it.
    assert.equal(scrollCorrection(200, 1400), 1200);
  });

  it("works in both directions", () => {
    assert.equal(scrollCorrection(600, 100), -500);
  });

  it("ignores sub-pixel drift, which is layout noise rather than a jump", () => {
    assert.equal(scrollCorrection(200, 200), 0);
    assert.equal(scrollCorrection(200, 200.4), 0);
    assert.equal(scrollCorrection(200, 199.6), 0);
  });
});

describe("wasScrollApplied", () => {
  it("accepts the write the scroller honoured", () => {
    assert.equal(wasScrollApplied(4200, 4200), true);
  });

  it("tolerates the fractional pixel a scroller rounds to", () => {
    assert.equal(wasScrollApplied(4200, 4199.5), true);
  });

  it("rejects a write the scroller clamped at its end", () => {
    // Collapsing the last long block asks to scroll past the content bottom:
    // the browser stops short, the block cannot come back, and holding on
    // would repeat a forced layout every frame for nothing.
    assert.equal(wasScrollApplied(4200, 3100), false);
  });
});
