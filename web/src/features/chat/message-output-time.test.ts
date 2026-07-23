import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  formatMessageOutputTime,
  MessageOutputTime,
} from "./message-output-time.ts";

const minute = 60_000;
const hour = 60 * minute;
const day = 24 * hour;
const now = Date.UTC(2026, 6, 23, 9, 30, 0);
const TIME_ELEMENT_PATTERN = /<time/;
const DATE_TIME_ATTRIBUTE_PATTERN = /dateTime=/;
const FIVE_MINUTES_AGO_PATTERN = /5分钟前/;

test("formats assistant output time boundaries", () => {
  assert.equal(formatMessageOutputTime(now - 30_000, now), "刚刚");
  assert.equal(formatMessageOutputTime(now - 5 * minute, now), "5分钟前");
  assert.equal(formatMessageOutputTime(now - 2 * hour, now), "2小时前");
  assert.equal(formatMessageOutputTime(now - 3 * day, now), "3天前");
  assert.equal(
    formatMessageOutputTime(now - 8 * day, now),
    new Date(now - 8 * day).toLocaleDateString(),
  );
});

test("renders semantic time markup", () => {
  const completedAt = now - 5 * minute;
  const markup = renderToStaticMarkup(
    createElement(MessageOutputTime, { completedAt, now }),
  );

  assert.match(markup, TIME_ELEMENT_PATTERN);
  assert.match(markup, DATE_TIME_ATTRIBUTE_PATTERN);
  assert.match(markup, FIVE_MINUTES_AGO_PATTERN);
});
