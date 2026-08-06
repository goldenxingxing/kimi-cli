import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  LEGACY_UPLOADS_REGEX,
  MEDIA_TAG_PATH_REGEX,
  isUploadedFilePath,
  uploadedFileName,
} from "./upload-paths.ts";

const SESSION_ID = "0f9c1e7a-4b2d-4c1e-9f3a-8d5b6c7e0a12";

describe("uploadedFileName", () => {
  it("takes the filename off a POSIX path", () => {
    assert.equal(
      uploadedFileName(`/Users/me/.kimi/${SESSION_ID}/uploads/report_ab12cd.pdf`),
      "report_ab12cd.pdf",
    );
  });

  it("takes the filename off a Windows path", () => {
    // The case that produced no attachment chip at all: str(Path) on Windows
    // hands the frontend backslashes, and splitting on "/" returned the whole
    // path as the "filename".
    assert.equal(
      uploadedFileName(
        `C:\\Users\\me\\AppData\\Roaming\\OpenKimo\\sessions\\${SESSION_ID}\\uploads\\report_ab12cd.pdf`,
      ),
      "report_ab12cd.pdf",
    );
  });

  it("handles a UNC path", () => {
    assert.equal(
      uploadedFileName("\\\\server\\share\\uploads\\notes_1a2b3c.md"),
      "notes_1a2b3c.md",
    );
  });

  it("returns a bare filename unchanged", () => {
    assert.equal(uploadedFileName("notes.md"), "notes.md");
  });

  it("falls back to the input when there is nothing to take", () => {
    assert.equal(uploadedFileName(""), "");
  });
});

describe("isUploadedFilePath", () => {
  it("accepts a POSIX absolute path", () => {
    assert.equal(isUploadedFilePath(`/home/me/${SESSION_ID}/uploads/a.png`), true);
  });

  it("accepts a Windows absolute path", () => {
    assert.equal(isUploadedFilePath("C:\\Users\\me\\uploads\\a.png"), true);
    assert.equal(isUploadedFilePath("d:/Users/me/uploads/a.png"), true);
  });

  it("accepts a UNC path", () => {
    assert.equal(isUploadedFilePath("\\\\server\\share\\uploads\\a.png"), true);
  });

  it("accepts the legacy relative form with either separator", () => {
    assert.equal(isUploadedFilePath("uploads/a.png"), true);
    assert.equal(isUploadedFilePath("uploads\\a.png"), true);
  });

  it("rejects prose the user wrote as a numbered list", () => {
    // These lines get stripped from the visible message, so a false positive
    // eats the user's own text.
    assert.equal(isUploadedFilePath("Install the dependencies"), false);
    assert.equal(isUploadedFilePath("uploadsomething"), false);
    assert.equal(isUploadedFilePath(""), false);
  });
});

describe("MEDIA_TAG_PATH_REGEX", () => {
  const matchAll = (text: string) => {
    MEDIA_TAG_PATH_REGEX.lastIndex = 0;
    return [...text.matchAll(MEDIA_TAG_PATH_REGEX)];
  };

  it("pulls the session id and filename out of a POSIX tag", () => {
    const matches = matchAll(
      `<image path="/home/me/${SESSION_ID}/uploads/shot_ab12cd.png" content_type="image/png">`,
    );
    assert.equal(matches.length, 1);
    assert.equal(matches[0][2], SESSION_ID);
    assert.equal(matches[0][3], "shot_ab12cd.png");
  });

  it("pulls the same out of a Windows tag", () => {
    const matches = matchAll(
      `<video path="C:\\Users\\me\\sessions\\${SESSION_ID}\\uploads\\clip_ab12cd.mp4" content_type="video/mp4">`,
    );
    assert.equal(matches.length, 1);
    assert.equal(matches[0][2], SESSION_ID);
    assert.equal(matches[0][3], "clip_ab12cd.mp4");
  });

  it("ignores a tag that does not point into a session's uploads", () => {
    assert.equal(matchAll('<image path="/tmp/elsewhere/shot.png">').length, 0);
  });
});

describe("LEGACY_UPLOADS_REGEX", () => {
  it("matches the legacy mention with either separator", () => {
    assert.equal(LEGACY_UPLOADS_REGEX.exec("see `uploads/a.png`")?.[1], "a.png");
    assert.equal(LEGACY_UPLOADS_REGEX.exec("see `uploads\\a.png`")?.[1], "a.png");
  });
});
