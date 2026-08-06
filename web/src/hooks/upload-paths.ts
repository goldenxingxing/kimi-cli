/**
 * Reading the uploaded-file paths the server puts in a user message.
 *
 * The server names an upload by its real path on the host — `str(Path)` in
 * the `<uploaded_files>` list and in the `path=` attribute of the
 * `<image>` / `<video>` / `<document>` tags. On Windows that path is
 * `C:\...\<session>\uploads\report.pdf`, and every rule here used to assume
 * a POSIX one: a leading `/` to recognise the path at all, `split("/")` to
 * take the filename off it. Both silently produced nothing on Windows, so an
 * upload left no attachment chip in the history and the reader had no way to
 * tell whether the file had arrived.
 *
 * Changing what the server emits is not an option — the agent needs the
 * native path to open the file — so the separator is normalised here, at the
 * point of reading.
 */

/** Either separator. A path may legitimately arrive with `\` on Windows. */
const SEPARATOR_REGEX = /[\\/]/;

/** `C:\...` or `C:/...` — a Windows absolute path. */
const WINDOWS_DRIVE_REGEX = /^[A-Za-z]:[\\/]/;

/** `uploads/...` or `uploads\...` — the legacy relative form. */
const RELATIVE_UPLOADS_REGEX = /^uploads[\\/]/;

/**
 * Match `<image path="...">` / `<video path="...">` and pull out the session
 * id and filename, with either separator around the `uploads` segment.
 *
 * Groups: 1 = full path, 2 = session id, 3 = filename.
 */
export const MEDIA_TAG_PATH_REGEX =
  /<(?:image|video)\s+[^>]*path="([^"]*[\\/]([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})[\\/]uploads[\\/]([^"]+))"/g;

/** Match a legacy `` `uploads/file.name` `` mention, either separator. */
export const LEGACY_UPLOADS_REGEX = /`uploads[\\/]([^`]+)`/;

/**
 * The filename at the end of a path, whichever separator produced it.
 *
 * Uploads are stored under a name filtered to alphanumerics, dot, underscore,
 * hyphen and space (`sanitize_filename` on the server), so neither separator
 * can appear inside the filename itself and splitting on both is safe.
 */
export function uploadedFileName(path: string): string {
  const segments = path.split(SEPARATOR_REGEX).filter(Boolean);
  return segments.at(-1) ?? path;
}

/**
 * Whether a numbered-list item names an uploaded file rather than being
 * ordinary prose the user happened to write as a list.
 *
 * Deliberately narrow: these lines are removed from the visible text, so a
 * false positive eats something the user wrote.
 */
export function isUploadedFilePath(candidate: string): boolean {
  if (!candidate) {
    return false;
  }
  return (
    candidate.startsWith("/") ||
    candidate.startsWith("\\\\") || // UNC share
    WINDOWS_DRIVE_REGEX.test(candidate) ||
    RELATIVE_UPLOADS_REGEX.test(candidate)
  );
}
