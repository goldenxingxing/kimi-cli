const HEIC_MIME_TYPES = new Set(["image/heic", "image/heif"]);
const HEIC_EXT_RE = /\.(heic|heif)$/i;

function isHeicFile(file: File): boolean {
  if (file.type && HEIC_MIME_TYPES.has(file.type.toLowerCase())) {
    return true;
  }
  // Some browsers / OSes leave file.type empty for HEIC. Fall back to extension.
  if (!file.type && HEIC_EXT_RE.test(file.name)) {
    return true;
  }
  return false;
}

async function convertHeicToJpeg(file: File): Promise<File> {
  // Dynamic import keeps heic2any out of the main bundle until needed.
  const { default: heic2any } = await import("heic2any");
  const result = await heic2any({
    blob: file,
    toType: "image/jpeg",
    quality: 0.9,
  });
  // heic2any may return Blob or Blob[] depending on whether the HEIC contained
  // multiple images. Pick the first frame in the multi-frame case.
  const blob = Array.isArray(result) ? result[0] : result;
  const newName = file.name.replace(HEIC_EXT_RE, ".jpg");
  // Ensure we always end in .jpg (in case original had no extension).
  const finalName = HEIC_EXT_RE.test(file.name)
    ? newName
    : `${file.name || "image"}.jpg`;
  return new File([blob], finalName, { type: "image/jpeg" });
}

/**
 * Walk a File[] / FileList and transparently transcode any HEIC/HEIF entries
 * to JPEG. Non-HEIC files pass through untouched. If client-side conversion
 * fails (heic2any has poor compatibility with some Apple HEIC variants), the
 * raw file is passed through so the server can decode it with pillow-heif.
 */
export async function transcodeHeicFiles(
  files: File[] | FileList,
): Promise<File[]> {
  const incoming = Array.from(files);
  const out: File[] = [];
  for (const file of incoming) {
    if (!isHeicFile(file)) {
      out.push(file);
      continue;
    }
    try {
      const converted = await convertHeicToJpeg(file);
      out.push(converted);
    } catch (err) {
      // Client-side heic2any has poor compatibility with some Apple HEIC
      // variants (especially HEVC). Pass the raw file through so the server
      // (which has pillow-heif) can do the transcoding.
      console.warn(
        "[heic] client conversion failed; falling back to server-side decode",
        file.name,
        err,
      );
      out.push(file);
    }
  }
  return out;
}
