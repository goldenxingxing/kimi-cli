import { toast } from "sonner";

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
 * to JPEG. Non-HEIC files pass through untouched. Files that fail to convert
 * are dropped and the user is notified via a toast.
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
      console.error("[heic] conversion failed", file.name, err);
      toast.error("Failed to convert HEIC image", {
        description: file.name,
      });
      // Skip the bad file so it never makes it into attachments / the wire.
    }
  }
  return out;
}
