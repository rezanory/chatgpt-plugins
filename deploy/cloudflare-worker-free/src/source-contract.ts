import { getKernel, type WorkerEnv } from "./kaggle";

const TEMPLATE_REF = "azadka/pneumonia-v6-2-2-train-w01-m01-r224";

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function notebookText(source: string): { text: string; notebookParsed: boolean; cellCount: number } {
  try {
    const parsed: unknown = JSON.parse(source);
    const root = record(parsed);
    const cells = Array.isArray(root.cells) ? root.cells : [];
    const parts: string[] = [];
    for (const cell of cells) {
      const rec = record(cell);
      const value = rec.source;
      if (typeof value === "string") parts.push(value);
      if (Array.isArray(value)) {
        for (const item of value) if (typeof item === "string") parts.push(item);
      }
    }
    if (parts.length) return { text: parts.join("\n"), notebookParsed: true, cellCount: cells.length };
  } catch {
    // Plain source fallback below.
  }
  return { text: source, notebookParsed: false, cellCount: 0 };
}

function selectedLines(source: string): string[] {
  const pattern = /(M(?:0[0-9]|1[0-2])|model.?id|image.?size|resolution|seed|output|runner|SOURCE_PYTHON|FROZEN_SHARED|python|argv|sys\.argv|argparse|train|matrix|working|worker|shard|run.?name|head.?epochs|finetune.?epochs)/i;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.length > 700 || !pattern.test(line) || seen.has(line)) continue;
    seen.add(line);
    out.push(line);
    if (out.length >= 260) break;
  }
  return out;
}

export async function v622KernelSourceContract(env: WorkerEnv): Promise<Record<string, unknown>> {
  const kernel = await getKernel(env, "master", TEMPLATE_REF);
  const metadata = record(kernel.metadata);
  const blob = record(kernel.blob);
  const source = typeof blob.source === "string" ? blob.source : "";
  if (!source) throw new Error("template GetKernel returned no source");
  const normalized = notebookText(source);

  const safeMetadata: Record<string, unknown> = {};
  for (const key of [
    "title",
    "language",
    "kernelType",
    "enableGpu",
    "enableInternet",
    "machineShape",
    "datasetDataSources",
    "kernelDataSources",
    "competitionDataSources",
    "modelDataSources",
  ]) {
    if (metadata[key] !== undefined) safeMetadata[key] = metadata[key];
  }

  return {
    template_ref: TEMPLATE_REF,
    source_chars: source.length,
    notebook_parsed: normalized.notebookParsed,
    cell_count: normalized.cellCount,
    metadata: safeMetadata,
    source_signal_lines: selectedLines(normalized.text),
  };
}
