import { getKernel, type WorkerEnv } from "./kaggle";

const TEMPLATE_REF = "azadka/pneumonia-v6-2-2-train-w01-m01-r224";

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function selectedLines(source: string): string[] {
  const pattern = /(M(?:0[0-9]|1[0-2])|model.?id|image.?size|resolution|seed|output|runner|SOURCE_PYTHON|FROZEN_SHARED|python|argv|sys\.argv|argparse|train|matrix|working)/i;
  const out: string[] = [];
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.length > 600 || !pattern.test(line)) continue;
    out.push(line);
    if (out.length >= 180) break;
  }
  return out;
}

export async function v622KernelSourceContract(env: WorkerEnv): Promise<Record<string, unknown>> {
  const kernel = await getKernel(env, "master", TEMPLATE_REF);
  const metadata = record(kernel.metadata);
  const blob = record(kernel.blob);
  const source = typeof blob.source === "string" ? blob.source : "";
  if (!source) throw new Error("template GetKernel returned no source");

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
    metadata: safeMetadata,
    source_signal_lines: selectedLines(source),
  };
}
