import type { WorkerEnv } from "./kaggle";

const KAGGLE_API_ROOT = "https://api.kaggle.com/v1";
const KAGGLE_SERVICE = "kernels.KernelsApiService";
const OWNER = "azadka";
const KERNEL_SLUG = "pneumonia-v6-2-2-train-w01-m01-r224";
const MAX_TEXT_BYTES = 768 * 1024;
const MAX_PAGES = 20;

interface OutputFile {
  fileName: string;
  url: string;
}

function token(env: WorkerEnv): string {
  const value = env.CGP_KAGGLE_MASTER_TOKEN?.trim();
  if (!value) throw new Error("Master token is not configured");
  return value;
}

function authorization(env: WorkerEnv): string {
  const value = token(env);
  return value.startsWith("KGAT_") ? `Bearer ${value}` : `Basic ${btoa(`${OWNER}:${value}`)}`;
}

async function kaggleCall(env: WorkerEnv, body: Record<string, unknown>): Promise<Record<string, unknown>> {
  const response = await fetch(`${KAGGLE_API_ROOT}/${KAGGLE_SERVICE}/ListKernelSessionOutput`, {
    method: "POST",
    headers: {
      Authorization: authorization(env),
      "Content-Type": "application/json",
      "User-Agent": "chatgpt-kaggle-project-plan/0.1",
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let payload: unknown = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Kaggle output listing returned non-JSON HTTP ${response.status}`);
  }
  if (!response.ok || !payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error(`Kaggle output listing failed HTTP ${response.status}`);
  }
  return payload as Record<string, unknown>;
}

function outputFiles(value: unknown): OutputFile[] {
  if (!Array.isArray(value)) return [];
  const files: OutputFile[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const record = item as Record<string, unknown>;
    const fileName = typeof record.fileName === "string" ? record.fileName : "";
    const url = typeof record.url === "string" ? record.url : "";
    if (fileName && url) files.push({ fileName, url });
  }
  return files;
}

function nextPageToken(value: Record<string, unknown>): string {
  if (typeof value.nextPageToken === "string") return value.nextPageToken;
  if (typeof value.next_page_token === "string") return value.next_page_token;
  return "";
}

async function enumerateOutputs(env: WorkerEnv): Promise<OutputFile[]> {
  const files: OutputFile[] = [];
  let pageToken = "";
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const request: Record<string, unknown> = {
      userName: OWNER,
      kernelSlug: KERNEL_SLUG,
      pageSize: 100,
    };
    if (pageToken) request.pageToken = pageToken;
    const payload = await kaggleCall(env, request);
    files.push(...outputFiles(payload.files));
    const next = nextPageToken(payload);
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  return files;
}

function allowedOutputUrl(raw: string): URL {
  const url = new URL(raw);
  if (url.protocol !== "https:") throw new Error("output URL must be HTTPS");
  const host = url.hostname.toLocaleLowerCase("en-US");
  const ok =
    host === "api.kaggle.com" ||
    host === "www.kaggle.com" ||
    host === "storage.googleapis.com" ||
    host.endsWith(".kaggleusercontent.com") ||
    host.endsWith(".googleusercontent.com");
  if (!ok) throw new Error("output URL host is not allowlisted");
  return url;
}

async function readSmallOutput(env: WorkerEnv, files: OutputFile[], suffix: string): Promise<string> {
  const match = files.find((file) => file.fileName.endsWith(suffix));
  if (!match) throw new Error(`required recovered artifact not found: ${suffix}`);
  const response = await fetch(allowedOutputUrl(match.url).toString(), { redirect: "follow" });
  if (!response.ok) throw new Error(`artifact download failed HTTP ${response.status}: ${suffix}`);
  const declared = Number(response.headers.get("content-length") ?? "0");
  if (declared > MAX_TEXT_BYTES) throw new Error(`artifact exceeds text budget: ${suffix}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > MAX_TEXT_BYTES) throw new Error(`artifact exceeds text budget: ${suffix}`);
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

const RELEVANT_KEY = /(model|backbone|architect|resolution|image|input|matrix|train|hpo|confirm|ensemble|champion|merge|stage|phase|recipe|seed|epoch|batch|learning|\blr\b|threshold|calibrat|external|benchmark|inference|selection)/i;

function relevantJson(value: unknown): Array<{ path: string; value: string | number | boolean | null }> {
  const result: Array<{ path: string; value: string | number | boolean | null }> = [];
  const visit = (node: unknown, path: string, inherited: boolean, depth: number): void => {
    if (result.length >= 220 || depth > 8) return;
    if (node === null || typeof node === "string" || typeof node === "number" || typeof node === "boolean") {
      if (inherited && (typeof node !== "string" || node.length <= 500)) result.push({ path, value: node });
      return;
    }
    if (Array.isArray(node)) {
      node.slice(0, 40).forEach((item, index) => visit(item, `${path}[${index}]`, inherited, depth + 1));
      return;
    }
    if (!node || typeof node !== "object") return;
    for (const [key, child] of Object.entries(node as Record<string, unknown>)) {
      const relevant = inherited || RELEVANT_KEY.test(key);
      visit(child, path ? `${path}.${key}` : key, relevant, depth + 1);
    }
  };
  visit(value, "", false, 0);
  return result;
}

function signalLines(texts: string[]): string[] {
  const pattern = /(M0[1-9]|convnext|efficientnet|resnet|densenet|mobilenet|swin|maxvit|coatnet|regnet|vit|transformer|backbone|ensemble|champion|confirm|hpo|merge|resolution|matrix)/i;
  const out: string[] = [];
  const seen = new Set<string>();
  for (const text of texts) {
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.length > 420 || !pattern.test(line) || seen.has(line)) continue;
      seen.add(line);
      out.push(line);
      if (out.length >= 180) return out;
    }
  }
  return out;
}

function matches(text: string, regex: RegExp): string[] {
  const values = new Set<string>();
  for (const match of text.matchAll(regex)) values.add(match[0]);
  return [...values];
}

export async function v622ProjectPlan(env: WorkerEnv): Promise<Record<string, unknown>> {
  const files = await enumerateOutputs(env);
  const recipeText = await readSmallOutput(env, files, "/FROZEN_SHARED_TRAINING_RECIPE.json");
  const configText = await readSmallOutput(env, files, "/training/M01/W01_M01_r224__M01/config.json");
  const reportText = await readSmallOutput(env, files, "/training/M01/W01_M01_r224__M01/final_report.json");
  const matrixText = await readSmallOutput(env, files, "/training/matrix_execution_report.json");

  const sourceSuffixes = [
    "/SOURCE_PYTHON/shared_hpo_recipe.py",
    "/SOURCE_PYTHON/backbone_comparison_contracts.py",
    "/SOURCE_PYTHON/pneumonia_runner_base.py",
    "/SOURCE_PYTHON/auto_ensemble_selection.py",
    "/SOURCE_PYTHON/backbone_ensemble_selection.py",
    "/SOURCE_PYTHON/select_champion.py",
    "/SOURCE_PYTHON/v6_master.py",
    "/SOURCE_PYTHON/v6_worker.py",
  ];
  const sources: string[] = [];
  const sourceFiles: string[] = [];
  for (const suffix of sourceSuffixes) {
    try {
      sources.push(await readSmallOutput(env, files, suffix));
      sourceFiles.push(suffix.split("/").pop() ?? suffix);
    } catch {
      // Optional source module may not be present in a particular recovered package.
    }
  }

  const allText = [recipeText, configText, reportText, matrixText, ...sources].join("\n");
  const modelCodes = matches(allText.toUpperCase(), /\bM0[1-9]\b/g).sort();
  const architectureTerms = matches(
    allText.toLocaleLowerCase("en-US"),
    /\b(?:convnext\w*|efficientnet\w*|resnet\w*|densenet\w*|mobilenet\w*|swin\w*|maxvit\w*|coatnet\w*|regnet\w*|vit\w*|transformer\w*)\b/g,
  ).sort();
  const resolutions = matches(allText, /\b(?:160|192|224|256|288|320|384|448|512)\b/g)
    .map((value) => Number(value))
    .filter((value, index, array) => array.indexOf(value) === index)
    .sort((a, b) => a - b);

  return {
    source_kernel: `${OWNER}/${KERNEL_SLUG}`,
    recovered_output_count: files.length,
    source_files_inspected: sourceFiles,
    planned_model_codes: modelCodes,
    architecture_terms: architectureTerms,
    resolutions,
    recipe_signals: relevantJson(parseJson(recipeText)),
    config_signals: relevantJson(parseJson(configText)),
    final_report_signals: relevantJson(parseJson(reportText)),
    matrix_report_signals: relevantJson(parseJson(matrixText)),
    source_signal_lines: signalLines(sources),
  };
}
