interface Env {
  CGP_PROJECT_CONTROL_TOKEN?: string;
  CGP_KAGGLE_MASTER_TOKEN?: string;
}

type Located = { fileName: string; url: string };

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const OWNER = "azadka";
const KERNEL_SLUG = "pneumonia-v6-2-2-train-w01-m01-r224";
const SELECT_SHA = "f6be9ddf5060dbea5fab233ee0d349820ef679c64662eec23dd01c2929aec9b2";
const PAGE_SIZE = 100;
const MAX_PAGES = 20;
const MAX_BYTES = 8 * 1024 * 1024;
const ALLOWED = new Set([
  "select_champion.py",
  "auto_ensemble_selection.py",
  "backbone_ensemble_selection.py",
  "balanced_metrics.py",
  "backbone_comparison_contracts.py",
  "freeze_selected_recipe.py",
  "run_backbone_comparison.py",
  "governance_contracts.py",
  "final_experiment_contracts.py",
  "inference_preprocessing_contract.py",
  "source_package_integrity.py",
  "v6_fingerprint_contracts.py",
  "run_final_canonical_workflow.py",
  "run_qualification_model_selection.py"
]);

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`;
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return !!expected && expected.length >= 32 && request.headers.get("authorization") === `Bearer ${expected}`;
}

function nextPageToken(value: Record<string, unknown>): string {
  if (typeof value.nextPageToken === "string") return value.nextPageToken;
  if (typeof value.next_page_token === "string") return value.next_page_token;
  return "";
}

function rows(value: unknown): Located[] {
  if (!Array.isArray(value)) return [];
  const out: Located[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const row = item as Record<string, unknown>;
    if (typeof row.fileName === "string" && typeof row.url === "string") out.push({ fileName: row.fileName.replaceAll("\\", "/"), url: row.url });
  }
  return out;
}

function parent(fileName: string): string {
  const index = fileName.lastIndexOf("/");
  return index < 0 ? "" : fileName.slice(0, index + 1);
}

function allowedUrl(raw: string): URL {
  const url = new URL(raw);
  const host = url.hostname.toLowerCase();
  if (url.protocol !== "https:" || !(host === "api.kaggle.com" || host === "www.kaggle.com" || host === "storage.googleapis.com" || host.endsWith(".kaggleusercontent.com") || host.endsWith(".googleusercontent.com"))) throw new Error("artifact URL is not allowlisted");
  return url;
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
}

async function download(rawUrl: string, name: string): Promise<ArrayBuffer> {
  const response = await fetch(allowedUrl(rawUrl).toString(), { redirect: "follow" });
  if (!response.ok) throw new Error(`${name} download HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > MAX_BYTES) throw new Error(`${name} exceeds size limit`);
  return bytes;
}

async function listMatches(env: Env, name: string): Promise<Located[]> {
  const token = env.CGP_KAGGLE_MASTER_TOKEN?.trim();
  if (!token) throw new Error("master Kaggle token missing");
  let pageToken = "";
  const matches: Located[] = [];
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const body: Record<string, unknown> = { userName: OWNER, kernelSlug: KERNEL_SLUG, pageSize: PAGE_SIZE };
    if (pageToken) body.pageToken = pageToken;
    const response = await fetch(`${API_ROOT}/ListKernelSessionOutput`, {
      method: "POST",
      headers: { Authorization: authHeader(token), "Content-Type": "application/json", "User-Agent": "v622-governance-source-bridge/1.0" },
      body: JSON.stringify(body),
    });
    const value = await response.json() as Record<string, unknown>;
    if (!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`);
    for (const row of rows(value.files)) if (row.fileName === name || row.fileName.endsWith(`/${name}`)) matches.push(row);
    const next = nextPageToken(value);
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  if (!matches.length) throw new Error(`${name} not found`);
  return matches;
}

async function canonicalDirectory(env: Env): Promise<string> {
  for (const candidate of await listMatches(env, "select_champion.py")) {
    const bytes = await download(candidate.url, "select_champion.py");
    if (await sha256(bytes) === SELECT_SHA) return parent(candidate.fileName);
  }
  throw new Error("canonical governance source directory not found");
}

async function readSource(env: Env, name: string): Promise<Record<string, unknown>> {
  if (!ALLOWED.has(name)) throw new Error("source_name is not allowlisted");
  const directory = await canonicalDirectory(env);
  const scoped = (await listMatches(env, name)).filter(item => parent(item.fileName) === directory);
  if (scoped.length !== 1) throw new Error(`${name} must resolve exactly once in canonical source directory; found ${scoped.length}`);
  const bytes = await download(scoped[0].url, name);
  return {
    project: "PNEUMONIA V6.2.2",
    purpose: "canonical_governance_source_recovery",
    source_name: name,
    file_name: scoped[0].fileName,
    sha256: await sha256(bytes),
    bytes: bytes.byteLength,
    content: new TextDecoder("utf-8", { fatal: false }).decode(bytes),
    canonical_source_directory_pinned: true,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") return json({ service: "v622-governance-source-bridge", status: "ready", protected: true });
    if (url.pathname === "/control/v6-2-2/governance-source" && request.method === "POST") {
      if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
      try {
        const raw = await request.json() as Record<string, unknown>;
        return json(await readSource(env, String(raw.source_name ?? "")));
      } catch (error) {
        return json({ ok: false, error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error" }, 502);
      }
    }
    return new Response("Not found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;
