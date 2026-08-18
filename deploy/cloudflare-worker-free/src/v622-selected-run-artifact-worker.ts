interface Env {
  CGP_PROJECT_CONTROL_TOKEN?: string;
  CGP_KAGGLE_KG05_TOKEN?: string;
}

type Located = { fileName: string; url: string };

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const OWNER = "trickermark";
const KERNEL_SLUG = "pneumonia-v6-2-2-train-w16-m06-r224";
const EXACT_SUFFIX = "/training/M06/W16_M06_r224__M06/config.json";
const MAX_PAGES = 20;
const PAGE_SIZE = 100;
const MAX_BYTES = 2 * 1024 * 1024;

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return !!expected && expected.length >= 32 && request.headers.get("authorization") === `Bearer ${expected}`;
}

function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`;
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
    if (typeof row.fileName === "string" && typeof row.url === "string") {
      out.push({ fileName: row.fileName.replaceAll("\\", "/"), url: row.url });
    }
  }
  return out;
}

function allowedUrl(raw: string): URL {
  const url = new URL(raw);
  const host = url.hostname.toLowerCase();
  const ok = url.protocol === "https:" && (
    host === "api.kaggle.com" || host === "www.kaggle.com" || host === "storage.googleapis.com" ||
    host.endsWith(".kaggleusercontent.com") || host.endsWith(".googleusercontent.com")
  );
  if (!ok) throw new Error("artifact URL is not allowlisted");
  return url;
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
}

async function locate(env: Env): Promise<Located> {
  const token = env.CGP_KAGGLE_KG05_TOKEN?.trim();
  if (!token) throw new Error("Kaggle kg-05 token missing");
  let pageToken = "";
  const matches: Located[] = [];
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const body: Record<string, unknown> = { userName: OWNER, kernelSlug: KERNEL_SLUG, pageSize: PAGE_SIZE };
    if (pageToken) body.pageToken = pageToken;
    const response = await fetch(`${API_ROOT}/ListKernelSessionOutput`, {
      method: "POST",
      headers: {
        Authorization: authHeader(token),
        "Content-Type": "application/json",
        "User-Agent": "v622-selected-run-artifact-bridge/1.0",
      },
      body: JSON.stringify(body),
    });
    const value = await response.json() as Record<string, unknown>;
    if (!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`);
    for (const row of rows(value.files)) {
      const normalized = row.fileName.replaceAll("\\", "/");
      if (normalized.endsWith(EXACT_SUFFIX)) matches.push(row);
    }
    const next = nextPageToken(value);
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  if (matches.length !== 1) throw new Error(`exact W16 selected-run config must resolve once; found ${matches.length}`);
  return matches[0];
}

async function readConfig(env: Env): Promise<Record<string, unknown>> {
  const located = await locate(env);
  const response = await fetch(allowedUrl(located.url).toString(), { redirect: "follow" });
  if (!response.ok) throw new Error(`config.json download HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > MAX_BYTES) throw new Error("config.json exceeds size limit");
  return {
    project: "PNEUMONIA V6.2.2",
    purpose: "selected_run_recipe_source_recovery",
    worker_id: "W16",
    model_id: "M06",
    resolution: 224,
    kernel_ref: `${OWNER}/${KERNEL_SLUG}`,
    artifact_name: "config.json",
    file_name: located.fileName,
    sha256: await sha256(bytes),
    bytes: bytes.byteLength,
    exact_training_run_path_pinned: true,
    content: new TextDecoder("utf-8", { fatal: false }).decode(bytes),
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") return json({ service: "v622-selected-run-artifact-bridge", status: "ready", protected: true });
    if (url.pathname === "/control/v6-2-2/selected-run-config" && request.method === "POST") {
      if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
      try { return json(await readConfig(env)); }
      catch (error) { return json({ ok: false, error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error" }, 502); }
    }
    return new Response("Not found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;
