import type { WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;

const ACCOUNT_ID = "kg-04";
const USERNAME = "reyhanehazad";
const KERNEL_SLUG = "notebook5007d729f7";
const KERNEL_REF = `${USERNAME}/${KERNEL_SLUG}`;
const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const MAX_SOURCE_CHARS = 2_500_000;
const MAX_LOG_CHARS = 160_000;

function rec(value: unknown): Rec {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Rec : {};
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(expected && expected.length >= 32 && request.headers.get("authorization") === `Bearer ${expected}`);
}

function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${USERNAME}:${token}`)}`;
}

async function call(env: Env, method: string, body: Rec): Promise<Rec> {
  const token = env.CGP_KAGGLE_KG04_TOKEN?.trim();
  if (!token) throw new Error("Kaggle kg-04 token missing");
  const response = await fetch(`${API_ROOT}/${method}`, {
    method: "POST",
    headers: {
      Authorization: authHeader(token),
      "Content-Type": "application/json",
      "User-Agent": "cgp-notebook5007-readonly-inspector/1.0",
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let value: unknown = {};
  try { value = text ? JSON.parse(text) : {}; }
  catch { throw new Error(`${method} returned non-JSON HTTP ${response.status}`); }
  const obj = rec(value);
  const code = typeof obj.code === "number" ? obj.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) {
    const message = typeof obj.message === "string" ? obj.message : `HTTP ${response.status}`;
    throw new Error(`${method} failed: ${message.slice(0, 1200)}`);
  }
  return obj;
}

async function sha256(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
}

function notebookSummary(source: string): Rec {
  try {
    const nb = JSON.parse(source) as Rec;
    const cells = Array.isArray(nb.cells) ? nb.cells : [];
    let codeCells = 0, markdownCells = 0, rawCells = 0, executedCodeCells = 0, errorOutputs = 0;
    const errorNames: string[] = [];
    const cellPreview: Array<Rec> = [];
    cells.forEach((cellValue, index) => {
      const cell = rec(cellValue);
      const type = String(cell.cell_type ?? "unknown");
      if (type === "code") codeCells += 1;
      else if (type === "markdown") markdownCells += 1;
      else if (type === "raw") rawCells += 1;
      if (type === "code" && cell.execution_count !== null && cell.execution_count !== undefined) executedCodeCells += 1;
      const outputs = Array.isArray(cell.outputs) ? cell.outputs : [];
      for (const outputValue of outputs) {
        const output = rec(outputValue);
        if (String(output.output_type ?? "") === "error") {
          errorOutputs += 1;
          const name = String(output.ename ?? "Error");
          if (!errorNames.includes(name) && errorNames.length < 20) errorNames.push(name);
        }
      }
      if (cellPreview.length < 30) {
        const src = Array.isArray(cell.source) ? cell.source.join("") : String(cell.source ?? "");
        cellPreview.push({ index, type, execution_count: cell.execution_count ?? null, source_preview: src.slice(0, 500) });
      }
    });
    return {
      parsed: true,
      nbformat: nb.nbformat ?? null,
      nbformat_minor: nb.nbformat_minor ?? null,
      cell_count: cells.length,
      code_cells: codeCells,
      markdown_cells: markdownCells,
      raw_cells: rawCells,
      executed_code_cells: executedCodeCells,
      error_outputs: errorOutputs,
      error_names: errorNames,
      notebook_metadata: rec(nb.metadata),
      first_30_cells: cellPreview,
    };
  } catch (error) {
    return { parsed: false, parse_error: error instanceof Error ? error.message : "unknown parse error" };
  }
}

async function collect(env: Env): Promise<Rec> {
  const base = { userName: USERNAME, kernelSlug: KERNEL_SLUG };
  const [kernel, status, output, listing] = await Promise.all([
    call(env, "GetKernel", base),
    call(env, "GetKernelSessionStatus", base),
    call(env, "ListKernelSessionOutput", { ...base, pageSize: 100 }),
    call(env, "ListKernels", { group: "PROFILE", sortBy: "DATE_RUN", pageSize: 20, search: KERNEL_SLUG }),
  ]);

  const blob = rec(kernel.blob);
  const metadata = rec(kernel.metadata);
  const rawSource = typeof blob.source === "string" ? blob.source : "";
  const sourceTruncated = rawSource.length > MAX_SOURCE_CHARS;
  const source = sourceTruncated ? rawSource.slice(0, MAX_SOURCE_CHARS) : rawSource;
  const logRaw = String(output.log ?? "");
  const log = logRaw.length > MAX_LOG_CHARS ? logRaw.slice(-MAX_LOG_CHARS) : logRaw;
  const files = Array.isArray(output.files) ? output.files.map((v) => {
    const r = rec(v);
    return { fileName: r.fileName ?? null, size: r.size ?? r.fileSize ?? null };
  }) : [];

  return {
    inspected_at: new Date().toISOString(),
    read_only: true,
    kaggle_compute_launched: false,
    account_id: ACCOUNT_ID,
    kernel_ref: KERNEL_REF,
    list_match_count: Array.isArray(listing.kernels) ? listing.kernels.length : 0,
    list_matches: Array.isArray(listing.kernels) ? listing.kernels.slice(0, 20) : [],
    status,
    metadata,
    blob_metadata: {
      language: blob.language ?? null,
      kernel_type: blob.kernelType ?? blob.kernel_type ?? null,
      slug: blob.slug ?? null,
    },
    source_chars: rawSource.length,
    source_sha256: await sha256(rawSource),
    source_truncated: sourceTruncated,
    source,
    notebook_summary: notebookSummary(rawSource),
    log_chars: logRaw.length,
    log_truncated: logRaw.length > MAX_LOG_CHARS,
    log_tail: log,
    output_file_count_first_page: files.length,
    output_files_first_page: files,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return json({ service: "notebook5007-readonly-inspector", status: "ready", protected: true });
    }
    if (url.pathname !== "/control/inspect/notebook5007" || request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try { return json(await collect(env)); }
    catch (error) {
      return json({ ok: false, read_only: true, kaggle_compute_launched: false, error: error instanceof Error ? error.message.slice(0, 2000) : "unknown error" }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
