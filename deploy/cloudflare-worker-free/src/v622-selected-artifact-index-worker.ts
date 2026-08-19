import { kernelStatus, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

const ACCOUNT_ID = "kg-05";
const USERNAME = "trickermark";
const KERNEL_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const KERNEL_SLUG = "pneumonia-v6-2-2-backbone-m06-r224";
const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const PAGE_SIZE = 100;
const MAX_PAGES = 20;
const SELECTED = [
  "M06__convnext_tiny__seed-42",
  "M06__densenet121__seed-42",
  "M06__resnet50v2__seed-42",
] as const;

type Located = { fileName: string; url: string };

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function normalizedStatus(raw: Record<string, unknown>): string {
  const session = record(raw.session);
  for (const candidate of [
    raw.status, raw.statusName, raw.status_name, raw.state,
    raw.sessionStatus, raw.kernelSessionStatus,
    session.status, session.statusName, session.state,
  ]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim().toUpperCase();
  }
  return "UNKNOWN";
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(expected && expected.length >= 32 && request.headers.get("authorization") === `Bearer ${expected}`);
}

function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${USERNAME}:${token}`)}`;
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
    const row = record(item);
    if (typeof row.fileName === "string" && typeof row.url === "string") {
      out.push({ fileName: row.fileName.replaceAll("\\", "/"), url: row.url });
    }
  }
  return out;
}

async function listOutput(env: Env): Promise<Located[]> {
  const token = env.CGP_KAGGLE_KG05_TOKEN?.trim();
  if (!token) throw new Error("Kaggle kg-05 token missing");
  const files: Located[] = [];
  let pageToken = "";
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const body: Record<string, unknown> = {
      userName: USERNAME,
      kernelSlug: KERNEL_SLUG,
      pageSize: PAGE_SIZE,
    };
    if (pageToken) body.pageToken = pageToken;
    const response = await fetch(`${API_ROOT}/ListKernelSessionOutput`, {
      method: "POST",
      headers: {
        Authorization: authHeader(token),
        "Content-Type": "application/json",
        "User-Agent": "v622-selected-artifact-index/1.0",
      },
      body: JSON.stringify(body),
    });
    const text = await response.text();
    const value = record(text ? JSON.parse(text) : {});
    if (!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`);
    files.push(...rows(value.files));
    const next = nextPageToken(value);
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  return files;
}

function candidateFiles(files: Located[], candidateId: string): string[] {
  const marker = `/backbone_comparison/${candidateId}/`;
  return files
    .map((row) => row.fileName)
    .filter((name) => name.includes(marker))
    .sort((a, b) => a.localeCompare(b));
}

async function buildIndex(env: Env): Promise<Record<string, unknown>> {
  const rawStatus = await kernelStatus(env, ACCOUNT_ID, KERNEL_REF);
  const status = normalizedStatus(rawStatus);
  if (status !== "COMPLETE") throw new Error(`kernel is not COMPLETE: ${status}`);
  const files = await listOutput(env);
  const selected: Record<string, string[]> = {};
  for (const candidateId of SELECTED) selected[candidateId] = candidateFiles(files, candidateId);
  return {
    project: "PNEUMONIA V6.2.2",
    stage: "selected_backbone_artifact_index",
    account_id: ACCOUNT_ID,
    kernel_ref: KERNEL_REF,
    status,
    selected_candidate_ids: SELECTED,
    selected_files: selected,
    total_kernel_output_files_enumerated: files.length,
    kaggle_compute_launched: false,
    canonical_m01_m12_rerun: false,
    hpo_or_confirmation_repeated: false,
    locked_test_used: false,
    external_data_used: false,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return json({ service: "v622-selected-artifact-index", status: "ready", protected: true });
    }
    if (url.pathname !== "/control/v6-2-2/selected-artifact-index" || request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try {
      return json(await buildIndex(env));
    } catch (error) {
      return json({
        ok: false,
        error_type: error instanceof Error ? error.name : "Error",
        error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
        kaggle_compute_launched: false,
      }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
