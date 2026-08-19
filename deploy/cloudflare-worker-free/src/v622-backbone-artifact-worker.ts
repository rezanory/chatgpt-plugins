import { kernelStatus, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

const ACCOUNT_ID = "kg-05";
const USERNAME = "trickermark";
const KERNEL_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const KERNEL_SLUG = "pneumonia-v6-2-2-backbone-m06-r224";
const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const MAX_PAGES = 20;
const PAGE_SIZE = 100;
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const MAX_TOTAL_BYTES = 12 * 1024 * 1024;

const ARTIFACT_NAMES = [
  "BACKBONE_COMPARISON_RUN_INDEX.csv",
  "BACKBONE_COMPARISON_SUMMARY.csv",
  "BACKBONE_COMPARISON_MANIFEST.json",
  "BACKBONE_INDIVIDUAL_OOF_RANKING.csv",
  "BACKBONE_ENSEMBLE_OOF_PREDICTIONS.csv",
  "BACKBONE_ENSEMBLE_COMBINATION_RANKING.csv",
  "FROZEN_BACKBONE_ENSEMBLE_POLICY.json",
] as const;

type Located = { fileName: string; url: string };

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
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
    raw.status,
    raw.statusName,
    raw.status_name,
    raw.state,
    raw.sessionStatus,
    raw.kernelSessionStatus,
    session.status,
    session.statusName,
    session.state,
  ]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim().toUpperCase();
  }
  return "UNKNOWN";
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(
    expected &&
    expected.length >= 32 &&
    request.headers.get("authorization") === `Bearer ${expected}`
  );
}

function authHeader(token: string): string {
  return token.startsWith("KGAT_")
    ? `Bearer ${token}`
    : `Basic ${btoa(`${USERNAME}:${token}`)}`;
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

function allowedOutputUrl(raw: string): URL {
  const url = new URL(raw);
  const host = url.hostname.toLowerCase();
  const allowed =
    url.protocol === "https:" &&
    (
      host === "api.kaggle.com" ||
      host === "www.kaggle.com" ||
      host === "storage.googleapis.com" ||
      host.endsWith(".kaggleusercontent.com") ||
      host.endsWith(".googleusercontent.com")
    );
  if (!allowed) throw new Error("artifact URL is not allowlisted");
  return url;
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
        "User-Agent": "v622-backbone-artifact-collector/1.0",
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

function exactMatch(files: Located[], name: string): Located {
  const matches = files.filter((file) => file.fileName === name || file.fileName.endsWith(`/${name}`));
  if (matches.length !== 1) {
    throw new Error(`${name} must resolve exactly once; found ${matches.length}`);
  }
  return matches[0];
}

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), b => b.toString(16).padStart(2, "0")).join("");
}

async function collect(env: Env): Promise<Record<string, unknown>> {
  const rawStatus = await kernelStatus(env, ACCOUNT_ID, KERNEL_REF);
  const status = normalizedStatus(rawStatus);
  if (status !== "COMPLETE") throw new Error(`kernel is not COMPLETE: ${status}`);

  const files = await listOutput(env);
  let totalBytes = 0;
  const artifacts: Array<Record<string, unknown>> = [];

  for (const name of ARTIFACT_NAMES) {
    const located = exactMatch(files, name);
    const response = await fetch(allowedOutputUrl(located.url).toString(), { redirect: "follow" });
    if (!response.ok) throw new Error(`${name} download HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > MAX_FILE_BYTES) throw new Error(`${name} exceeds file limit`);
    totalBytes += bytes.byteLength;
    if (totalBytes > MAX_TOTAL_BYTES) throw new Error("artifact total exceeds limit");
    artifacts.push({
      artifact_name: name,
      file_name: located.fileName,
      bytes: bytes.byteLength,
      sha256: hex(await crypto.subtle.digest("SHA-256", bytes)),
      content: new TextDecoder("utf-8", { fatal: false }).decode(bytes),
    });
  }

  return {
    project: "PNEUMONIA V6.2.2",
    stage: "post_ablation_backbone_comparison_artifact_recovery",
    account_id: ACCOUNT_ID,
    kernel_ref: KERNEL_REF,
    status,
    artifact_count: artifacts.length,
    artifacts,
    raw_predictions_publicly_exposed: false,
    kaggle_compute_launched_by_collector: false,
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
      return json({ service: "v622-backbone-artifact-collector", status: "ready", protected: true });
    }
    if (url.pathname !== "/control/v6-2-2/backbone-artifacts" || request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try {
      return json(await collect(env));
    } catch (error) {
      return json({
        ok: false,
        error_type: error instanceof Error ? error.name : "Error",
        error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
        kaggle_compute_launched_by_collector: false,
      }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
