import { type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;
const OWNER = "trickermark";
const DATASET_SLUG = "pneumonia-v6-2-2-finalization-artifacts";
const DATASET_REF = `${OWNER}/${DATASET_SLUG}`;
const READER = "azadka";
const TITLE = "PNEUMONIA V6.2.2 Finalization Artifacts";
const DESCRIPTION = "Private frozen selected artifacts for PNEUMONIA V6.2.2 finalization.";
const API_ROOT = "https://api.kaggle.com/v1/datasets.DatasetApiService";

function rec(v: unknown): Rec {
  return v && typeof v === "object" && !Array.isArray(v) ? v as Rec : {};
}

function json(v: unknown, status = 200): Response {
  return new Response(JSON.stringify(v), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function authorized(req: Request, env: Env): boolean {
  const token = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(token && token.length >= 32 && req.headers.get("authorization") === `Bearer ${token}`);
}

function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${OWNER}:${token}`)}`;
}

async function call(env: Env, method: string, body: Rec): Promise<Rec> {
  const token = env.CGP_KAGGLE_KG05_TOKEN?.trim();
  if (!token) throw new Error("trickermark Kaggle token missing");
  const response = await fetch(`${API_ROOT}/${method}`, {
    method: "POST",
    headers: {
      Authorization: authHeader(token),
      "Content-Type": "application/json",
      "User-Agent": "chatgpt-v622-dataset-permission/2.0",
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let value: Rec = {};
  try {
    value = rec(text ? JSON.parse(text) : {});
  } catch {
    throw new Error(`${method} non-JSON HTTP ${response.status}: ${text.slice(0, 500)}`);
  }
  if (!response.ok) {
    throw new Error(`${method} HTTP ${response.status}: ${String(value.message ?? text).slice(0, 900)}`);
  }
  return value;
}

function normalizedCollaborators(value: unknown): Rec[] {
  if (!Array.isArray(value)) return [];
  return value.map(rec).filter((row) => typeof row.username === "string" && row.username.trim());
}

function isAzadkaReader(row: Rec): boolean {
  const username = String(row.username ?? "").trim().toLowerCase();
  const role = row.role;
  return username === READER && (role === 1 || String(role).toUpperCase() === "READER");
}

async function grant(env: Env): Promise<Rec> {
  const dataset = await call(env, "GetDataset", { ownerSlug: OWNER, datasetSlug: DATASET_SLUG });
  const version = Number(dataset.currentVersionNumber ?? 0);
  const totalBytes = Number(dataset.totalBytes ?? 0);
  const versions = Array.isArray(dataset.versions) ? dataset.versions.map(rec) : [];
  const ready = versions.some((row) => Number(row.versionNumber ?? 0) === version && String(row.status ?? "").toUpperCase() === "READY");
  if (version < 1 || totalBytes <= 0 || !ready) {
    throw new Error(`artifact Dataset is not ready: version=${version} totalBytes=${totalBytes} ready=${ready}`);
  }

  const beforeMetadata = await call(env, "GetDatasetMetadata", {
    ownerSlug: OWNER,
    datasetSlug: DATASET_SLUG,
  });
  const beforeInfo = rec(beforeMetadata.info);
  const collaborators = normalizedCollaborators(beforeInfo.collaborators);
  if (!collaborators.some((row) => String(row.username ?? "").trim().toLowerCase() === READER)) {
    collaborators.push({ username: READER, role: 1 });
  } else {
    for (const row of collaborators) {
      if (String(row.username ?? "").trim().toLowerCase() === READER) row.role = 1;
    }
  }

  const title = typeof beforeInfo.title === "string" && beforeInfo.title.trim().length >= 6 && beforeInfo.title.trim().length <= 50
    ? beforeInfo.title.trim()
    : TITLE;
  const description = typeof beforeInfo.description === "string" && beforeInfo.description.trim()
    ? beforeInfo.description
    : DESCRIPTION;

  const settings: Rec = {
    title,
    description,
    isPrivate: true,
    collaborators,
  };
  if (Array.isArray(beforeInfo.licenses)) settings.licenses = beforeInfo.licenses;
  if (Array.isArray(beforeInfo.keywords)) settings.keywords = beforeInfo.keywords;

  const update = await call(env, "UpdateDatasetMetadata", {
    ownerSlug: OWNER,
    datasetSlug: DATASET_SLUG,
    settings,
  });
  const errors = Array.isArray(update.errors) ? update.errors.map(String).filter(Boolean) : [];
  if (errors.length) {
    throw new Error(`UpdateDatasetMetadata validation errors: ${errors.join(" | ").slice(0, 1000)}`);
  }

  const afterMetadata = await call(env, "GetDatasetMetadata", {
    ownerSlug: OWNER,
    datasetSlug: DATASET_SLUG,
  });
  const afterInfo = rec(afterMetadata.info);
  const afterCollaborators = normalizedCollaborators(afterInfo.collaborators);
  const verified = afterCollaborators.some(isAzadkaReader);
  if (!verified) {
    throw new Error(`azadka READER verification failed; collaborators=${JSON.stringify(afterCollaborators).slice(0, 1000)}`);
  }

  return {
    project: "PNEUMONIA V6.2.2",
    status: "PASS",
    dataset_ref: DATASET_REF,
    dataset_version: version,
    dataset_bytes: totalBytes,
    dataset_ready: true,
    reader: READER,
    role: "READER",
    role_value: 1,
    collaborator_verified: true,
    collaborators: afterCollaborators,
    model_compute: false,
    training: false,
    hpo: false,
    confirmation: false,
    locked_test: false,
    external_validation: false,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz") {
      return json({ service: "v622-dataset-permission", status: "ready", protected: true });
    }
    if (url.pathname !== "/control/v6-2-2/dataset-permission/grant" || request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try {
      return json(await grant(env));
    } catch (error) {
      return json({
        project: "PNEUMONIA V6.2.2",
        ok: false,
        error: error instanceof Error ? error.message.slice(0, 1400) : "unknown error",
        dataset_ref: DATASET_REF,
        model_compute: false,
      }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
