import {
  kernelLogs,
  kernelOutputFiles,
  kernelStatus,
  listKernels,
  type AccountId,
  type WorkerEnv,
} from "./kaggle";
import { v622ProjectPlan } from "./project-plan";

type RunKind = "train" | "hpo" | "confirm" | "finalization";

interface RecoveryTarget {
  id: string;
  kind: RunKind;
  accountId: AccountId;
  ownerSlug: string;
  kernelRef: string;
  modelCode?: string;
  resolution?: number;
}

const RECOVERY_TARGETS: readonly RecoveryTarget[] = [
  { id: "W01", kind: "train", accountId: "master", ownerSlug: "azadka", kernelRef: "azadka/pneumonia-v6-2-2-train-w01-m01-r224", modelCode: "M01", resolution: 224 },
  { id: "W02", kind: "train", accountId: "kg-02", ownerSlug: "radlinaradlina", kernelRef: "radlinaradlina/pneumonia-v6-2-2-train-w02-m01-r320", modelCode: "M01", resolution: 320 },
  { id: "W03", kind: "train", accountId: "kg-04", ownerSlug: "reyhanehazad", kernelRef: "reyhanehazad/pneumonia-v6-2-2-train-w03-m01-r384", modelCode: "M01", resolution: 384 },
  { id: "W04", kind: "train", accountId: "kg-05", ownerSlug: "trickermark", kernelRef: "trickermark/pneumonia-v6-2-2-train-w04-m02-r224", modelCode: "M02", resolution: 224 },
  { id: "W05", kind: "train", accountId: "kg-06", ownerSlug: "msdenis", kernelRef: "msdenis/pneumonia-v6-2-2-train-w05-m02-r320", modelCode: "M02", resolution: 320 },
  { id: "W06", kind: "train", accountId: "kg-07", ownerSlug: "nisabulutmark", kernelRef: "nisabulutmark/pneumonia-v6-2-2-train-w06-m02-r384", modelCode: "M02", resolution: 384 },
  { id: "HPO-S01", kind: "hpo", accountId: "kg-02", ownerSlug: "radlinaradlina", kernelRef: "radlinaradlina/pneumonia-v6-2-2-hpo-s01" },
  { id: "HPO-S02", kind: "hpo", accountId: "kg-04", ownerSlug: "reyhanehazad", kernelRef: "reyhanehazad/pneumonia-v6-2-2-hpo-s02" },
  { id: "HPO-S03", kind: "hpo", accountId: "kg-05", ownerSlug: "trickermark", kernelRef: "trickermark/pneumonia-v6-2-2-hpo-s03" },
  { id: "HPO-S04", kind: "hpo", accountId: "kg-06", ownerSlug: "msdenis", kernelRef: "msdenis/pneumonia-v6-2-2-hpo-s04" },
  { id: "HPO-S05", kind: "hpo", accountId: "kg-07", ownerSlug: "nisabulutmark", kernelRef: "nisabulutmark/pneumonia-v6-2-2-hpo-s05" },
  { id: "CONFIRM-C01", kind: "confirm", accountId: "kg-02", ownerSlug: "radlinaradlina", kernelRef: "radlinaradlina/pneumonia-v6-2-2-confirm-c01" },
  { id: "CONFIRM-C02", kind: "confirm", accountId: "kg-05", ownerSlug: "trickermark", kernelRef: "trickermark/pneumonia-v6-2-2-confirm-c02" },
  { id: "CONFIRM-C03", kind: "confirm", accountId: "kg-06", ownerSlug: "msdenis", kernelRef: "msdenis/pneumonia-v6-2-2-confirm-c03" },
] as const;

const ALL_ACCOUNTS: readonly AccountId[] = ["master", "kg-02", "kg-03", "kg-04", "kg-05", "kg-06", "kg-07"];
const ARCHITECTURE_TERMS = ["convnext", "efficientnet", "resnet", "densenet", "mobilenet", "swin", "maxvit", "coatnet", "regnet", "seresnet", "vision transformer", "vit"] as const;

function normalizeStatus(payload: Record<string, unknown>): string {
  const raw = payload.status;
  return typeof raw === "string" && raw.trim() ? raw.trim().toUpperCase() : "UNKNOWN";
}

function lifecycle(status: string): "complete" | "in_progress" | "needs_repair" | "unknown" {
  if (status === "COMPLETE") return "complete";
  if (["RUNNING", "QUEUED", "STARTING", "PENDING"].includes(status)) return "in_progress";
  if (["ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status)) return "needs_repair";
  return "unknown";
}

function modelHints(log: string, fallback?: string): string[] {
  const lower = log.toLocaleLowerCase("en-US");
  const hints = new Set<string>();
  if (fallback) hints.add(fallback);
  for (const term of ARCHITECTURE_TERMS) if (lower.includes(term)) hints.add(term);
  for (let index = 1; index <= 12; index += 1) {
    const code = `M${String(index).padStart(2, "0")}`;
    if (lower.includes(code.toLocaleLowerCase("en-US"))) hints.add(code);
  }
  return [...hints];
}

async function targetStatus(env: WorkerEnv, target: RecoveryTarget): Promise<Record<string, unknown>> {
  try {
    const payload = await kernelStatus(env, target.accountId, target.kernelRef);
    const status = normalizeStatus(payload);
    let hints: string[] = target.modelCode ? [target.modelCode] : [];
    if (target.kind === "train") {
      try {
        const log = await kernelLogs(env, target.accountId, target.kernelRef);
        hints = modelHints(log, target.modelCode);
      } catch {
        // Status remains authoritative even if the bounded log probe fails.
      }
    }
    return {
      id: target.id,
      kind: target.kind,
      account_id: target.accountId,
      owner_slug: target.ownerSlug,
      kernel_ref: target.kernelRef,
      model_code: target.modelCode ?? null,
      resolution: target.resolution ?? null,
      model_hints: hints,
      status,
      lifecycle: lifecycle(status),
    };
  } catch (error) {
    return {
      id: target.id,
      kind: target.kind,
      account_id: target.accountId,
      owner_slug: target.ownerSlug,
      kernel_ref: target.kernelRef,
      model_code: target.modelCode ?? null,
      resolution: target.resolution ?? null,
      model_hints: target.modelCode ? [target.modelCode] : [],
      status: "PROBE_ERROR",
      lifecycle: "needs_repair",
      error_type: error instanceof Error ? error.name : "Error",
      error: error instanceof Error ? error.message.slice(0, 500) : "unknown error",
    };
  }
}

function kernelField(value: unknown, field: "ref" | "title"): string {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  const raw = (value as Record<string, unknown>)[field];
  return typeof raw === "string" ? raw : "";
}

async function finalizationCandidates(env: WorkerEnv): Promise<Array<Record<string, unknown>>> {
  const found: Array<{ accountId: AccountId; ref: string; title: string }> = [];
  const seen = new Set<string>();
  const matcher = /(merge|champion|ensemble|selection)/i;
  const inventories = await Promise.all(
    ALL_ACCOUNTS.map(async (accountId) => {
      try {
        return { accountId, kernels: await listKernels(env, accountId, "pneumonia-v6-2-2", 100) };
      } catch {
        return { accountId, kernels: [] as unknown[] };
      }
    }),
  );
  for (const inventory of inventories) {
    for (const kernel of inventory.kernels) {
      const ref = kernelField(kernel, "ref");
      const title = kernelField(kernel, "title");
      if (!ref || !matcher.test(`${ref} ${title}`) || seen.has(ref)) continue;
      seen.add(ref);
      found.push({ accountId: inventory.accountId, ref, title });
      if (found.length >= 8) break;
    }
    if (found.length >= 8) break;
  }
  return Promise.all(
    found.map(async (candidate) => {
      try {
        const payload = await kernelStatus(env, candidate.accountId, candidate.ref);
        const status = normalizeStatus(payload);
        return { kind: "finalization", account_id: candidate.accountId, kernel_ref: candidate.ref, title: candidate.title, status, lifecycle: lifecycle(status) };
      } catch (error) {
        return { kind: "finalization", account_id: candidate.accountId, kernel_ref: candidate.ref, title: candidate.title, status: "PROBE_ERROR", lifecycle: "needs_repair", error: error instanceof Error ? error.message.slice(0, 500) : "unknown error" };
      }
    }),
  );
}

interface MatrixCandidate {
  accountId: AccountId;
  ownerSlug: string;
  kernelRef: string;
  workerId: string;
  workerNumber: number;
  modelId: string;
  resolution: number;
}

async function dynamicMatrixRuns(env: WorkerEnv): Promise<Array<Record<string, unknown>>> {
  const found = new Map<string, MatrixCandidate>();
  const inventories = await Promise.all(
    ALL_ACCOUNTS.map(async (accountId) => {
      try {
        return { accountId, kernels: await listKernels(env, accountId, "pneumonia-v6-2-2-train-w", 100) };
      } catch {
        return { accountId, kernels: [] as unknown[] };
      }
    }),
  );
  for (const inventory of inventories) {
    for (const kernel of inventory.kernels) {
      const ref = kernelField(kernel, "ref");
      const match = ref.match(/^([^/]+)\/pneumonia-v6-2-2-train-(w(\d+))-(m\d+)-r(\d+)$/i);
      if (!match) continue;
      found.set(ref.toLocaleLowerCase("en-US"), {
        accountId: inventory.accountId,
        ownerSlug: match[1],
        kernelRef: ref,
        workerId: match[2].toUpperCase(),
        workerNumber: Number(match[3]),
        modelId: match[4].toUpperCase(),
        resolution: Number(match[5]),
      });
    }
  }
  const candidates = [...found.values()].sort((a, b) => a.workerNumber - b.workerNumber);
  return Promise.all(
    candidates.map(async (candidate) => {
      try {
        const payload = await kernelStatus(env, candidate.accountId, candidate.kernelRef);
        const status = normalizeStatus(payload);
        return {
          worker_id: candidate.workerId,
          worker_number: candidate.workerNumber,
          account_id: candidate.accountId,
          owner_slug: candidate.ownerSlug,
          kernel_ref: candidate.kernelRef,
          model_id: candidate.modelId,
          resolution: candidate.resolution,
          status,
          lifecycle: lifecycle(status),
        };
      } catch (error) {
        return {
          worker_id: candidate.workerId,
          worker_number: candidate.workerNumber,
          account_id: candidate.accountId,
          owner_slug: candidate.ownerSlug,
          kernel_ref: candidate.kernelRef,
          model_id: candidate.modelId,
          resolution: candidate.resolution,
          status: "PROBE_ERROR",
          lifecycle: "needs_repair",
          error: error instanceof Error ? error.message.slice(0, 500) : "unknown error",
        };
      }
    }),
  );
}

function runString(run: Record<string, unknown>, key: string): string {
  const value = run[key];
  return typeof value === "string" ? value : "";
}

function latestEpoch(log: string): { current: number | null; total: number | null } {
  const matches = [...log.matchAll(/\bEpoch\s+(\d+)\s*\/\s*(\d+)/gi)];
  const last = matches[matches.length - 1];
  if (!last) return { current: null, total: null };
  return { current: Number(last[1]), total: Number(last[2]) };
}

function stageFromLog(log: string): "head" | "finetune" | "unknown" {
  const lower = log.toLocaleLowerCase("en-US");
  const head = Math.max(lower.lastIndexOf("head stage"), lower.lastIndexOf("head training"), lower.lastIndexOf("best_head"));
  const finetune = Math.max(lower.lastIndexOf("finetune"), lower.lastIndexOf("fine-tune"), lower.lastIndexOf("best_finetune"));
  if (head < 0 && finetune < 0) return "unknown";
  return finetune > head ? "finetune" : "head";
}

function errorMarker(log: string): string | null {
  const checks: Array<[string, RegExp]> = [
    ["traceback", /Traceback \(most recent call last\)/i],
    ["oom", /(?:out of memory|ResourceExhaustedError|CUDA.*memory)/i],
    ["killed", /(?:^|\n)Killed(?:\n|$)/i],
    ["disk", /No space left on device/i],
  ];
  for (const [label, pattern] of checks) if (pattern.test(log)) return label;
  return null;
}

export async function v622MatrixProgress(env: WorkerEnv): Promise<Record<string, unknown>> {
  const matrixRuns = await dynamicMatrixRuns(env);
  const active = matrixRuns.filter(
    (run) => run.lifecycle === "in_progress" || run.lifecycle === "needs_repair",
  );
  const jobs = await Promise.all(
    active.map(async (run) => {
      const accountId = runString(run, "account_id");
      const kernelRef = runString(run, "kernel_ref");
      let log = "";
      let logProbeOk = false;
      try {
        log = await kernelLogs(env, accountId, kernelRef);
        logProbeOk = true;
      } catch {
        // Status remains authoritative; progress probe intentionally exposes no raw error text.
      }
      const epoch = latestEpoch(log);
      return {
        worker_id: run.worker_id ?? null,
        account_id: accountId,
        model_id: run.model_id ?? null,
        resolution: run.resolution ?? null,
        status: run.status ?? null,
        lifecycle: run.lifecycle ?? null,
        log_probe_ok: logProbeOk,
        bounded_log_chars: log.length,
        stage: logProbeOk ? stageFromLog(log) : "unknown",
        last_epoch_current: epoch.current,
        last_epoch_total: epoch.total,
        pass_marker: /["']status["']\s*:\s*["']PASS["']/i.test(log),
        error_marker: errorMarker(log),
      };
    }),
  );
  return {
    project: "PNEUMONIA V6.2.2",
    generated_at: new Date().toISOString(),
    active_count: jobs.length,
    jobs,
  };
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function relevantArtifacts(fileNames: string[]): Record<string, string[]> {
  const usable = fileNames.filter((name) => !/SOURCE_PYTHON|__pycache__/i.test(name));
  const groups: Record<string, RegExp> = {
    checkpoints: /(?:checkpoint|weights|best_model|model_best|\.pt$|\.pth$|\.ckpt$|\.h5$|\.keras$|\.onnx$|\.safetensors$)/i,
    metrics_reports: /(?:metric|score|report|evaluation|eval_|result|matrix_execution_report)/i,
    predictions: /(?:prediction|preds?|probabilit|logits?|oof|submission)/i,
    recipes_manifests: /(?:recipe|manifest|fingerprint|config|threshold|calibrat)/i,
    ensemble_champion: /(?:ensemble|champion|selection|selected|handoff|merge)/i,
  };
  const output: Record<string, string[]> = {};
  for (const [key, pattern] of Object.entries(groups)) {
    output[key] = usable.filter((name) => pattern.test(name)).slice(0, 120);
  }
  return output;
}

export async function v622ShardArtifacts(env: WorkerEnv, shardId: string): Promise<Record<string, unknown>> {
  const normalized = shardId.toUpperCase();
  if (normalized === "PLAN") return v622ProjectPlan(env);
  if (normalized === "PROGRESS") return v622MatrixProgress(env);
  const target = RECOVERY_TARGETS.find((item) => item.kind === "train" && item.id === normalized);
  if (!target) throw new Error("unknown V6.2.2 shard; expected PLAN, PROGRESS or W01..W06");
  const listing = await kernelOutputFiles(env, target.accountId, target.kernelRef, "", 2000);
  const fileNames = stringArray(listing.file_names);
  const groups = relevantArtifacts(fileNames);
  const hints = new Set<string>();
  const joined = fileNames.join("\n").toLocaleLowerCase("en-US");
  for (const term of ARCHITECTURE_TERMS) if (joined.includes(term)) hints.add(term);
  if (target.modelCode) hints.add(target.modelCode);
  return {
    shard_id: target.id,
    account_id: target.accountId,
    owner_slug: target.ownerSlug,
    kernel_ref: target.kernelRef,
    model_code: target.modelCode ?? null,
    resolution: target.resolution ?? null,
    model_hints: [...hints],
    page_count: listing.page_count ?? null,
    enumerated_file_count: listing.enumerated_file_count ?? fileNames.length,
    truncated: listing.truncated ?? null,
    artifact_groups: groups,
  };
}

export async function v622RecoveryStatus(env: WorkerEnv): Promise<Record<string, unknown>> {
  const runs = await Promise.all(RECOVERY_TARGETS.map((target) => targetStatus(env, target)));
  const finalization = await finalizationCandidates(env);
  const matrixRuns = await dynamicMatrixRuns(env);
  const matrixComplete = matrixRuns.filter((run) => run.lifecycle === "complete");
  const matrixInProgress = matrixRuns.filter((run) => run.lifecycle === "in_progress");
  const matrixNeedsRepair = matrixRuns.filter((run) => run.lifecycle === "needs_repair");
  const matrixUnknown = matrixRuns.filter((run) => run.lifecycle === "unknown");
  const historical = [...runs, ...finalization];
  const complete = historical.filter((run) => run.lifecycle === "complete");
  const inProgress = historical.filter((run) => run.lifecycle === "in_progress");
  const needsRepair = historical.filter((run) => run.lifecycle === "needs_repair");
  return {
    project: "PNEUMONIA V6.2.2",
    generated_at: new Date().toISOString(),
    write_enabled: env.CGP_WRITE_ENABLED === "1",
    runs,
    canonical_matrix_runs: matrixRuns,
    canonical_matrix_summary: {
      discovered: matrixRuns.length,
      expected_total: 36,
      complete: matrixComplete.length,
      in_progress: matrixInProgress.length,
      needs_repair: matrixNeedsRepair.length,
      unknown: matrixUnknown.length,
      remaining_not_created: Math.max(0, 36 - matrixRuns.length),
    },
    finalization_candidates: finalization,
    summary: {
      complete: complete.length,
      in_progress: inProgress.length,
      needs_repair: needsRepair.length,
      unknown: historical.length - complete.length - inProgress.length - needsRepair.length,
    },
    in_progress: [...matrixInProgress, ...inProgress],
    needs_repair: [...matrixNeedsRepair, ...needsRepair],
  };
}
