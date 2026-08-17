import {
  kernelLogs,
  kernelOutputFiles,
  kernelStatus,
  listKernels,
  type AccountId,
  type WorkerEnv,
} from "./kaggle";

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
  for (const code of ["M01", "M02", "M03", "M04", "M05", "M06"]) {
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
  const target = RECOVERY_TARGETS.find((item) => item.kind === "train" && item.id === shardId.toUpperCase());
  if (!target) throw new Error("unknown V6.2.2 shard; expected W01..W06");
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
  const all = [...runs, ...finalization];
  const complete = all.filter((run) => run.lifecycle === "complete");
  const inProgress = all.filter((run) => run.lifecycle === "in_progress");
  const needsRepair = all.filter((run) => run.lifecycle === "needs_repair");
  return {
    project: "PNEUMONIA V6.2.2",
    generated_at: new Date().toISOString(),
    write_enabled: env.CGP_WRITE_ENABLED === "1",
    runs,
    finalization_candidates: finalization,
    summary: {
      complete: complete.length,
      in_progress: inProgress.length,
      needs_repair: needsRepair.length,
      unknown: all.length - complete.length - inProgress.length - needsRepair.length,
    },
    in_progress: inProgress,
    needs_repair: needsRepair,
  };
}
