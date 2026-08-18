import { kernelOutputFiles, kernelStatus, type AccountId, type WorkerEnv } from "./kaggle";
import { v622WavePlan } from "./matrix-run";

interface StatusTask {
  workerId: string;
  accountId: AccountId;
  ownerSlug: string;
  modelId: string;
  resolution: number;
  kernelRef: string;
}

const WAVE1: readonly StatusTask[] = [
  {
    workerId: "W01",
    accountId: "master",
    ownerSlug: "azadka",
    modelId: "M01",
    resolution: 224,
    kernelRef: "azadka/pneumonia-v6-2-2-train-w01-m01-r224",
  },
  {
    workerId: "W02",
    accountId: "kg-02",
    ownerSlug: "radlinaradlina",
    modelId: "M01",
    resolution: 320,
    kernelRef: "radlinaradlina/pneumonia-v6-2-2-train-w02-m01-r320",
  },
  {
    workerId: "W03",
    accountId: "kg-04",
    ownerSlug: "reyhanehazad",
    modelId: "M01",
    resolution: 384,
    kernelRef: "reyhanehazad/pneumonia-v6-2-2-train-w03-m01-r384",
  },
  {
    workerId: "W04",
    accountId: "kg-05",
    ownerSlug: "trickermark",
    modelId: "M02",
    resolution: 224,
    kernelRef: "trickermark/pneumonia-v6-2-2-train-w04-m02-r224",
  },
  {
    workerId: "W05",
    accountId: "kg-06",
    ownerSlug: "msdenis",
    modelId: "M02",
    resolution: 320,
    kernelRef: "msdenis/pneumonia-v6-2-2-train-w05-m02-r320",
  },
  {
    workerId: "W06",
    accountId: "kg-07",
    ownerSlug: "nisabulutmark",
    modelId: "M02",
    resolution: 384,
    kernelRef: "nisabulutmark/pneumonia-v6-2-2-train-w06-m02-r384",
  },
] as const;

const IN_PROGRESS_STATUSES = new Set(["RUNNING", "QUEUED", "STARTING", "PENDING"]);
const TERMINAL_ARTIFACTS = [
  "training/matrix_execution_report.json",
  "/final_report.json",
  "/validation_metrics.json",
  "/val_predictions.csv",
] as const;

function planForWave(wave: number): StatusTask[] {
  if (!Number.isInteger(wave) || wave < 1 || wave > 6) {
    throw new Error("wave must be an integer from 1 through 6");
  }
  if (wave === 1) return [...WAVE1];
  return v622WavePlan(wave) as unknown as StatusTask[];
}

function normalizeStatus(value: Record<string, unknown>): string {
  const status = value.status;
  return typeof status === "string" && status.trim() ? status.trim().toUpperCase() : "UNKNOWN";
}

function lifecycle(status: string): "complete" | "in_progress" | "needs_repair" | "missing" | "unknown" {
  if (status === "COMPLETE") return "complete";
  if (IN_PROGRESS_STATUSES.has(status)) return "in_progress";
  if (["ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status)) return "needs_repair";
  if (status === "NOT_FOUND") return "missing";
  return "unknown";
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function terminalArtifactMatches(fileNames: string[]): string[] {
  const normalized = fileNames.map((name) => name.replaceAll("\\", "/"));
  return TERMINAL_ARTIFACTS.filter((needle) =>
    normalized.some((name) => needle.startsWith("/") ? name.endsWith(needle) : name.endsWith(needle)),
  );
}

async function corroborateCompletion(
  env: WorkerEnv,
  task: StatusTask,
): Promise<{ complete: boolean; matched: string[]; error: string | null }> {
  try {
    // Only bounded filename metadata is read. No artifact bytes are downloaded.
    const listing = await kernelOutputFiles(env, task.accountId, task.kernelRef, "", 300);
    const matched = terminalArtifactMatches(stringArray(listing.file_names));
    return {
      complete: matched.length === TERMINAL_ARTIFACTS.length,
      matched,
      error: null,
    };
  } catch (error) {
    return {
      complete: false,
      matched: [],
      error: error instanceof Error ? error.message.slice(0, 500) : "unknown error",
    };
  }
}

export async function v622WaveStatus(env: WorkerEnv, wave: number): Promise<Record<string, unknown>> {
  const plan = planForWave(wave);
  const tasks = await Promise.all(
    plan.map(async (task) => {
      try {
        const response = await kernelStatus(env, task.accountId, task.kernelRef);
        const rawStatus = normalizeStatus(response);
        let status = rawStatus;
        let completionEvidence: Record<string, unknown> | null = null;
        let completionProbeError: string | null = null;

        // Kaggle's session-status endpoint can occasionally lag behind the UI.
        // For an in-progress state only, corroborate completion using the exact
        // canonical terminal output set. Failure states are never overridden.
        if (IN_PROGRESS_STATUSES.has(rawStatus)) {
          const corroboration = await corroborateCompletion(env, task);
          if (corroboration.complete) {
            status = "COMPLETE";
            completionEvidence = {
              source: "terminal_artifacts",
              raw_status: rawStatus,
              matched: corroboration.matched,
            };
          } else if (corroboration.error) {
            completionProbeError = corroboration.error;
          }
        }

        return {
          worker_id: task.workerId,
          account_id: task.accountId,
          owner_slug: task.ownerSlug,
          model_id: task.modelId,
          resolution: task.resolution,
          kernel_ref: task.kernelRef,
          status,
          lifecycle: lifecycle(status),
          raw_status: rawStatus,
          completion_evidence: completionEvidence,
          completion_probe_error: completionProbeError,
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : "unknown error";
        const status = /not found|404|does not exist/i.test(message) ? "NOT_FOUND" : "PROBE_ERROR";
        return {
          worker_id: task.workerId,
          account_id: task.accountId,
          owner_slug: task.ownerSlug,
          model_id: task.modelId,
          resolution: task.resolution,
          kernel_ref: task.kernelRef,
          status,
          lifecycle: lifecycle(status),
          raw_status: status,
          probe_error: status === "PROBE_ERROR" ? message.slice(0, 500) : null,
        };
      }
    }),
  );

  const counts = {
    complete: tasks.filter((item) => item.lifecycle === "complete").length,
    in_progress: tasks.filter((item) => item.lifecycle === "in_progress").length,
    needs_repair: tasks.filter((item) => item.lifecycle === "needs_repair").length,
    missing: tasks.filter((item) => item.lifecycle === "missing").length,
    unknown: tasks.filter((item) => item.lifecycle === "unknown").length,
  };
  return { project: "PNEUMONIA V6.2.2", wave, generated_at: new Date().toISOString(), counts, tasks };
}
