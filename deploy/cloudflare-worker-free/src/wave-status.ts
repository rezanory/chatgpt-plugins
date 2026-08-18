import { kernelStatus, type WorkerAccountId, type WorkerEnv } from "./kaggle";
import { v622WavePlan } from "./matrix-run";

interface StatusTask {
  workerId: string;
  accountId: WorkerAccountId;
  ownerSlug: string;
  modelId: string;
  resolution: number;
  kernelRef: string;
}

function normalizeStatus(value: Record<string, unknown>): string {
  const status = value.status;
  return typeof status === "string" && status.trim() ? status.trim().toUpperCase() : "UNKNOWN";
}

function lifecycle(status: string): "complete" | "in_progress" | "needs_repair" | "missing" | "unknown" {
  if (status === "COMPLETE") return "complete";
  if (["RUNNING", "QUEUED", "STARTING", "PENDING"].includes(status)) return "in_progress";
  if (["ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status)) return "needs_repair";
  if (status === "NOT_FOUND") return "missing";
  return "unknown";
}

export async function v622WaveStatus(env: WorkerEnv, wave: number): Promise<Record<string, unknown>> {
  const plan = v622WavePlan(wave) as unknown as StatusTask[];
  const tasks = await Promise.all(
    plan.map(async (task) => {
      try {
        const response = await kernelStatus(env, task.accountId, task.kernelRef);
        const status = normalizeStatus(response);
        return {
          worker_id: task.workerId,
          account_id: task.accountId,
          owner_slug: task.ownerSlug,
          model_id: task.modelId,
          resolution: task.resolution,
          kernel_ref: task.kernelRef,
          status,
          lifecycle: lifecycle(status),
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
          error: message.slice(0, 500),
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
