import {
  getKernel,
  kernelStatus,
  listKernels,
  type WorkerAccountId,
  type WorkerEnv,
} from "./kaggle";

type MatrixEnv = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

type Resolution = 224 | 320 | 384;

interface MatrixTask {
  wave: number;
  workerId: string;
  accountId: WorkerAccountId;
  ownerSlug: string;
  modelId: string;
  resolution: Resolution;
  slug: string;
  title: string;
  kernelRef: string;
}

const TEMPLATE_REF = "azadka/pneumonia-v6-2-2-train-w01-m01-r224";
const KAGGLE_API_ROOT = "https://api.kaggle.com/v1";
const KAGGLE_SERVICE = "kernels.KernelsApiService";
const EXPECTED_SOURCE_FINGERPRINT = "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838";
const EXPECTED_RECIPE_SHA256 = "f5ebe3321e75d7edaf1dfa60b53835bbb23008592001bc833405e10c8890116c";
const RESOLUTIONS: readonly Resolution[] = [224, 320, 384];
const ACCOUNTS: readonly { accountId: WorkerAccountId; ownerSlug: string; username: string }[] = [
  { accountId: "kg-02", ownerSlug: "radlinaradlina", username: "radlinaradlina" },
  { accountId: "kg-03", ownerSlug: "rezanory", username: "rezanory" },
  { accountId: "kg-04", ownerSlug: "reyhanehazad", username: "reyhanehazad" },
  { accountId: "kg-05", ownerSlug: "trickermark", username: "trickermark" },
  { accountId: "kg-06", ownerSlug: "msdenis", username: "msdenis" },
  { accountId: "kg-07", ownerSlug: "nisabulutmark", username: "nisabulutmark" },
] as const;

function tokenFor(env: MatrixEnv, accountId: WorkerAccountId): string {
  const value =
    accountId === "kg-02" ? env.CGP_KAGGLE_KG02_TOKEN :
    accountId === "kg-03" ? env.CGP_KAGGLE_KG03_TOKEN :
    accountId === "kg-04" ? env.CGP_KAGGLE_KG04_TOKEN :
    accountId === "kg-05" ? env.CGP_KAGGLE_KG05_TOKEN :
    accountId === "kg-06" ? env.CGP_KAGGLE_KG06_TOKEN :
    env.CGP_KAGGLE_KG07_TOKEN;
  const token = value?.trim();
  if (!token) throw new Error(`Kaggle token missing for ${accountId}`);
  return token;
}

function authorization(token: string, username: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${username}:${token}`)}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function copyIfPresent(target: Record<string, unknown>, source: Record<string, unknown>, key: string): void {
  const value = source[key];
  if (value !== undefined && value !== null && value !== "") target[key] = value;
}

function taskFor(wave: number, slot: number): MatrixTask {
  if (!Number.isInteger(wave) || wave < 2 || wave > 6) throw new Error("wave must be an integer from 2 through 6");
  if (!Number.isInteger(slot) || slot < 0 || slot >= 6) throw new Error("slot must be 0..5");
  const firstModelNumber = 1 + (wave - 1) * 2;
  const modelNumber = firstModelNumber + (slot >= 3 ? 1 : 0);
  const modelId = `M${String(modelNumber).padStart(2, "0")}`;
  const resolution = RESOLUTIONS[slot % 3];
  const workerNumber = 1 + (wave - 1) * 6 + slot;
  const workerId = `W${String(workerNumber).padStart(2, "0")}`;
  const account = ACCOUNTS[slot];
  const slug = `pneumonia-v6-2-2-train-${workerId.toLowerCase()}-${modelId.toLowerCase()}-r${resolution}`;
  const title = `PNEUMONIA V6.2.2 TRAIN ${workerId} ${modelId} R${resolution}`;
  return {
    wave,
    workerId,
    accountId: account.accountId,
    ownerSlug: account.ownerSlug,
    modelId,
    resolution,
    slug,
    title,
    kernelRef: `${account.ownerSlug}/${slug}`,
  };
}

export function v622WavePlan(wave: number): MatrixTask[] {
  return Array.from({ length: 6 }, (_, slot) => taskFor(wave, slot));
}

function replaceExactly(text: string, oldValue: string, newValue: string, expected: number, label: string): string {
  const count = text.split(oldValue).length - 1;
  if (count !== expected) throw new Error(`template contract mismatch for ${label}: expected ${expected}, found ${count}`);
  return text.split(oldValue).join(newValue);
}

function transformNotebookSource(source: string, task: MatrixTask): string {
  let root: unknown;
  try {
    root = JSON.parse(source);
  } catch {
    throw new Error("canonical template is no longer a JSON notebook");
  }
  const notebook = asRecord(root);
  const cells = Array.isArray(notebook.cells) ? notebook.cells : [];
  if (cells.length !== 1) throw new Error(`canonical template cell-count changed: ${cells.length}`);
  const cell = asRecord(cells[0]);
  const rawSource = cell.source;
  const parts = typeof rawSource === "string"
    ? [rawSource]
    : Array.isArray(rawSource)
      ? rawSource.map((value) => typeof value === "string" ? value : "")
      : [];
  if (!parts.length) throw new Error("canonical template code cell has no source");
  let code = parts.join("");
  if (!code.includes(EXPECTED_SOURCE_FINGERPRINT)) throw new Error("canonical source fingerprint is not embedded in template");

  code = replaceExactly(
    code,
    "PNEUMONIA_V62_2_TRAIN_W01",
    `PNEUMONIA_V62_2_TRAIN_${task.workerId}`,
    1,
    "work-root",
  );
  code = replaceExactly(code, "'--models','M01'", `'--models','${task.modelId}'`, 1, "model-arg");
  code = replaceExactly(code, "'--image-size','224'", `'--image-size','${task.resolution}'`, 1, "resolution-arg");
  code = replaceExactly(code, "'--run-name','W01_M01_r224'", `'--run-name','${task.workerId}_${task.modelId}_r${task.resolution}'`, 1, "run-name");
  code = replaceExactly(code, "'worker_id':'W01'", `'worker_id':'${task.workerId}'`, 1, "worker-id");
  code = replaceExactly(code, "'model':'M01'", `'model':'${task.modelId}'`, 1, "completion-model");
  code = replaceExactly(code, "'resolution':224", `'resolution':${task.resolution}`, 1, "completion-resolution");

  const updatedCell = { ...cell, source: [code] };
  notebook.cells = [updatedCell];
  return JSON.stringify(notebook);
}

function existingRef(kernel: unknown): string {
  const rec = asRecord(kernel);
  return typeof rec.ref === "string" ? rec.ref : "";
}

async function saveNewKernel(
  env: MatrixEnv,
  task: MatrixTask,
  source: string,
  template: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const account = ACCOUNTS.find((item) => item.accountId === task.accountId);
  if (!account) throw new Error(`unknown target account ${task.accountId}`);
  const token = tokenFor(env, task.accountId);
  const metadata = asRecord(template.metadata);
  const blob = asRecord(template.blob);
  const request: Record<string, unknown> = {
    slug: task.kernelRef,
    newTitle: task.title,
    text: source,
    kernelExecutionType: "SAVE_AND_RUN_ALL",
  };
  copyIfPresent(request, blob, "language");
  if (request.language === undefined) copyIfPresent(request, metadata, "language");
  copyIfPresent(request, blob, "kernelType");
  if (request.kernelType === undefined) copyIfPresent(request, metadata, "kernelType");
  for (const key of [
    "datasetDataSources",
    "kernelDataSources",
    "competitionDataSources",
    "categoryIds",
    "isPrivate",
    "enableGpu",
    "enableInternet",
    "dockerImagePinningType",
    "modelDataSources",
    "enableTpu",
    "sessionTimeoutSeconds",
    "priority",
    "dockerImage",
    "machineShape",
  ]) copyIfPresent(request, metadata, key);

  const response = await fetch(`${KAGGLE_API_ROOT}/${KAGGLE_SERVICE}/SaveKernel`, {
    method: "POST",
    headers: {
      Authorization: authorization(token, account.username),
      "Content-Type": "application/json",
      "User-Agent": "chatgpt-kaggle-v622-matrix/0.1",
    },
    body: JSON.stringify(request),
  });
  const text = await response.text();
  let payload: unknown;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`SaveKernel returned non-JSON HTTP ${response.status}`);
  }
  const record = asRecord(payload);
  const code = typeof record.code === "number" ? record.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) {
    const message = typeof record.message === "string" ? record.message : `SaveKernel HTTP ${response.status}`;
    throw new Error(message.slice(0, 1000));
  }
  return record;
}

async function launchTask(env: MatrixEnv, task: MatrixTask, template: Record<string, unknown>): Promise<Record<string, unknown>> {
  const existing = await listKernels(env, task.accountId, task.slug, 100);
  if (existing.some((kernel) => existingRef(kernel).toLocaleLowerCase("en-US") === task.kernelRef.toLocaleLowerCase("en-US"))) {
    let status: Record<string, unknown> = {};
    try { status = await kernelStatus(env, task.accountId, task.kernelRef); } catch { /* report existing even if status probe fails */ }
    return { ...task, action: "existing", status };
  }
  const blob = asRecord(template.blob);
  const templateSource = typeof blob.source === "string" ? blob.source : "";
  if (!templateSource) throw new Error("canonical template GetKernel returned no source");
  const source = transformNotebookSource(templateSource, task);
  const result = await saveNewKernel(env, task, source, template);
  return { ...task, action: "submitted", result };
}

export function projectControlAuthorized(request: Request, env: MatrixEnv): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  if (!expected || expected.length < 32) return false;
  const header = request.headers.get("authorization") ?? "";
  return header === `Bearer ${expected}`;
}

export async function launchV622Wave(env: MatrixEnv, wave: number): Promise<Record<string, unknown>> {
  const plan = v622WavePlan(wave);
  const template = await getKernel(env, "master", TEMPLATE_REF);
  const results = await Promise.all(plan.map((task) => launchTask(env, task, template)));
  return {
    project: "PNEUMONIA V6.2.2",
    wave,
    source_template: TEMPLATE_REF,
    source_fingerprint: EXPECTED_SOURCE_FINGERPRINT,
    frozen_recipe_sha256: EXPECTED_RECIPE_SHA256,
    seed: 42,
    head_epochs: 3,
    finetune_epochs: 47,
    results,
  };
}
