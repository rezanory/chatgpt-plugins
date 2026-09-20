import canonicalWorker from './index';
import type { WorkerEnv } from './kaggle';
import {
  P0_EXPIRES_AT_MS,
  P0_GITHUB_RUN_ID,
  P0_SCRIPT,
  P0_SCRIPT_SHA256,
  W16_NOTEBOOK_SHA256,
  W16_SOURCE_ZIP_SHA256,
} from './v623-p0-payload';

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };
type Rec = Record<string, unknown>;
type AccountId = 'kg-02' | 'kg-03' | 'kg-04' | 'kg-05' | 'kg-06' | 'kg-07' | 'master';
interface Account { accountId: AccountId; owner: string; }

const API_ROOT = 'https://api.kaggle.com/v1/kernels.KernelsApiService';
const TARGET_SLUG = 'pneumonia-v6-2-3-p0-sandbox';
const TARGET_TITLE = 'PNEUMONIA V6.2.3 P0 SINGLE MODEL SANDBOX';
const RAW_DATASET = 'paultimothymooney/chest-xray-pneumonia';
const ACTIVE = new Set(['RUNNING', 'QUEUED', 'STARTING', 'PENDING']);
const TERMINAL_BAD = new Set(['ERROR', 'FAILED', 'CANCELLED', 'CANCELED', 'DEAD']);
const ACCOUNTS: readonly Account[] = [
  { accountId: 'kg-02', owner: 'radlinaradlina' },
  { accountId: 'kg-03', owner: 'rezanory' },
  { accountId: 'kg-04', owner: 'reyhanehazad' },
  { accountId: 'kg-05', owner: 'trickermark' },
  { accountId: 'kg-06', owner: 'msdenis' },
  { accountId: 'kg-07', owner: 'nisabulutmark' },
  { accountId: 'master', owner: 'azadka' },
] as const;

function rec(value: unknown): Rec {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Rec : {};
}
function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' } });
}
function tokenFor(env: Env, accountId: AccountId): string | undefined {
  switch (accountId) {
    case 'kg-02': return env.CGP_KAGGLE_KG02_TOKEN;
    case 'kg-03': return env.CGP_KAGGLE_KG03_TOKEN;
    case 'kg-04': return env.CGP_KAGGLE_KG04_TOKEN;
    case 'kg-05': return env.CGP_KAGGLE_KG05_TOKEN;
    case 'kg-06': return env.CGP_KAGGLE_KG06_TOKEN;
    case 'kg-07': return env.CGP_KAGGLE_KG07_TOKEN;
    case 'master': return env.CGP_KAGGLE_MASTER_TOKEN;
  }
}
function authHeader(account: Account, token: string): string {
  return token.startsWith('KGAT_') ? `Bearer ${token}` : `Basic ${btoa(`${account.owner}:${token}`)}`;
}
function normalizedStatus(value: Rec): string {
  const session = rec(value.session);
  for (const candidate of [value.status, value.statusName, value.status_name, value.state, value.sessionStatus, value.kernelSessionStatus, session.status, session.statusName, session.state]) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim().toUpperCase();
  }
  return 'UNKNOWN';
}
function seconds(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const r = value as Rec;
    const s = typeof r.seconds === 'number' ? r.seconds : Number(r.seconds ?? 0);
    const n = typeof r.nanos === 'number' ? r.nanos : Number(r.nanos ?? 0);
    if (Number.isFinite(s) && Number.isFinite(n)) return s + n / 1e9;
  }
  if (typeof value === 'string') {
    const match = value.trim().match(/^(-?[0-9]+(?:\.[0-9]+)?)s$/);
    if (match) return Number(match[1]);
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return null;
}
async function kaggleCall(env: Env, account: Account, method: string, body: Rec): Promise<Rec> {
  const token = tokenFor(env, account.accountId)?.trim();
  if (!token) throw new Error(`Kaggle credential missing for ${account.accountId}`);
  const response = await fetch(`${API_ROOT}/${method}`, {
    method: 'POST',
    headers: { Authorization: authHeader(account, token), 'Content-Type': 'application/json', 'User-Agent': 'chatgpt-v623-p0-bridge/1.0' },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let value: Rec = {};
  try { value = rec(text ? JSON.parse(text) : {}); } catch { throw new Error(`${method} non-JSON HTTP ${response.status}`); }
  const code = typeof value.code === 'number' ? value.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) throw new Error(String(value.message ?? `${method} HTTP ${response.status}`).slice(0, 1200));
  return value;
}
function accountById(id: string): Account {
  const account = ACCOUNTS.find((item) => item.accountId === id);
  if (!account) throw new Error(`unknown P0 account_id: ${id}`);
  return account;
}
function targetRef(account: Account): string { return `${account.owner}/${TARGET_SLUG}`; }
function normalizeKernelRef(value: string, account: Account, fallback: string): string {
  const raw = value.trim().replace(/^\/code\//i, '');
  if (raw.includes('/')) return raw;
  if (/^[A-Za-z0-9._-]+$/.test(raw)) return `${account.owner}/${raw}`;
  return fallback;
}
function kernelRefFrom(value: Rec, account: Account, fallback: string): string {
  const nested = [rec(value.kernel), rec(value.kernelInfo), rec(value.kernel_info), rec(value.result)];
  const candidates = [value.ref, value.kernelRef, value.kernel_ref, value.kernelSlug, value.kernel_slug, ...nested.flatMap((x) => [x.ref, x.kernelRef, x.kernel_ref, x.slug])];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || !candidate.trim()) continue;
    return normalizeKernelRef(candidate, account, fallback);
  }
  return fallback;
}
function kernelSlug(ref: string): string {
  const parts = ref.trim().split('/');
  return parts.length === 2 ? parts[1] : ref.trim();
}
function authorized(request: Request, env: Env): boolean {
  const token = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  const run = request.headers.get('x-cgp-github-run') ?? '';
  return Boolean(
    token && token.length >= 32 &&
    P0_GITHUB_RUN_ID && run === P0_GITHUB_RUN_ID &&
    request.headers.get('authorization') === `Bearer ${token}` &&
    Date.now() <= P0_EXPIRES_AT_MS
  );
}
async function listExact(env: Env, account: Account): Promise<Rec[]> {
  if (!tokenFor(env, account.accountId)?.trim()) return [];
  const response = await kaggleCall(env, account, 'ListKernels', { group: 'PROFILE', sortBy: 'DATE_RUN', pageSize: 100, search: 'pneumonia-v6-2-3-p0' });
  const kernels = Array.isArray(response.kernels) ? response.kernels.map(rec) : [];
  const expected = targetRef(account).toLowerCase();
  const ownerPrefix = `${account.owner.toLowerCase()}/pneumonia-v6-2-3-p0-`;
  return kernels.filter((row) => {
    const ref = String(row.ref ?? '').toLowerCase();
    const title = String(row.title ?? row.name ?? '').toLowerCase();
    return ref === expected || (ref.startsWith(ownerPrefix) && title === TARGET_TITLE.toLowerCase());
  });
}
async function status(env: Env, account: Account, ref: string = targetRef(account)): Promise<string> {
  const raw = await kaggleCall(env, account, 'GetKernelSessionStatus', { userName: account.owner, kernelSlug: kernelSlug(ref) });
  return normalizedStatus(raw);
}
async function gpuRemainingHours(env: Env, account: Account): Promise<number | null> {
  const value = await kaggleCall(env, account, 'GetAcceleratorQuotaStatistics', {});
  const q = rec(value.gpuQuota ?? value.gpu_quota);
  const used = seconds(q.timeUsed ?? q.time_used);
  const total = seconds(q.totalTimeAllowed ?? q.total_time_allowed);
  if (used === null || total === null) return null;
  return Math.max(0, (total - used) / 3600);
}
async function allowedOutputFetch(rawUrl: string): Promise<Response> {
  const validate = (raw: string) => {
    const u = new URL(raw);
    const h = u.hostname.toLowerCase();
    if (u.protocol !== 'https:' || !(h === 'api.kaggle.com' || h === 'www.kaggle.com' || h === 'storage.googleapis.com' || h.endsWith('.kaggleusercontent.com') || h.endsWith('.googleusercontent.com'))) {
      throw new Error(`output URL host not allowlisted: ${h}`);
    }
  };
  validate(rawUrl);
  const response = await fetch(rawUrl, { redirect: 'follow' });
  validate(response.url || rawUrl);
  if (!response.ok) throw new Error(`output download HTTP ${response.status}`);
  return response;
}
async function outputs(env: Env, account: Account, ref: string = targetRef(account)): Promise<Rec[]> {
  const value = await kaggleCall(env, account, 'ListKernelSessionOutput', { userName: account.owner, kernelSlug: kernelSlug(ref), pageSize: 100 });
  return Array.isArray(value.files) ? value.files.map(rec) : [];
}
async function validatedReceipt(env: Env, account: Account, ref: string = targetRef(account)): Promise<Rec | null> {
  const files = await outputs(env, account, ref);
  const required = [
    'P0_RECEIPT.json', 'P0_SPLIT_RECEIPT.json',
    'P0_SEED_42_RECEIPT.json', 'P0_SEED_2026_RECEIPT.json',
    'P0_SHADOW_PREDICTIONS_seed42.csv', 'P0_SHADOW_PREDICTIONS_seed2026.csv',
  ];
  for (const suffix of required) if (!files.some((x) => String(x.fileName ?? '').endsWith(suffix))) return null;
  const row = files.find((x) => String(x.fileName ?? '').endsWith('P0_RECEIPT.json') && typeof x.url === 'string');
  if (!row || typeof row.url !== 'string') return null;
  const response = await allowedOutputFetch(row.url);
  const value = rec(JSON.parse(await response.text()));
  const counts = rec(value.sample_counts);
  const recipe = rec(value.training_recipe);
  const governance = rec(value.governance);
  if (
    value.project !== 'PNEUMONIA V6.2.3-P0' ||
    value.stage !== 'P0_SINGLE_MODEL_DIRECTIONAL_SANDBOX' ||
    value.status !== 'PASS' || value.official_result !== false || value.development_only !== true ||
    value.test_partition_enumerated !== false || value.locked_test_used !== false || value.external_data_used !== false ||
    value.chexpert_used !== false || value.nih_used !== false ||
    value.model_id !== 'M06' || value.backbone !== 'convnext_tiny' || value.image_size !== 224 ||
    counts.train !== 480 || counts.cal !== 120 || counts.shadow !== 120 ||
    value.canonical_source_zip_sha256 !== W16_SOURCE_ZIP_SHA256 ||
    recipe.validation_data_used_in_fit !== false ||
    governance.old_locked_624_reopened !== false || governance.old_external_cohorts_reused !== false || governance.no_shadow_feedback_within_this_run !== true
  ) throw new Error('P0 receipt leakage/identity/integrity mismatch');
  return value;
}
async function findExisting(env: Env): Promise<{ account: Account; status: string; receipt?: Rec; kernelRef?: string } | null> {
  const found: Array<{ account: Account; status: string; receipt?: Rec; kernelRef?: string }> = [];
  for (const account of ACCOUNTS) {
    if (!tokenFor(env, account.accountId)?.trim()) continue;
    const exact = await listExact(env, account);
    if (exact.length > 1) throw new Error(`multiple exact P0 kernels found for ${account.accountId}`);
    if (exact.length === 1) {
      const kernelRef = normalizeKernelRef(String(exact[0].ref ?? ''), account, targetRef(account));
      const s = await status(env, account, kernelRef);
      const item: { account: Account; status: string; receipt?: Rec; kernelRef?: string } = { account, status: s, kernelRef };
      if (s === 'COMPLETE') {
        const r = await validatedReceipt(env, account, kernelRef);
        if (r) item.receipt = r;
      }
      found.push(item);
    }
  }
  if (found.length > 1) throw new Error(`duplicate P0 kernels already exist across accounts: ${found.map((x) => targetRef(x.account)).join(', ')}`);
  return found[0] ?? null;
}
async function chooseAccount(env: Env): Promise<{ account: Account; remaining: number | null; quota: Rec[] }> {
  const quota: Rec[] = [];
  const candidates: Array<{ account: Account; remaining: number | null }> = [];
  for (const account of ACCOUNTS) {
    if (!tokenFor(env, account.accountId)?.trim()) continue;
    try {
      const remaining = await gpuRemainingHours(env, account);
      quota.push({ account_id: account.accountId, owner: account.owner, status: 'OK', gpu_remaining_hours: remaining });
      candidates.push({ account, remaining });
    } catch (error) {
      quota.push({ account_id: account.accountId, owner: account.owner, status: 'QUOTA_ERROR', error: error instanceof Error ? error.message.slice(0, 500) : 'unknown error' });
    }
  }
  if (!candidates.length) throw new Error('no Kaggle account with usable credential/quota probe');
  candidates.sort((a, b) => (b.remaining ?? -1) - (a.remaining ?? -1));
  return { ...candidates[0], quota };
}
async function saveKernel(env: Env, account: Account): Promise<Rec> {
  const script = String(P0_SCRIPT);
  const scriptSha256 = String(P0_SCRIPT_SHA256);
  if (!script || !scriptSha256 || script.length < 1000) throw new Error('generated P0 payload is not ready');
  const value = await kaggleCall(env, account, 'SaveKernel', {
    slug: targetRef(account), newTitle: TARGET_TITLE, text: script,
    language: 'python', kernelType: 'script', kernelExecutionType: 'SAVE_AND_RUN_ALL',
    isPrivate: true, enableGpu: true, enableTpu: false, enableInternet: true,
    kernelDataSources: [], datasetDataSources: [RAW_DATASET], competitionDataSources: [], modelDataSources: [],
  });
  const badD = Array.isArray(value.invalidDatasetSources) ? value.invalidDatasetSources : [];
  const badK = Array.isArray(value.invalidKernelSources) ? value.invalidKernelSources : [];
  if (badD.length || badK.length) throw new Error(`SaveKernel rejected sources: datasets=${JSON.stringify(badD)} kernels=${JSON.stringify(badK)}`);
  return value;
}
async function launch(env: Env): Promise<Rec> {
  const existing = await findExisting(env);
  if (existing) {
    if (ACTIVE.has(existing.status)) return { action: 'existing_active', submitted: false, selected_account_id: existing.account.accountId, selected_owner: existing.account.owner, target_kernel_ref: existing.kernelRef ?? targetRef(existing.account), status: existing.status };
    if (existing.status === 'COMPLETE' && existing.receipt) return { action: 'existing_complete', submitted: false, selected_account_id: existing.account.accountId, selected_owner: existing.account.owner, target_kernel_ref: existing.kernelRef ?? targetRef(existing.account), status: existing.status, receipt: existing.receipt };
    if (existing.status === 'COMPLETE') throw new Error('existing COMPLETE P0 lacks validated receipt; duplicate launch forbidden');
    if (TERMINAL_BAD.has(existing.status)) throw new Error(`existing P0 terminal ${existing.status}; automatic rerun forbidden`);
    throw new Error(`existing P0 has ambiguous status ${existing.status}; automatic rerun forbidden`);
  }
  const selected = await chooseAccount(env);
  const result = await saveKernel(env, selected.account);
  const submittedRef = kernelRefFrom(result, selected.account, targetRef(selected.account));
  return {
    action: 'created_and_submitted', submitted: true,
    selected_account_id: selected.account.accountId, selected_owner: selected.account.owner,
    gpu_remaining_hours_before_launch: selected.remaining, quota_observation: selected.quota,
    target_kernel_ref: submittedRef, result,
    p0_script_sha256: P0_SCRIPT_SHA256, w16_notebook_sha256: W16_NOTEBOOK_SHA256, w16_source_zip_sha256: W16_SOURCE_ZIP_SHA256,
    locked_test_used: false, external_data_used: false, training: true, hpo: false, ensemble: false,
  };
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/healthz' && request.method === 'GET') {
      return json({ service: 'v623-p0-bridge', status: 'ready', protected: true, payload_ready: Boolean(P0_SCRIPT && P0_SCRIPT_SHA256), github_run_id: P0_GITHUB_RUN_ID, expires_at_ms: P0_EXPIRES_AT_MS, target_slug: TARGET_SLUG, development_only: true, locked_test: false, external: false });
    }
    if (url.pathname.startsWith('/control/v6-2-3/p0/') && !authorized(request, env)) return new Response('Forbidden', { status: 403 });
    try {
      if (url.pathname === '/control/v6-2-3/p0/launch' && request.method === 'POST') {
        return json({ project: 'PNEUMONIA V6.2.3-P0', ...(await launch(env)) });
      }
      if (url.pathname === '/control/v6-2-3/p0/status' && request.method === 'POST') {
        const body = rec(await request.json());
        const account = accountById(String(body.account_id ?? ''));
        const requestedRef = normalizeKernelRef(String(body.kernel_ref ?? ''), account, '');
        if (!requestedRef || requestedRef.split('/')[0].toLowerCase() !== account.owner.toLowerCase()) throw new Error('P0 status target mismatch');
        const s = await status(env, account, requestedRef);
        const receipt = s === 'COMPLETE' ? await validatedReceipt(env, account, requestedRef) : null;
        return json({ project: 'PNEUMONIA V6.2.3-P0', account_id: account.accountId, target_kernel_ref: requestedRef, status: s, receipt, automatic_rerun_forbidden: true });
      }
      return canonicalWorker.fetch(request, env, ctx);
    } catch (error) {
      return json({ project: 'PNEUMONIA V6.2.3-P0', ok: false, error: error instanceof Error ? error.message.slice(0, 1800) : 'unknown error', automatic_rerun_forbidden: true, locked_test_used: false, external_data_used: false }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
