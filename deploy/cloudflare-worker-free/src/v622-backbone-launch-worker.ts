import { getKernel, kernelStatus, listKernels, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

const ACCOUNT_ID = "kg-05";
const USERNAME = "trickermark";
const SOURCE_REF = "trickermark/pneumonia-v6-2-2-train-w16-m06-r224";
const TARGET_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";
const TARGET_SLUG = "pneumonia-v6-2-2-backbone-m06-r224";
const TARGET_TITLE = "PNEUMONIA V6.2.2 BACKBONE M06 R224";
const EXPECTED_SOURCE_SHA256 = "5abff78b9b10a13792b6d1b6cd24df780a64d16b7ec891d2ed74628dbb0b910b";
const EXPECTED_SOURCE_FINGERPRINT = "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838";
const RECIPE_SHA256 = "27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f";
const EXPECTED_TRANSFORMED_CODE_SHA256 = "03724400a3dd744f9e952a02c4a6ff202c29bcf93942683ff74c8b9982a24f67";
const BACKBONES = ["convnext_tiny", "densenet121", "efficientnetv2b0", "resnet50v2"] as const;
const RECIPE = {"schema_version":1,"purpose":"FAIR_POST_ABLATION_BACKBONE_COMPARISON","selected_model_id":"M06","selected_convnext_experiment":{"model_id":"M06","loss_name":"dynamic_focal","backbone_name":"convnext_tiny","use_cbam":false,"use_edge":true,"batch_policy":"none","use_ema":false,"use_swa":false,"description":"ConvNeXt-Tiny + trainable EdgeBlock + FLSD-53 focal."},"frozen_training_arguments":{"image_size":224,"batch_size":8,"balanced_batches":true,"head_epochs":3,"finetune_epochs":47,"head_lr":0.000058025153801026264,"finetune_lr":0.00004723037146949559,"weight_decay":0.00001637057637057399,"lr_schedule":"plateau","min_lr":0.000001,"head_warmup_epochs":0,"finetune_warmup_epochs":0,"dropout":0.15,"edge_filters":16,"edge_max_gate":0.25,"cbam_reduction":16,"cbam_spatial_kernel":7,"ema_momentum":0.99,"patience":6,"mixed_precision":"auto","deterministic":true,"num_parallel_calls":4,"prefetch":2,"mix_alpha":0.2,"static_gamma":2,"flsd_threshold":0.2,"flsd_hard_gamma":5,"flsd_easy_gamma":3,"swa_start":8,"weights":"imagenet","adaptive_evidence_zoom":true,"aez_uncertainty_low":0.3,"aez_uncertainty_high":0.7,"aez_crop_ratio":0.45,"aez_min_crop_ratio":0.25,"aez_max_crop_ratio":0.7,"aez_context":0.15,"aez_activation_quantile":0.85,"aez_map_method":"dual_gradcam","aez_save_examples":8,"aez_max_samples":0},"seed_policy":"USE_IDENTICAL_SEED_LIST_FOR_EVERY_BACKBONE","reference_seed":42,"outer_image_preprocessing_fixed":true,"backbone_native_serialized_preprocessing_allowed":true,"preprocessing_contract":{"schema_version":1,"policy_id":"BACKBONE_COMPARISON_R224","image_size":224,"resize_operation":"tensorflow.image.resize_with_pad","aspect_ratio":"PRESERVE_WITH_SYMMETRIC_PADDING","padding":"CONSTANT_ZERO_BY_TF_RESIZE_WITH_PAD","interpolation":"BILINEAR","antialias":true,"channel_count":3,"channel_order":"RGB","decode_operation":"tensorflow.io.decode_image","expand_animations":false,"dtype":"float32","pixel_domain":"FLOAT32_0_255","pixel_scaling":"NONE","clip_min":0,"clip_max":255,"backbone_name":"BACKBONE_NATIVE_SERIALIZED_TRANSFORM","backbone_preprocessing":"SERIALIZED_INSIDE_MODEL","probability_polarity":"PNEUMONIA_POSITIVE_CLASS_1","training_reference":"pneumonia_runner_base.decode_resize","contract_sha256":"7a905f70b5a62bd5fbee0a3d02cf59b9ec3c6865cf2381303b6dc37b037bfd0f"},"source_run_dir":"/mnt/data/v622_freeze_work/selected_run","source_config_sha256":"afe5640451245449a97f9ad0ca61d5ce95f714bbc208de8b3e8aceae9ba8bf94","source_final_report_sha256":"df0fbbb9687f3bb04a9726d1ce83876e1c2bb64d90714b3d3ca19055ee97b923","allowed_backbones":["convnext_tiny","densenet121","efficientnetv2b0","resnet50v2"],"only_backbone_may_change":true,"locked_test_allowed_for_selection":false,"external_data_allowed_for_selection":false,"recipe_sha256":"27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f"};

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function authorized(request: Request, env: Env): boolean {
  const token = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(token && token.length >= 32 && request.headers.get("authorization") === `Bearer ${token}`);
}
function rec(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), b => b.toString(16).padStart(2, "0")).join("");
}
async function sha256(text: string): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}
function authHeader(token: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${USERNAME}:${token}`)}`;
}
function copyIfPresent(target: Record<string, unknown>, source: Record<string, unknown>, key: string): void {
  const value = source[key];
  if (value !== undefined && value !== null && value !== "") target[key] = value;
}
function cellCode(notebook: Record<string, unknown>): { cell: Record<string, unknown>; code: string } {
  const cells = Array.isArray(notebook.cells) ? notebook.cells : [];
  if (cells.length !== 1) throw new Error(`W16 notebook cell-count changed: ${cells.length}`);
  const cell = rec(cells[0]);
  const source = cell.source;
  const parts = typeof source === "string" ? [source] : Array.isArray(source) ? source.map(x => typeof x === "string" ? x : "") : [];
  if (!parts.length) throw new Error("W16 notebook source is empty");
  return { cell, code: parts.join("") };
}
async function transformedNotebook(source: string): Promise<string> {
  if (await sha256(source) !== EXPECTED_SOURCE_SHA256) throw new Error("W16 source SHA-256 mismatch");
  if (!source.includes(EXPECTED_SOURCE_FINGERPRINT)) throw new Error("W16 source fingerprint mismatch");
  const notebook = rec(JSON.parse(source));
  const { cell, code: originalCode } = cellCode(notebook);
  const oldWork = "WORK=Path('/kaggle/working/PNEUMONIA_V62_2_TRAIN_W16')";
  if (originalCode.split(oldWork).length - 1 !== 1) throw new Error("W16 WORK marker mismatch");
  let code = originalCode.replace(oldWork, "WORK=Path('/kaggle/working/PNEUMONIA_V62_2_BACKBONE_M06')");
  const marker = "cmd=[sys.executable,str(PROJECT/'run_experiment_matrix.py')";
  if (code.split(marker).length - 1 !== 1) throw new Error("W16 training marker mismatch");
  const pos = code.indexOf(marker);
  const prefix = code.slice(0, pos);
  const recipeJson = JSON.stringify(RECIPE);
  const block = `BACKBONE_RECIPE=json.loads(r'''${recipeJson}''')\nrecipe_path=WORK/'FROZEN_SELECTED_BACKBONE_RECIPE.json'\nrecipe_path.write_text(json.dumps(BACKBONE_RECIPE,indent=2)+"\\n",encoding='utf-8')\ncomparison_root=WORK/'backbone_comparison'\nsubprocess.run([sys.executable,str(PROJECT/'run_backbone_comparison.py'),'--recipe',str(recipe_path),'--data-root',str(effective),'--source-root',str(PROJECT),'--output-root',str(comparison_root),'--backbones','convnext_tiny,densenet121,efficientnetv2b0,resnet50v2','--seeds','42'],check=True)\nselection_root=WORK/'backbone_selection'\nsubprocess.run([sys.executable,str(PROJECT/'backbone_ensemble_selection.py'),'--run-index',str(comparison_root/'BACKBONE_COMPARISON_RUN_INDEX.csv'),'--output-dir',str(selection_root),'--folds','5','--seed','2026','--weight-step','0.05','--minimum-mean-weight','0.05'],check=True)\nprint(json.dumps({'status':'PASS','stage':'POST_ABLATION_BACKBONE_COMPARISON','selected_model_id':'M06','recipe_sha256':BACKBONE_RECIPE['recipe_sha256'],'backbones':['convnext_tiny','densenet121','efficientnetv2b0','resnet50v2'],'seeds':[42],'locked_test_used':False,'external_data_used':False},indent=2))\n`;
  code = prefix + block;
  if (code.includes("run_experiment_matrix.py")) throw new Error("canonical COMPLETE training invocation survived transformation");
  if (code.split("run_backbone_comparison.py").length - 1 !== 1) throw new Error("backbone runner count mismatch");
  if (code.split("backbone_ensemble_selection.py").length - 1 !== 1) throw new Error("backbone selector count mismatch");
  if (code.includes("evaluate_saved_champion_test.py") || code.includes("final-test-confirm")) throw new Error("locked-test path present in transformed notebook");
  if (await sha256(code) !== EXPECTED_TRANSFORMED_CODE_SHA256) throw new Error("transformed code SHA-256 mismatch");
  notebook.cells = [{ ...cell, source: [code] }];
  return JSON.stringify(notebook);
}
async function saveKernel(env: Env, sourcePayload: Record<string, unknown>, text: string): Promise<Record<string, unknown>> {
  const token = env.CGP_KAGGLE_KG05_TOKEN?.trim();
  if (!token) throw new Error("Kaggle kg-05 token missing");
  const metadata = rec(sourcePayload.metadata);
  const blob = rec(sourcePayload.blob);
  const request: Record<string, unknown> = {
    slug: TARGET_REF,
    newTitle: TARGET_TITLE,
    text,
    kernelExecutionType: "SAVE_AND_RUN_ALL",
  };
  copyIfPresent(request, blob, "language");
  if (request.language === undefined) copyIfPresent(request, metadata, "language");
  copyIfPresent(request, blob, "kernelType");
  if (request.kernelType === undefined) copyIfPresent(request, metadata, "kernelType");
  for (const key of ["datasetDataSources","kernelDataSources","competitionDataSources","categoryIds","isPrivate","enableGpu","enableInternet","dockerImagePinningType","modelDataSources","enableTpu","sessionTimeoutSeconds","priority","dockerImage","machineShape"]) copyIfPresent(request, metadata, key);
  const response = await fetch(`${API_ROOT}/SaveKernel`, {
    method: "POST",
    headers: { Authorization: authHeader(token), "Content-Type": "application/json", "User-Agent": "chatgpt-v622-backbone-launch/1.0" },
    body: JSON.stringify(request),
  });
  const raw = await response.text();
  let result: Record<string, unknown> = {};
  try { result = rec(raw ? JSON.parse(raw) : {}); } catch { throw new Error(`SaveKernel non-JSON HTTP ${response.status}`); }
  const code = typeof result.code === "number" ? result.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) throw new Error(String(result.message ?? `SaveKernel HTTP ${response.status}`).slice(0, 1000));
  return result;
}
async function launch(env: Env): Promise<Record<string, unknown>> {
  const existing = await listKernels(env, ACCOUNT_ID, TARGET_SLUG, 100);
  for (const item of existing) {
    const row = rec(item);
    if (String(row.ref ?? "").toLowerCase() === TARGET_REF.toLowerCase()) {
      let status: Record<string, unknown> = {};
      try { status = await kernelStatus(env, ACCOUNT_ID, TARGET_REF); } catch { /* diagnostic only */ }
      return { action: "existing", target_kernel_ref: TARGET_REF, status, recipe_sha256: RECIPE_SHA256, backbones: BACKBONES, seeds: [42] };
    }
  }
  const sourcePayload = await getKernel(env, ACCOUNT_ID, SOURCE_REF);
  const sourceBlob = rec(sourcePayload.blob);
  const source = typeof sourceBlob.source === "string" ? sourceBlob.source : "";
  if (!source) throw new Error("W16 GetKernel returned no source");
  const text = await transformedNotebook(source);
  const result = await saveKernel(env, sourcePayload, text);
  return { action: "submitted", target_kernel_ref: TARGET_REF, result, recipe_sha256: RECIPE_SHA256, source_sha256: EXPECTED_SOURCE_SHA256, transformed_code_sha256: EXPECTED_TRANSFORMED_CODE_SHA256, backbones: BACKBONES, seeds: [42], locked_test_used: false, external_data_used: false, canonical_m01_m12_rerun: false };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") return json({ service: "v622-backbone-launch", status: "ready", protected: true });
    if (url.pathname !== "/control/v6-2-2/launch-backbone-comparison" || request.method !== "POST") return new Response("Not found", { status: 404 });
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try { return json({ project: "PNEUMONIA V6.2.2", stage: "post_ablation_backbone_comparison", ...(await launch(env)) }); }
    catch (error) {
      return json({
        project: "PNEUMONIA V6.2.2",
        stage: "post_ablation_backbone_comparison",
        action: "blocked",
        target_kernel_ref: TARGET_REF,
        recipe_sha256: RECIPE_SHA256,
        error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
        kaggle_submit_performed: false,
      });
    }
  },
} satisfies ExportedHandler<Env>;
