import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const [sourceDirArg, templateArg, outputArg] = process.argv.slice(2);
if (!sourceDirArg || !templateArg || !outputArg) {
  throw new Error('usage: node scripts/v623_build_p0_payload.mjs <source-artifact-dir> <template.py> <output.ts>');
}

const EXPECTED_NOTEBOOK_SHA = '5abff78b9b10a13792b6d1b6cd24df780a64d16b7ec891d2ed74628dbb0b910b';
const EXPECTED_SOURCE_ZIP_SHA = 'ce3623b87ca221a52bcf886836070ebaebc490cf7574f6f3ce4db78095a32cde';
const PLACEHOLDER = '__SOURCE_ZIP_B64__';
const sha256 = (value) => crypto.createHash('sha256').update(value).digest('hex');

function replaceExactly(source, oldText, newText, label) {
  const count = source.split(oldText).length - 1;
  if (count !== 1) {
    throw new Error(`${label} replacement cardinality mismatch: ${count}`);
  }
  return source.replace(oldText, newText);
}

function applyStructuralNearDuplicateGate(template) {
  let patched = template;

  patched = replaceExactly(
    patched,
    'NEAR_DUP_HAMMING = 2',
    [
      'DHASH_CANDIDATE_HAMMING = 2',
      'PHASH_CANDIDATE_HAMMING = 4',
      'PIXEL_CORR_MIN = 0.97',
      'GRADIENT_CORR_MIN = 0.75',
      'STRUCTURAL_SIZE = 128',
    ].join('\n'),
    'near-duplicate constants',
  );

  patched = replaceExactly(
    patched,
    '_hash_cache: dict[str, str] = {}\n_dhash_cache: dict[str, int] = {}',
    [
      '_hash_cache: dict[str, str] = {}',
      '_dhash_cache: dict[str, int] = {}',
      '_phash_cache: dict[str, int] = {}',
      '_structural_cache: dict[str, np.ndarray] = {}',
    ].join('\n'),
    'near-duplicate caches',
  );

  const oldDhashBlock = `def dhash(path: Path) -> int:
    key = str(path)
    if key not in _dhash_cache:
        with Image.open(path) as image:
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(gray, dtype=np.uint8)
        bits = pixels[:, 1:] > pixels[:, :-1]
        value = 0
        for bit in bits.reshape(-1):
            value = (value << 1) | int(bit)
        _dhash_cache[key] = value
    return _dhash_cache[key]`;

  const newHashAndStructuralBlock = `def dhash(path: Path) -> int:
    key = str(path)
    if key not in _dhash_cache:
        with Image.open(path) as image:
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(gray, dtype=np.uint8)
        bits = pixels[:, 1:] > pixels[:, :-1]
        value = 0
        for bit in bits.reshape(-1):
            value = (value << 1) | int(bit)
        _dhash_cache[key] = value
    return _dhash_cache[key]


def phash(path: Path) -> int:
    key = str(path)
    if key not in _phash_cache:
        with Image.open(path) as image:
            pixels = np.asarray(
                image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
        frequencies = np.abs(np.fft.fft2(pixels))[:8, :8].reshape(-1)
        median = np.median(frequencies[1:])
        value = 0
        for bit in frequencies > median:
            value = (value << 1) | int(bit)
        _phash_cache[key] = value
    return _phash_cache[key]


def structural_image(path: Path) -> np.ndarray:
    key = str(path)
    cached = _structural_cache.get(key)
    if cached is not None:
        return cached
    with Image.open(path) as image:
        value = np.asarray(
            image.convert("L").resize(
                (STRUCTURAL_SIZE, STRUCTURAL_SIZE), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0
    _structural_cache[key] = value
    return value


def array_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1)
    b = right.reshape(-1)
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def structural_metrics(left: Path, right: Path) -> tuple[float, float, float]:
    a = structural_image(left)
    b = structural_image(right)
    pixel_corr = array_correlation(a, b)
    a_gradient = np.concatenate(
        [np.diff(a, axis=0).reshape(-1), np.diff(a, axis=1).reshape(-1)]
    )
    b_gradient = np.concatenate(
        [np.diff(b, axis=0).reshape(-1), np.diff(b, axis=1).reshape(-1)]
    )
    gradient_corr = array_correlation(a_gradient, b_gradient)
    pixel_mae = float(np.mean(np.abs(a - b)))
    return pixel_corr, gradient_corr, pixel_mae`;

  patched = replaceExactly(
    patched,
    oldDhashBlock,
    newHashAndStructuralBlock,
    'near-duplicate hash helpers',
  );

  patched = replaceExactly(
    patched,
    'rows.append({"split": split_name, "sample": sample, "sha256": file_hash(sample.path), "dhash": dhash(sample.path)})',
    'rows.append({"split": split_name, "sample": sample, "sha256": file_hash(sample.path), "dhash": dhash(sample.path), "phash": phash(sample.path)})',
    'near-duplicate row features',
  );

  const oldNearBlock = `    near_cross = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["split"] == right["split"] or left["sha256"] == right["sha256"]:
                continue
            dist = (int(left["dhash"]) ^ int(right["dhash"])).bit_count()
            if dist <= NEAR_DUP_HAMMING:
                near_cross.append({"a": left["split"], "b": right["split"], "distance": int(dist), "a_name": left["sample"].path.name, "b_name": right["sample"].path.name})
                if len(near_cross) >= 50:
                    break
        if len(near_cross) >= 50:
            break

    return {"patient_overlap": patient_overlap, "exact_duplicate_cross_split": exact_cross, "near_duplicate_cross_split": near_cross, "near_duplicate_hamming_threshold": NEAR_DUP_HAMMING, "pass": not patient_overlap and not exact_cross and not near_cross}`;

  const newNearBlock = `    near_cross = []
    candidate_pairs = 0
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["split"] == right["split"] or left["sha256"] == right["sha256"]:
                continue
            dhash_distance = (int(left["dhash"]) ^ int(right["dhash"])).bit_count()
            phash_distance = (int(left["phash"]) ^ int(right["phash"])).bit_count()
            if dhash_distance > DHASH_CANDIDATE_HAMMING and phash_distance > PHASH_CANDIDATE_HAMMING:
                continue
            candidate_pairs += 1
            pixel_corr, gradient_corr, pixel_mae = structural_metrics(left["sample"].path, right["sample"].path)
            if pixel_corr < PIXEL_CORR_MIN or gradient_corr < GRADIENT_CORR_MIN:
                continue
            near_cross.append({
                "a": left["split"], "b": right["split"],
                "dhash_distance": int(dhash_distance), "phash_distance": int(phash_distance),
                "pixel_corr": pixel_corr, "gradient_corr": gradient_corr, "pixel_mae": pixel_mae,
                "a_name": left["sample"].path.name, "b_name": right["sample"].path.name,
                "a_patient": left["sample"].patient_id, "b_patient": right["sample"].patient_id,
            })
            if len(near_cross) >= 50:
                break
        if len(near_cross) >= 50:
            break

    near_duplicate_policy = {
        "candidate_rule": "dhash64_hamming<=2 OR fft_phash64_hamming<=4",
        "confirmation_rule": "pixel_corr>=0.97 AND gradient_corr>=0.75",
        "dhash_candidate_hamming_max": DHASH_CANDIDATE_HAMMING,
        "phash_candidate_hamming_max": PHASH_CANDIDATE_HAMMING,
        "pixel_corr_min": PIXEL_CORR_MIN,
        "gradient_corr_min": GRADIENT_CORR_MIN,
        "structural_resize": [STRUCTURAL_SIZE, STRUCTURAL_SIZE],
        "calibration_inventory_samples": 5232,
        "calibration_exact_cross_patient_sha_groups": 0,
        "calibration_max_observed_false_positive_pixel_corr": 0.9048865772067772,
        "calibration_max_observed_false_positive_gradient_corr": 0.2820293799227516,
    }
    return {
        "patient_overlap": patient_overlap,
        "exact_duplicate_cross_split": exact_cross,
        "near_duplicate_cross_split": near_cross,
        "near_duplicate_candidate_pairs_checked": candidate_pairs,
        "near_duplicate_policy": near_duplicate_policy,
        "pass": not patient_overlap and not exact_cross and not near_cross,
    }`;

  patched = replaceExactly(
    patched,
    oldNearBlock,
    newNearBlock,
    'near-duplicate audit',
  );

  if (patched.includes('NEAR_DUP_HAMMING')) {
    throw new Error('legacy dHash-only near-duplicate gate remains after patch');
  }
  for (const marker of [
    'DHASH_CANDIDATE_HAMMING = 2',
    'PHASH_CANDIDATE_HAMMING = 4',
    'PIXEL_CORR_MIN = 0.97',
    'GRADIENT_CORR_MIN = 0.75',
    'near_duplicate_candidate_pairs_checked',
  ]) {
    if (!patched.includes(marker)) throw new Error(`patched P0 template missing marker: ${marker}`);
  }
  return patched;
}

const notebookPath = path.join(sourceDirArg, 'w16-kernel.ipynb');
if (!fs.existsSync(notebookPath)) throw new Error(`missing recovered notebook: ${notebookPath}`);
const notebookBytes = fs.readFileSync(notebookPath);
const notebookSha = sha256(notebookBytes);
if (notebookSha !== EXPECTED_NOTEBOOK_SHA) throw new Error(`W16 notebook SHA drift: ${notebookSha}`);

const notebook = JSON.parse(notebookBytes.toString('utf8'));
const code = (notebook.cells ?? [])
  .filter((cell) => cell && cell.cell_type === 'code')
  .map((cell) => Array.isArray(cell.source) ? cell.source.join('') : String(cell.source ?? ''))
  .join('\n');
const match = code.match(/ARCHIVE_B64\s*=\s*['"]([A-Za-z0-9+/=]+)['"]/s);
if (!match) throw new Error('ARCHIVE_B64 not found in recovered W16 source');
const sourceZipBytes = Buffer.from(match[1], 'base64');
const sourceZipSha = sha256(sourceZipBytes);
if (sourceZipSha !== EXPECTED_SOURCE_ZIP_SHA) throw new Error(`W16 source ZIP SHA drift: ${sourceZipSha}`);

const rawTemplate = fs.readFileSync(templateArg, 'utf8');
if ((rawTemplate.split(PLACEHOLDER).length - 1) !== 1) throw new Error('P0 template source placeholder cardinality mismatch');
const calibratedTemplate = applyStructuralNearDuplicateGate(rawTemplate);
const script = calibratedTemplate.replace(PLACEHOLDER, sourceZipBytes.toString('base64'));
if (script.includes(PLACEHOLDER)) throw new Error('P0 template replacement incomplete');
const scriptSha = sha256(Buffer.from(script, 'utf8'));

const githubRunId = String(process.env.GITHUB_RUN_ID ?? '').trim();
if (!/^\d+$/.test(githubRunId)) throw new Error('GITHUB_RUN_ID missing or invalid');
const expiresAtMs = Date.now() + 6 * 60 * 60 * 1000;

const out = [
  '// GENERATED BY scripts/v623_build_p0_payload.mjs — DO NOT COMMIT GENERATED CONTENT',
  `export const P0_SCRIPT: string = ${JSON.stringify(script)};`,
  `export const P0_SCRIPT_SHA256: string = ${JSON.stringify(scriptSha)};`,
  `export const W16_NOTEBOOK_SHA256: string = ${JSON.stringify(notebookSha)};`,
  `export const W16_SOURCE_ZIP_SHA256: string = ${JSON.stringify(sourceZipSha)};`,
  `export const P0_GITHUB_RUN_ID: string = ${JSON.stringify(githubRunId)};`,
  `export const P0_EXPIRES_AT_MS: number = ${expiresAtMs};`,
  '',
].join('\n');
fs.writeFileSync(outputArg, out, 'utf8');
console.log(JSON.stringify({
  p0_payload: 'PASS',
  near_duplicate_gate: 'structural-confirmed-v2',
  script_sha256: scriptSha,
  w16_notebook_sha256: notebookSha,
  w16_source_zip_sha256: sourceZipSha,
  github_run_id: githubRunId,
  expires_at_ms: expiresAtMs,
}));
