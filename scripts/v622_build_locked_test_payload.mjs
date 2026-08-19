import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const EXPECTED_NOTEBOOK_SHA = '5abff78b9b10a13792b6d1b6cd24df780a64d16b7ec891d2ed74628dbb0b910b';
const EXPECTED_SOURCE_ZIP_SHA = 'ce3623b87ca221a52bcf886836070ebaebc490cf7574f6f3ce4db78095a32cde';
const RECONCILIATION_SHA = '99a319c8d397342aaefd3c15e0eb8f968a973738f55ef97925a2a16da8c485d8';

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}
function fail(message) {
  throw new Error(message);
}

const [artifactDirArg, templateArg, payloadArg, metadataArg] = process.argv.slice(2);
if (!artifactDirArg || !templateArg || !payloadArg || !metadataArg) {
  fail('usage: node v622_build_locked_test_payload.mjs <artifact_dir> <template.py> <payload.ts> <metadata.json>');
}
const artifactDir = path.resolve(artifactDirArg);
const notebookPath = path.join(artifactDir, 'w16-kernel.ipynb');
const templatePath = path.resolve(templateArg);
const payloadPath = path.resolve(payloadArg);
const metadataPath = path.resolve(metadataArg);
if (!fs.existsSync(notebookPath)) fail(`missing recovered W16 notebook: ${notebookPath}`);
if (!fs.existsSync(templatePath)) fail(`missing locked-test template: ${templatePath}`);

const notebookBytes = fs.readFileSync(notebookPath);
const notebookSha = sha256(notebookBytes);
if (notebookSha !== EXPECTED_NOTEBOOK_SHA) fail(`W16 notebook SHA drift: ${notebookSha}`);
const notebook = JSON.parse(notebookBytes.toString('utf8'));
const code = (notebook.cells ?? [])
  .filter((cell) => cell && cell.cell_type === 'code')
  .map((cell) => Array.isArray(cell.source) ? cell.source.join('') : String(cell.source ?? ''))
  .join('\n');
const match = code.match(/ARCHIVE_B64\s*=\s*['\"]([A-Za-z0-9+/=]+)['\"]/s);
if (!match) fail('ARCHIVE_B64 not found in recovered W16 source');
const sourceZip = Buffer.from(match[1], 'base64');
const sourceZipSha = sha256(sourceZip);
if (sourceZipSha !== EXPECTED_SOURCE_ZIP_SHA) fail(`W16 source archive SHA drift: ${sourceZipSha}`);

let script = fs.readFileSync(templatePath, 'utf8');
if (!script.includes('__RECONCILIATION_SHA__') || !script.includes('__SOURCE_ZIP_B64__')) {
  fail('locked-test template placeholders are missing');
}
script = script
  .replaceAll('__RECONCILIATION_SHA__', RECONCILIATION_SHA)
  .replaceAll('__SOURCE_ZIP_B64__', sourceZip.toString('base64'));
if (script.includes('__RECONCILIATION_SHA__') || script.includes('__SOURCE_ZIP_B64__')) {
  fail('locked-test template replacement incomplete');
}

const scriptSha = sha256(Buffer.from(script, 'utf8'));
fs.mkdirSync(path.dirname(payloadPath), { recursive: true });
fs.writeFileSync(payloadPath,
  `// Generated only on the trusted runner from recovered W16 source. Do not commit this payload.\n` +
  `export const OFFICIAL_LOCKED_TEST_SCRIPT = ${JSON.stringify(script)} as const;\n` +
  `export const OFFICIAL_LOCKED_TEST_SCRIPT_SHA256 = ${JSON.stringify(scriptSha)} as const;\n` +
  `export const W16_NOTEBOOK_SHA256 = ${JSON.stringify(notebookSha)} as const;\n` +
  `export const W16_SOURCE_ZIP_SHA256 = ${JSON.stringify(sourceZipSha)} as const;\n`,
  'utf8');
const metadata = {
  project: 'PNEUMONIA V6.2.2',
  stage: 'OFFICIAL_LOCKED_TEST_PAYLOAD_BUILD',
  w16_notebook_sha256: notebookSha,
  w16_source_zip_sha256: sourceZipSha,
  final_freeze_manifest_sha256: '2313023819c9eec669d835b99dcc2524b03bb0da236c2ec8f98c4e1221948414',
  reconciliation_sha256: RECONCILIATION_SHA,
  official_locked_test_script_sha256: scriptSha,
  locked_benchmark_expected_samples: 624,
  training_hpo_confirmation: false,
  external_data_used: false,
};
fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2) + '\n', 'utf8');
console.log(JSON.stringify(metadata, null, 2));
