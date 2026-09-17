const GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com";
const GITHUB_OIDC_JWKS = "https://token.actions.githubusercontent.com/.well-known/jwks";
const READ_AUDIENCE = "cgp-control-plane-v3";
const ACTION_AUDIENCE = "cgp-control-plane-v3-action";
const TRUSTED_REPOSITORY = "rezanory/chatgpt-plugins";
const TRUSTED_REPOSITORY_ID = "1337215097";
const TRUSTED_OWNER_ID = "62356000";
const TRUSTED_ACTOR_ID = "62356000";
const TRUSTED_REF = "refs/heads/main";
const M07_REPAIR_REF = "refs/heads/fix/m07-r320-producer-source-v1";
const ACTION_TRUSTED_REFS = new Set([TRUSTED_REF, M07_REPAIR_REF]);
const READ_WORKFLOW_EVENTS = new Map<string, ReadonlySet<string>>([
  [
    "rezanory/chatgpt-plugins/.github/workflows/control-plane-v3-query.yml@refs/heads/main",
    new Set(["issue_comment", "workflow_dispatch"]),
  ],
]);
const ACTION_WORKFLOW_EVENTS = new Map<string, ReadonlySet<string>>([
  [
    "rezanory/chatgpt-plugins/.github/workflows/pneumonia-v17-8acct-20260912.yml@refs/heads/main",
    new Set(["issue_comment", "workflow_dispatch"]),
  ],
  [
    "rezanory/chatgpt-plugins/.github/workflows/pneumonia-v17-m07-continuation-20260914.yml@refs/heads/main",
    new Set(["workflow_dispatch"]),
  ],
  [
    "rezanory/chatgpt-plugins/.github/workflows/pneumonia-v17-m07-continuation-20260914.yml@refs/heads/fix/m07-r320-producer-source-v1",
    new Set(["workflow_dispatch"]),
  ],
]);
const CLOCK_SKEW_SECONDS = 60;

type Rec = Record<string, unknown>;
type BrokerKind = "read" | "action";

export interface VerifiedGitHubOidcIdentity {
  broker_kind: BrokerKind;
  repository: string;
  repository_id: string;
  repository_owner_id: string;
  actor_id: string;
  run_id: string;
  workflow_ref: string;
  ref: string;
  sha: string;
  event_name: string;
}

function object(value: unknown): Rec {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("OIDC value is not an object");
  }
  return value as Rec;
}

function base64UrlBytes(value: string): Uint8Array {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) throw new Error("invalid base64url segment");
  const padded = value.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(value.length / 4) * 4, "=");
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function concreteBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function decodeJsonSegment(value: string): Rec {
  const bytes = base64UrlBytes(value);
  return object(JSON.parse(new TextDecoder().decode(bytes)));
}

function claimString(claims: Rec, key: string): string {
  const value = claims[key];
  if (typeof value !== "string" || !value.trim()) throw new Error(`OIDC claim ${key} missing`);
  return value;
}

function numericClaim(claims: Rec, key: string): number {
  const value = claims[key];
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) throw new Error(`OIDC claim ${key} invalid`);
  return numeric;
}

function audienceMatches(value: unknown, expected: string): boolean {
  if (typeof value === "string") return value === expected;
  return Array.isArray(value) && value.some((item) => item === expected);
}

async function signingKey(kid: string): Promise<CryptoKey> {
  const response = await fetch(GITHUB_OIDC_JWKS, {
    headers: { Accept: "application/json", "User-Agent": "chatgpt-control-plane-v3-oidc/1.1" },
  });
  if (!response.ok) throw new Error(`GitHub OIDC JWKS HTTP ${response.status}`);
  const payload = object(await response.json());
  const keys = Array.isArray(payload.keys) ? payload.keys : [];
  const candidate = keys.find((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return false;
    const key = item as Rec;
    return key.kid === kid && key.kty === "RSA" && key.use === "sig" && key.alg === "RS256";
  });
  if (!candidate) throw new Error("GitHub OIDC signing key not found");
  return crypto.subtle.importKey(
    "jwk",
    candidate as JsonWebKey,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["verify"],
  );
}

async function verifyBrokerOidc(
  request: Request,
  kind: BrokerKind,
): Promise<VerifiedGitHubOidcIdentity> {
  const authorization = request.headers.get("authorization") ?? "";
  if (!authorization.startsWith("Bearer ")) throw new Error("GitHub OIDC bearer token missing");
  const token = authorization.slice(7).trim();
  if (token.length < 100 || token.length > 20_000) throw new Error("GitHub OIDC token length invalid");
  const segments = token.split(".");
  if (segments.length !== 3) throw new Error("GitHub OIDC JWT shape invalid");

  const header = decodeJsonSegment(segments[0]);
  const claims = decodeJsonSegment(segments[1]);
  const kid = claimString(header, "kid");
  if (header.alg !== "RS256" || header.typ !== "JWT") throw new Error("GitHub OIDC JWT algorithm/type invalid");

  const key = await signingKey(kid);
  const signingInput = concreteBuffer(new TextEncoder().encode(`${segments[0]}.${segments[1]}`));
  const signature = concreteBuffer(base64UrlBytes(segments[2]));
  const verified = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, signature, signingInput);
  if (!verified) throw new Error("GitHub OIDC JWT signature invalid");

  const expectedAudience = kind === "read" ? READ_AUDIENCE : ACTION_AUDIENCE;
  const allowedWorkflows = kind === "read" ? READ_WORKFLOW_EVENTS : ACTION_WORKFLOW_EVENTS;
  const now = Math.floor(Date.now() / 1000);
  const exp = numericClaim(claims, "exp");
  const nbf = claims.nbf === undefined ? now : numericClaim(claims, "nbf");
  const iat = numericClaim(claims, "iat");
  if (exp < now - CLOCK_SKEW_SECONDS) throw new Error("GitHub OIDC JWT expired");
  if (nbf > now + CLOCK_SKEW_SECONDS) throw new Error("GitHub OIDC JWT not active yet");
  if (iat > now + CLOCK_SKEW_SECONDS || iat < now - 15 * 60) {
    throw new Error("GitHub OIDC JWT issue time outside broker window");
  }
  if (claims.iss !== GITHUB_OIDC_ISSUER) throw new Error("GitHub OIDC issuer mismatch");
  if (!audienceMatches(claims.aud, expectedAudience)) throw new Error("GitHub OIDC audience mismatch");

  const repository = claimString(claims, "repository");
  const repositoryId = claimString(claims, "repository_id");
  const ownerId = claimString(claims, "repository_owner_id");
  const actorId = claimString(claims, "actor_id");
  const workflowRef = claimString(claims, "workflow_ref");
  const ref = claimString(claims, "ref");
  const eventName = claimString(claims, "event_name");
  const runId = claimString(claims, "run_id");
  const sha = claimString(claims, "sha");

  if (repository !== TRUSTED_REPOSITORY || repositoryId !== TRUSTED_REPOSITORY_ID) {
    throw new Error("GitHub OIDC repository identity mismatch");
  }
  if (ownerId !== TRUSTED_OWNER_ID || actorId !== TRUSTED_ACTOR_ID) {
    throw new Error("GitHub OIDC owner/actor identity mismatch");
  }
  const allowedEvents = allowedWorkflows.get(workflowRef);
  const refAllowed = kind === "read" ? ref === TRUSTED_REF : ACTION_TRUSTED_REFS.has(ref);
  if (!allowedEvents || !allowedEvents.has(eventName) || !refAllowed) {
    throw new Error("GitHub OIDC workflow/ref/event mismatch");
  }
  if (claims.repository_visibility !== "private") throw new Error("GitHub OIDC repository visibility mismatch");
  if (!/^\d+$/.test(runId) || !/^[0-9a-f]{40}$/.test(sha)) {
    throw new Error("GitHub OIDC run/sha claim invalid");
  }

  return {
    broker_kind: kind,
    repository,
    repository_id: repositoryId,
    repository_owner_id: ownerId,
    actor_id: actorId,
    run_id: runId,
    workflow_ref: workflowRef,
    ref,
    sha,
    event_name: eventName,
  };
}

export function verifyGitHubReadBrokerOidc(
  request: Request,
): Promise<VerifiedGitHubOidcIdentity> {
  return verifyBrokerOidc(request, "read");
}

export function verifyGitHubActionBrokerOidc(
  request: Request,
): Promise<VerifiedGitHubOidcIdentity> {
  return verifyBrokerOidc(request, "action");
}
