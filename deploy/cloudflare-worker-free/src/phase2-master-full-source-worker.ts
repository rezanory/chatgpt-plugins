import { getKernel, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

const PROJECT = "PNEUMONIA Phase-2";
const PURPOSE = "master_full_kernel_source_recovery";

const KERNELS: Readonly<Record<string, string>> = {
  "azadka/pneumonia-v1-7-phase2-m01-r224-a18-35698388883": "master",
  "azadka/pneumonia-v1-7-phase2-m01-r320-a05-35814596649": "master",
  "azadka/pneumonia-v1-7-phase2-m01-r384-a05-36351862662": "master",
  "radlinaradlina/pneumonia-v1-7-phase2-m02-r224-a19-35698392981": "kg-02",
  "radlinaradlina/pneumonia-v1-7-phase2-m02-r320-a05-35814599480": "kg-02",
  "radlinaradlina/pneumonia-v1-7-phase2-m02-r384-a05-35927816711": "kg-02",
  "rezanory/pneumonia-v1-7-phase2-m03-r224-a19-35698396139": "kg-03",
  "rezanory/pneumonia-v1-7-phase2-m03-r320-a05-35814602879": "kg-03",
  "rezanory/pneumonia-v1-7-phase2-m03-r384-a05-36362420209": "kg-03",
  "reyhanehazad/pneumonia-v1-7-phase2-m04-r224-a19-35698399363": "kg-04",
  "reyhanehazad/pneumonia-v1-7-phase2-m04-r320-a05-35814939405": "kg-04",
  "reyhanehazad/pneumonia-v1-7-phase2-m04-r384-a05-36337420014": "kg-04",
  "trickermark/pneumonia-v1-7-phase2-m05-r224-a12-36269524803": "kg-05",
  "trickermark/pneumonia-v1-7-phase2-m05-r320-a04-36370588557": "kg-05",
  "trickermark/pneumonia-v1-7-phase2-m05-r384-a05-36485988648": "kg-05",
  "msdenis/pneumonia-v1-7-phase2-m06-r224-a18-35698405831": "kg-06",
  "msdenis/pneumonia-v1-7-phase2-m06-r320-a05-35814605997": "kg-06",
  "msdenis/pneumonia-v1-7-phase2-m06-r384-a05-36371466038": "kg-06",
  "rezanory/m07-final-5fold-fix2-20260914": "kg-03",
  "trickermark/m07-runtime-r320-a13-20260917-35265801216": "kg-05",
  "trickermark/m07-runtime-r384-a12-20260920-35505225105": "kg-05",
  "nisabulutmark/pneumonia-v1-7-phase2-m08-r224-a18-35698409656": "kg-07",
  "nisabulutmark/pneumonia-v1-7-phase2-m08-r320-a05-35814941709": "kg-07",
  "nisabulutmark/pneumonia-v1-7-phase2-m08-r384-a05-36338577091": "kg-07",
  "azadkk/pneumonia-v1-7-phase2-m09-r224-a18-35698412479": "kg-08",
  "azadkk/pneumonia-v1-7-phase2-m09-r320-a05-35904860843": "kg-08",
  "azadkk/pneumonia-v1-7-phase2-m09-r384-a03-36398484836": "kg-08",
  "mylovevpn1/pneumonia-v1-7-phase2-m10-r224-a18-35698415361": "kg-09",
  "mylovevpn1/pneumonia-v1-7-phase2-m10-r320-a05-35814608153": "kg-09",
  "mylovevpn1/pneumonia-v1-7-phase2-m10-r384-a05-36354744478": "kg-09",
  "computstu1/pneumonia-v1-7-phase2-m11-r224-a11-36347200480": "kg-10",
  "computstu1/pneumonia-v1-7-phase2-m11-r320-a05-36394459369": "kg-10",
  "computstu1/pneumonia-v1-7-phase2-m11-r384-a05-36597162497": "kg-10",
  "jobreza1/pneumonia-v1-7-phase2-m12-r224-a18-35698425105": "kg-11",
  "jobreza1/pneumonia-v1-7-phase2-m12-r320-a05-35826043198": "kg-11",
  "jobreza1/pneumonia-v1-7-phase2-m12-r384-a03-36369326430": "kg-11",
};

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {"content-type":"application/json; charset=utf-8","cache-control":"no-store"},
  });
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(expected && expected.length >= 32 && request.headers.get("authorization") === `Bearer ${expected}`);
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

async function sha256(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2,"0")).join("");
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return json({service:"phase2-master-full-source",status:"ready",protected:true,kernel_count:Object.keys(KERNELS).length});
    }
    if (url.pathname !== "/control/phase2/master-full-source" || request.method !== "POST") {
      return new Response("Not found",{status:404});
    }
    if (!authorized(request,env)) return new Response("Forbidden",{status:403});
    try {
      const body = record(await request.json());
      const kernelRef = typeof body.kernel_ref === "string" ? body.kernel_ref : "";
      const accountId = KERNELS[kernelRef];
      if (!accountId) return json({ok:false,error:"kernel_ref not allowlisted"},400);
      const value = await getKernel(env,accountId,kernelRef);
      const metadata = record(value.metadata);
      const blob = record(value.blob);
      const source = typeof blob.source === "string" ? blob.source : "";
      if (!source) throw new Error("GetKernel returned no source");
      const sourceSha = await sha256(source);
      return json({
        project:PROJECT,
        purpose:PURPOSE,
        account_id:accountId,
        kernel_ref:kernelRef,
        metadata,
        source_sha256:sourceSha,
        source_bytes:new TextEncoder().encode(source).byteLength,
        blob:{source,language:blob.language ?? metadata.language ?? null,kernelType:blob.kernelType ?? metadata.kernelType ?? null},
      });
    } catch (error) {
      return json({ok:false,error:error instanceof Error ? error.message.slice(0,1000) : "unknown error"},502);
    }
  },
} satisfies ExportedHandler<Env>;
