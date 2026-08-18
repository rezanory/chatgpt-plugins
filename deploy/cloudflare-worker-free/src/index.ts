import { McpServer } from "@modelcontextprotocol/server";
import { createMcpHandler } from "agents/mcp/server";
import * as z from "zod/v4";

import { handleGitHubWebhook } from "./github";
import {
  authCheck,
  authCheckAll,
  inventoryAll,
  kernelLogs,
  kernelOutputFiles,
  kernelOutputManifest,
  kernelStatus,
  listKernels,
  masterAuthCheck,
  publicAccounts,
  publicMaster,
  type WorkerEnv,
} from "./kaggle";
import {
  launchV622Wave,
  projectControlAuthorized,
  repairV622Worker,
  v622WavePlan,
} from "./matrix-run";
import { v622FinalizationPlan } from "./project-plan";
import { v622RecoveryStatus, v622ShardArtifacts } from "./recovery";
import { v622WaveStatus } from "./wave-status";

const READ_ONLY = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true,
} as const;

const workerAccountId = z.enum(["kg-02", "kg-03", "kg-04", "kg-05", "kg-06", "kg-07"]);

function textResult(value: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(value) }],
  };
}

function buildServer(env: WorkerEnv): McpServer {
  const server = new McpServer({
    name: "Kaggle Direct Gateway",
    version: "0.1.0",
  });

  server.registerTool(
    "kaggle_accounts",
    {
      description: "List the six execution accounts without exposing credentials.",
      inputSchema: z.object({}),
      annotations: READ_ONLY,
    },
    async () => textResult(publicAccounts()),
  );

  server.registerTool(
    "kaggle_master_account",
    {
      description: "Describe the separate read-only Master Kaggle account.",
      inputSchema: z.object({}),
      annotations: READ_ONLY,
    },
    async () => textResult(publicMaster()),
  );

  server.registerTool(
    "kaggle_auth_check",
    {
      description: "Authenticate one execution account with a harmless ListKernels probe.",
      inputSchema: z.object({ account_id: workerAccountId }),
      annotations: READ_ONLY,
    },
    async ({ account_id }) => textResult(await authCheck(env, account_id)),
  );

  server.registerTool(
    "kaggle_auth_check_all",
    {
      description: "Verify all six execution accounts in parallel using direct HTTPS API calls.",
      inputSchema: z.object({ max_workers: z.number().int().min(1).max(6).default(6) }),
      annotations: READ_ONLY,
    },
    async () => textResult(await authCheckAll(env)),
  );

  server.registerTool(
    "kaggle_master_auth_check",
    {
      description: "Authenticate the separate Master account with a harmless ListKernels probe.",
      inputSchema: z.object({}),
      annotations: READ_ONLY,
    },
    async () => textResult(await masterAuthCheck(env)),
  );

  server.registerTool(
    "kaggle_kernels_list",
    {
      description: "List one execution account's kernels ordered by latest run.",
      inputSchema: z.object({
        account_id: workerAccountId,
        search: z.string().max(200).default(""),
        page_size: z.number().int().min(1).max(100).default(20),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, search, page_size }) =>
      textResult(await listKernels(env, account_id, search, page_size)),
  );

  server.registerTool(
    "kaggle_master_kernels_list",
    {
      description: "List the Master account's kernels ordered by latest run.",
      inputSchema: z.object({
        search: z.string().max(200).default(""),
        page_size: z.number().int().min(1).max(100).default(20),
      }),
      annotations: READ_ONLY,
    },
    async ({ search, page_size }) => textResult(await listKernels(env, "master", search, page_size)),
  );

  server.registerTool(
    "kaggle_kernels_inventory_all",
    {
      description: "Search every execution account without submitting new compute.",
      inputSchema: z.object({
        search: z.string().min(1).max(200),
        page_size: z.number().int().min(1).max(100).default(20),
        max_workers: z.number().int().min(1).max(6).default(6),
      }),
      annotations: READ_ONLY,
    },
    async ({ search, page_size }) => textResult(await inventoryAll(env, search, page_size)),
  );

  server.registerTool(
    "kaggle_kernel_status",
    {
      description: "Read the latest session status for an execution account's existing kernel.",
      inputSchema: z.object({
        account_id: workerAccountId,
        kernel_ref: z.string().min(3).max(200),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, kernel_ref }) =>
      textResult(await kernelStatus(env, account_id, kernel_ref)),
  );

  server.registerTool(
    "kaggle_master_kernel_status",
    {
      description: "Read the latest session status for an existing Master kernel.",
      inputSchema: z.object({ kernel_ref: z.string().min(3).max(200) }),
      annotations: READ_ONLY,
    },
    async ({ kernel_ref }) => textResult(await kernelStatus(env, "master", kernel_ref)),
  );

  server.registerTool(
    "kaggle_kernel_logs",
    {
      description: "Read the bounded tail of an execution account's existing kernel log.",
      inputSchema: z.object({
        account_id: workerAccountId,
        kernel_ref: z.string().min(3).max(200),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, kernel_ref }) =>
      textResult({ account_id, kernel_ref, log: await kernelLogs(env, account_id, kernel_ref) }),
  );

  server.registerTool(
    "kaggle_master_kernel_logs",
    {
      description: "Read the bounded tail of an existing Master kernel log.",
      inputSchema: z.object({ kernel_ref: z.string().min(3).max(200) }),
      annotations: READ_ONLY,
    },
    async ({ kernel_ref }) =>
      textResult({ account_id: "master", kernel_ref, log: await kernelLogs(env, "master", kernel_ref) }),
  );

  server.registerTool(
    "kaggle_kernel_output_files",
    {
      description:
        "Enumerate paginated output filenames for an existing execution-account kernel without downloading them.",
      inputSchema: z.object({
        account_id: workerAccountId,
        kernel_ref: z.string().min(3).max(200),
        contains: z.string().max(200).default(""),
        max_files: z.number().int().min(1).max(2000).default(1000),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, kernel_ref, contains, max_files }) =>
      textResult(await kernelOutputFiles(env, account_id, kernel_ref, contains, max_files)),
  );

  server.registerTool(
    "kaggle_master_kernel_output_files",
    {
      description:
        "Enumerate paginated output filenames for an existing Master kernel without downloading them.",
      inputSchema: z.object({
        kernel_ref: z.string().min(3).max(200),
        contains: z.string().max(200).default(""),
        max_files: z.number().int().min(1).max(2000).default(1000),
      }),
      annotations: READ_ONLY,
    },
    async ({ kernel_ref, contains, max_files }) =>
      textResult(await kernelOutputFiles(env, "master", kernel_ref, contains, max_files)),
  );

  server.registerTool(
    "kaggle_kernel_output_manifest",
    {
      description:
        "Hash a small allowlisted set of an execution account's existing output files and scan for a fingerprint.",
      inputSchema: z.object({
        account_id: workerAccountId,
        kernel_ref: z.string().min(3).max(200),
        artifact_names: z.array(z.string().min(1).max(160)).min(1).max(8),
        expected_fingerprint: z.string().max(256).default(""),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, kernel_ref, artifact_names, expected_fingerprint }) =>
      textResult(
        await kernelOutputManifest(
          env,
          account_id,
          kernel_ref,
          artifact_names,
          expected_fingerprint,
        ),
      ),
  );

  server.registerTool(
    "kaggle_master_kernel_output_manifest",
    {
      description: "Hash selected existing Master output files and scan them for a fingerprint.",
      inputSchema: z.object({
        kernel_ref: z.string().min(3).max(200),
        artifact_names: z.array(z.string().min(1).max(160)).min(1).max(8),
        expected_fingerprint: z.string().max(256).default(""),
      }),
      annotations: READ_ONLY,
    },
    async ({ kernel_ref, artifact_names, expected_fingerprint }) =>
      textResult(
        await kernelOutputManifest(
          env,
          "master",
          kernel_ref,
          artifact_names,
          expected_fingerprint,
        ),
      ),
  );

  return server;
}

function privateMcpPath(env: WorkerEnv): string | null {
  const token = env.CGP_MCP_PATH_TOKEN?.trim();
  if (!token) return null;
  if (!/^[A-Za-z0-9_-]{32,128}$/.test(token)) return null;
  return `/mcp/${token}`;
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function errorResponse(error: unknown): Response {
  return json(
    {
      ok: false,
      error_type: error instanceof Error ? error.name : "Error",
      error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
    },
    502,
  );
}

export default {
  async fetch(request: Request, env: WorkerEnv, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return json({
        service: "chatgpt-kaggle-gateway",
        transport: "cloudflare-workers-free",
        status: "ready",
        mcp_configured: Boolean(privateMcpPath(env)),
        kaggle_tokens_configured: [
          env.CGP_KAGGLE_KG02_TOKEN,
          env.CGP_KAGGLE_KG03_TOKEN,
          env.CGP_KAGGLE_KG04_TOKEN,
          env.CGP_KAGGLE_KG05_TOKEN,
          env.CGP_KAGGLE_KG06_TOKEN,
          env.CGP_KAGGLE_KG07_TOKEN,
        ].filter((value) => Boolean(value?.trim())).length,
        master_configured: Boolean(env.CGP_KAGGLE_MASTER_TOKEN?.trim()),
        write_enabled: env.CGP_WRITE_ENABLED === "1",
      });
    }

    if (url.pathname === "/recovery/v6-2-2/status" && request.method === "GET") {
      try {
        return json(await v622RecoveryStatus(env));
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/recovery/v6-2-2/wave-status" && request.method === "GET") {
      try {
        const wave = Number(url.searchParams.get("wave") ?? "2");
        return json(await v622WaveStatus(env, wave));
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/recovery/v6-2-2/shard-artifacts" && request.method === "GET") {
      try {
        return json(await v622ShardArtifacts(env, url.searchParams.get("shard") ?? ""));
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/recovery/v6-2-2/finalization-plan" && request.method === "GET") {
      try {
        return json(await v622FinalizationPlan(env));
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/recovery/v6-2-2/wave-plan" && request.method === "GET") {
      try {
        const wave = Number(url.searchParams.get("wave") ?? "2");
        return json({ wave, tasks: v622WavePlan(wave) });
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/control/v6-2-2/wave" && request.method === "POST") {
      if (!projectControlAuthorized(request, env)) return new Response("Forbidden", { status: 403 });
      try {
        const body: unknown = await request.json();
        const value = body && typeof body === "object" && !Array.isArray(body)
          ? body as Record<string, unknown>
          : {};
        return json(await launchV622Wave(env, Number(value.wave)));
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/control/v6-2-2/repair" && request.method === "POST") {
      if (!projectControlAuthorized(request, env)) return new Response("Forbidden", { status: 403 });
      try {
        const body: unknown = await request.json();
        const value = body && typeof body === "object" && !Array.isArray(body)
          ? body as Record<string, unknown>
          : {};
        const workerId = typeof value.worker_id === "string" ? value.worker_id : "";
        return json(await repairV622Worker(env, workerId));
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (url.pathname === "/github/webhook" && request.method === "POST") {
      return handleGitHubWebhook(request, env, ctx);
    }

    const mcpPath = privateMcpPath(env);
    const adminRoot = mcpPath ? `${mcpPath}/admin` : null;
    if (adminRoot && request.method === "GET") {
      try {
        if (url.pathname === `${adminRoot}/auth`) {
          return json({ workers: await authCheckAll(env), master: await masterAuthCheck(env) });
        }
        if (url.pathname === `${adminRoot}/list`) {
          const accountId = url.searchParams.get("account_id") ?? "";
          const search = url.searchParams.get("search") ?? "";
          const pageSize = Number(url.searchParams.get("page_size") ?? "5");
          return json({
            account_id: accountId,
            kernels: await listKernels(env, accountId, search, pageSize),
          });
        }
        if (url.pathname === `${adminRoot}/inventory`) {
          const search = url.searchParams.get("search") ?? "";
          const pageSize = Number(url.searchParams.get("page_size") ?? "20");
          return json({ search, accounts: await inventoryAll(env, search, pageSize) });
        }
        if (url.pathname === `${adminRoot}/kernel-status`) {
          const accountId = url.searchParams.get("account_id") ?? "";
          const kernelRef = url.searchParams.get("kernel_ref") ?? "";
          return json(await kernelStatus(env, accountId, kernelRef));
        }
        if (url.pathname === `${adminRoot}/kernel-logs`) {
          const accountId = url.searchParams.get("account_id") ?? "";
          const kernelRef = url.searchParams.get("kernel_ref") ?? "";
          return json({
            account_id: accountId,
            kernel_ref: kernelRef,
            log: await kernelLogs(env, accountId, kernelRef),
          });
        }
        if (url.pathname === `${adminRoot}/output-files`) {
          const accountId = url.searchParams.get("account_id") ?? "";
          const kernelRef = url.searchParams.get("kernel_ref") ?? "";
          const contains = url.searchParams.get("contains") ?? "";
          const maxFiles = Number(url.searchParams.get("max_files") ?? "1000");
          return json(await kernelOutputFiles(env, accountId, kernelRef, contains, maxFiles));
        }
        if (url.pathname === `${adminRoot}/output-manifest`) {
          const accountId = url.searchParams.get("account_id") ?? "";
          const kernelRef = url.searchParams.get("kernel_ref") ?? "";
          const artifactNames = url.searchParams.getAll("artifact_name");
          const expectedFingerprint = url.searchParams.get("expected_fingerprint") ?? "";
          return json(
            await kernelOutputManifest(
              env,
              accountId,
              kernelRef,
              artifactNames,
              expectedFingerprint,
            ),
          );
        }
      } catch (error) {
        return errorResponse(error);
      }
    }

    if (!mcpPath || url.pathname !== mcpPath) return new Response("Not found", { status: 404 });

    const handler = createMcpHandler(() => buildServer(env));
    return handler(request, env, ctx);
  },
} satisfies ExportedHandler<WorkerEnv>;
