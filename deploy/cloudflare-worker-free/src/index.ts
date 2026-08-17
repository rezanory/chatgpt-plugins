import { McpServer } from "@modelcontextprotocol/server";
import { createMcpHandler } from "agents/mcp/server";
import * as z from "zod/v4";

import { handleGitHubWebhook } from "./github";
import {
  authCheck,
  authCheckAll,
  inventoryAll,
  kernelLogs,
  kernelOutputManifest,
  kernelStatus,
  listKernels,
  publicAccounts,
  type WorkerEnv,
} from "./kaggle";

const READ_ONLY = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true,
} as const;

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
      description: "List configured logical Kaggle accounts without exposing credentials.",
      inputSchema: z.object({}),
      annotations: READ_ONLY,
    },
    async () => textResult(publicAccounts()),
  );

  server.registerTool(
    "kaggle_auth_check",
    {
      description: "Authenticate one configured Kaggle account with a harmless ListKernels probe.",
      inputSchema: z.object({ account_id: z.string().min(1).max(20) }),
      annotations: READ_ONLY,
    },
    async ({ account_id }) => textResult(await authCheck(env, account_id)),
  );

  server.registerTool(
    "kaggle_auth_check_all",
    {
      description: "Verify all six enabled Kaggle accounts in parallel using direct HTTPS API calls.",
      inputSchema: z.object({ max_workers: z.number().int().min(1).max(6).default(6) }),
      annotations: READ_ONLY,
    },
    async () => textResult(await authCheckAll(env)),
  );

  server.registerTool(
    "kaggle_kernels_list",
    {
      description: "List the selected account's own Kaggle kernels ordered by latest run.",
      inputSchema: z.object({
        account_id: z.string().min(1).max(20),
        search: z.string().max(200).default(""),
        page_size: z.number().int().min(1).max(100).default(20),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, search, page_size }) =>
      textResult(await listKernels(env, account_id, search, page_size)),
  );

  server.registerTool(
    "kaggle_kernels_inventory_all",
    {
      description: "Search every enabled account's own kernels without submitting new compute.",
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
      description: "Read the latest session status for an existing owner/kernel.",
      inputSchema: z.object({
        account_id: z.string().min(1).max(20),
        kernel_ref: z.string().min(3).max(200),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, kernel_ref }) =>
      textResult(await kernelStatus(env, account_id, kernel_ref)),
  );

  server.registerTool(
    "kaggle_kernel_logs",
    {
      description: "Read the latest execution log for an existing owner/kernel.",
      inputSchema: z.object({
        account_id: z.string().min(1).max(20),
        kernel_ref: z.string().min(3).max(200),
      }),
      annotations: READ_ONLY,
    },
    async ({ account_id, kernel_ref }) =>
      textResult({ account_id, kernel_ref, log: await kernelLogs(env, account_id, kernel_ref) }),
  );

  server.registerTool(
    "kaggle_kernel_output_manifest",
    {
      description:
        "Hash a small allowlisted set of existing output files and scan them for an expected fingerprint.",
      inputSchema: z.object({
        account_id: z.string().min(1).max(20),
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
          env.CGP_KAGGLE_KG01_TOKEN,
          env.CGP_KAGGLE_KG02_TOKEN,
          env.CGP_KAGGLE_KG04_TOKEN,
          env.CGP_KAGGLE_KG05_TOKEN,
          env.CGP_KAGGLE_KG06_TOKEN,
          env.CGP_KAGGLE_KG07_TOKEN,
        ].filter((value) => Boolean(value?.trim())).length,
        write_enabled: env.CGP_WRITE_ENABLED === "1",
      });
    }

    if (url.pathname === "/github/webhook" && request.method === "POST") {
      return handleGitHubWebhook(request, env, ctx);
    }

    const mcpPath = privateMcpPath(env);
    if (!mcpPath || url.pathname !== mcpPath) return new Response("Not found", { status: 404 });

    const handler = createMcpHandler(() => buildServer(env));
    return handler(request, env, ctx);
  },
} satisfies ExportedHandler<WorkerEnv>;
