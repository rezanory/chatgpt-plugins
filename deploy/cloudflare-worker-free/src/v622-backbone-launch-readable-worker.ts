import launchWorker from "./v622-backbone-launch-worker";

const delegated = launchWorker as unknown as {
  fetch(request: Request, env: unknown): Promise<Response>;
};

export default {
  async fetch(request: Request, env: unknown): Promise<Response> {
    const response = await delegated.fetch(request, env);
    const body = await response.text();
    let payload: Record<string, unknown> | null = null;
    try {
      const parsed: unknown = body ? JSON.parse(body) : null;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) payload = parsed as Record<string, unknown>;
    } catch { /* preserve original body */ }
    if (payload && payload.action === "blocked") {
      const error = typeof payload.error === "string" ? payload.error.slice(0, 900) : "unknown guarded block";
      payload.action = `blocked:${error}`;
      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
      });
    }
    const headers = new Headers(response.headers);
    headers.set("cache-control", "no-store");
    headers.set("x-v622-original-status", String(response.status));
    return new Response(body, {
      status: response.status === 502 ? 200 : response.status,
      headers,
    });
  },
};
