import launchWorker from "./v622-backbone-launch-worker";

const delegated = launchWorker as unknown as {
  fetch(request: Request, env: unknown): Promise<Response>;
};

export default {
  async fetch(request: Request, env: unknown): Promise<Response> {
    const response = await delegated.fetch(request, env);
    const body = await response.text();
    const headers = new Headers(response.headers);
    headers.set("cache-control", "no-store");
    headers.set("x-v622-original-status", String(response.status));
    return new Response(body, {
      status: response.status === 502 ? 200 : response.status,
      headers,
    });
  },
};
