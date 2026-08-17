import { Container, getContainer } from "@cloudflare/containers";
import { createRemoteJWKSet, jwtVerify } from "jose";

interface Env {
  KAGGLE_GATEWAY: DurableObjectNamespace<KaggleGatewayContainer>;

  TEAM_DOMAIN: string;
  POLICY_AUD: string;

  CGP_KAGGLE_KG01_USERNAME: string;
  CGP_KAGGLE_KG01_TOKEN: string;
  CGP_KAGGLE_KG02_USERNAME: string;
  CGP_KAGGLE_KG02_TOKEN: string;
  CGP_KAGGLE_KG04_USERNAME: string;
  CGP_KAGGLE_KG04_TOKEN: string;
  CGP_KAGGLE_KG05_USERNAME: string;
  CGP_KAGGLE_KG05_TOKEN: string;
  CGP_KAGGLE_KG06_USERNAME: string;
  CGP_KAGGLE_KG06_TOKEN: string;
  CGP_KAGGLE_KG07_USERNAME: string;
  CGP_KAGGLE_KG07_TOKEN: string;
}

const jwksByIssuer = new Map<string, ReturnType<typeof createRemoteJWKSet>>();

function normalizedTeamDomain(value: string): string {
  return value.replace(/\/+$/, "");
}

function jwksFor(teamDomain: string) {
  let jwks = jwksByIssuer.get(teamDomain);
  if (!jwks) {
    jwks = createRemoteJWKSet(new URL(`${teamDomain}/cdn-cgi/access/certs`));
    jwksByIssuer.set(teamDomain, jwks);
  }
  return jwks;
}

async function requireAccess(request: Request, env: Env): Promise<Response | null> {
  if (!env.TEAM_DOMAIN || !env.POLICY_AUD) {
    return new Response("Cloudflare Access is not configured.", { status: 503 });
  }

  const token = request.headers.get("cf-access-jwt-assertion");
  if (!token) {
    return new Response("Missing Cloudflare Access JWT.", { status: 401 });
  }

  const issuer = normalizedTeamDomain(env.TEAM_DOMAIN);
  try {
    await jwtVerify(token, jwksFor(issuer), {
      issuer,
      audience: env.POLICY_AUD,
    });
    return null;
  } catch {
    return new Response("Invalid Cloudflare Access JWT.", { status: 403 });
  }
}

export class KaggleGatewayContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "30m";
  enableInternet = true;

  envVars = {
    CGP_GATEWAY_HOST: "0.0.0.0",
    CGP_GATEWAY_PORT: "8080",
    CGP_GATEWAY_TRUSTED_PROXY: "cloudflare-container",

    CGP_KAGGLE_KG01_USERNAME: this.env.CGP_KAGGLE_KG01_USERNAME,
    CGP_KAGGLE_KG01_TOKEN: this.env.CGP_KAGGLE_KG01_TOKEN,
    CGP_KAGGLE_KG02_USERNAME: this.env.CGP_KAGGLE_KG02_USERNAME,
    CGP_KAGGLE_KG02_TOKEN: this.env.CGP_KAGGLE_KG02_TOKEN,
    CGP_KAGGLE_KG04_USERNAME: this.env.CGP_KAGGLE_KG04_USERNAME,
    CGP_KAGGLE_KG04_TOKEN: this.env.CGP_KAGGLE_KG04_TOKEN,
    CGP_KAGGLE_KG05_USERNAME: this.env.CGP_KAGGLE_KG05_USERNAME,
    CGP_KAGGLE_KG05_TOKEN: this.env.CGP_KAGGLE_KG05_TOKEN,
    CGP_KAGGLE_KG06_USERNAME: this.env.CGP_KAGGLE_KG06_USERNAME,
    CGP_KAGGLE_KG06_TOKEN: this.env.CGP_KAGGLE_KG06_TOKEN,
    CGP_KAGGLE_KG07_USERNAME: this.env.CGP_KAGGLE_KG07_USERNAME,
    CGP_KAGGLE_KG07_TOKEN: this.env.CGP_KAGGLE_KG07_TOKEN,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/healthz") {
      return Response.json({
        service: "chatgpt-kaggle-gateway",
        transport: "cloudflare-container",
        status: "ready",
      });
    }

    if (!url.pathname.startsWith("/mcp")) {
      return new Response("Not found", { status: 404 });
    }

    const denied = await requireAccess(request, env);
    if (denied) {
      return denied;
    }

    const forwarded = new Request(request);
    forwarded.headers.delete("cf-access-jwt-assertion");
    forwarded.headers.delete("cookie");

    return getContainer(env.KAGGLE_GATEWAY, "primary").fetch(forwarded);
  },
};
