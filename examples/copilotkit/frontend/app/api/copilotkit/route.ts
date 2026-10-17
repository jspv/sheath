// CopilotKit runtime route (Next.js App Router).
//
// The browser calls this same-origin endpoint; the runtime forwards each request
// to the sheath AG-UI backend via an HttpAgent. Because the proxy runs here on
// the server, no CORS configuration is needed on the Python backend.
//
// ExperimentalEmptyAdapter is the no-op service adapter: the sheath agent does
// its own LLM calls, so the CopilotKit runtime needs no model service of its own.

import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";

import { HttpAgent } from "@ag-ui/client";

import { NextRequest } from "next/server";

const serviceAdapter = new ExperimentalEmptyAdapter();

const runtime = new CopilotRuntime({
  agents: {
    // This key must match the `agent` prop on <CopilotKit> in app/layout.tsx.
    sheath: new HttpAgent({
      url: process.env.SHEATH_AGENT_URL ?? "http://localhost:8000/agent",
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
