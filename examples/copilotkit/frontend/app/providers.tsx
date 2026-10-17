"use client";

import type { ReactNode } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

// Bind the global fetch to `window` before CopilotKit mounts. @ag-ui/client's HttpAgent
// (which CopilotKit constructs in the browser) calls a *detached* global fetch
// (`this.fetch(url, init)`), and Safari/WebKit rejects that with
// "Can only call Window.fetch on instances of Window" (Chrome tolerates it). Re-binding
// once makes the detached call safe in every browser. Module-level so it runs at bundle
// load, before any agent connection.
if (typeof window !== "undefined") {
  const w = window as unknown as { __fetchBound?: boolean };
  if (!w.__fetchBound) {
    window.fetch = window.fetch.bind(window);
    w.__fetchBound = true;
  }
}

export function Providers({ children }: { children: ReactNode }) {
  // `agent` matches the key registered in CopilotRuntime's `agents` map (app/api/copilotkit/route.ts).
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="sheath">
      {children}
    </CopilotKit>
  );
}
