"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";

export default function Home() {
  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "3rem", maxWidth: 720 }}>
      <h1>sheath × CopilotKit</h1>
      <p>
        The chat sidebar on the right talks to the sheath AG-UI backend, which has a
        <code> sales </code> MCP server wired in. Try asking:
      </p>
      <ul>
        <li>“What was total EU revenue in 2025, excluding invalid rows?”</li>
        <li>“Compare EU vs NA revenue for 2025 and show the gap.”</li>
        <li>“Which region had the most invalid rows?”</li>
      </ul>
      <p>
        Behind the scenes the agent calls the MCP tool, the large result is spilled to a
        typed handle, and the agent writes sandboxed Python to compute the answer — you
        see its progress stream in the chat.
      </p>

      <CopilotSidebar
        defaultOpen
        labels={{ title: "sheath assistant", initial: "Ask me about 2025 sales 👋" }}
      />
    </main>
  );
}
