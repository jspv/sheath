import type { ReactNode } from "react";
import { Providers } from "./providers";

export const metadata = {
  title: "sheath × CopilotKit",
  description: "CopilotKit frontend for the sheath AG-UI backend",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
