// CSS is bundled by Next.js; TypeScript has no module declarations for this side-effect import.
// @ts-ignore
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Vehicle Tracking & ReID System",
  description: "Real-time traffic flow classification and wrong-way violation dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-900 text-slate-100 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}