import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TraceMind | SRE Incident RCA Console",
  description: "Automated Log Anomaly Clustering & Retrieval-Augmented Postmortem Correlation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} dark antialiased`}>
      <body className="min-h-screen flex flex-col bg-[#0A0A0B] text-[#EDEDEF] font-sans selection:bg-[#6E7AE8]/30 selection:text-[#FFFFFF]">
        <TooltipProvider delay={200}>
          {children}
          <Toaster
            theme="dark"
            position="bottom-right"
            toastOptions={{
              style: {
                background: "#17171A",
                border: "1px solid #26262B",
                color: "#EDEDEF",
                fontSize: "12px",
                borderRadius: "6px",
              },
            }}
          />
        </TooltipProvider>
      </body>
    </html>
  );
}
