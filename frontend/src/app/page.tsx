"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { toast } from "sonner";
import { useReducedMotion, motion } from "motion/react";
import {
  AlertCircle,
  Check,
  ChevronDown,
  Copy,
  ExternalLink,
  GitBranch,
  Layers,
  Loader2,
  RotateCcw,
  Terminal,
  Database,
  Cpu,
  Clock,
  ListChecks,
  CheckSquare,
  Square,
  Search,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { ThemeToggle } from "@/components/theme-toggle";

// --- Types matching Backend Models strictly ---

type RetrievedIncident = {
  incident_id: string;
  raw_log_excerpt: string;
  root_cause: string;
  remediation_steps: string;
  source_url: string;
  quality_score?: number;
  quality?: string | number;
};

type PipelineTrace = {
  cluster_id: number;
  is_noise: boolean;
  retrieved_incident: RetrievedIncident;
  similarity_score: number;
  retrieval_pass?: number;
  prompt_used: string;
  model?: string;
};

type AnalyzeResponse = {
  incident_id: string;
  raw_log_excerpt: string;
  root_cause: string;
  remediation_steps: string;
  source_url: string;
  retrieval_pass?: number;
  trace: PipelineTrace;
  is_mock?: boolean;
};

interface ApiError {
  message: string;
  statusCode?: number;
}

// --- Presets ---

const PRESET_EXAMPLES = [
  {
    title: "DB pool exhausted",
    log: `[2026-10-15 08:23:41] [ERROR] [db-proxy-pool] Connection timeout acquiring connection from pool.
FATAL: remaining connection slots are reserved for non-replication superuser connections.
PoolExhaustedError: database connection pool exhausted, 50/50 connections in use
[2026-10-15 08:23:42] [WARN] [health-check] Liveness probe failed for pod db-proxy-6b8f7d9`,
  },
  {
    title: "Kubelet OOMKilled",
    log: `CRITICAL 2026-10-14T14:12:05Z kubelet worker-node-04: OOMKilled process 18421 (payments-api) total-vm:8421048kB, anon-rss:4194304kB. Memory limit 4Gi exceeded.`,
  },
  {
    title: "Redis sentinel failover",
    log: `ERROR: Redis sentinel failover triggered - master 10.0.4.12:6379 unreachable for 30s. Failover elected replica 10.0.4.13:6379 as new master. Write latency spiked to 2400ms.`,
  },
  {
    title: "SSL verify failed",
    log: `WARNING: SSL certificate verification failed for api.stripe.com. OpenSSL Error: certificate has expired on 2026-10-14 23:59:59 GMT. TLS handshake terminated.`,
  },
];

const PIPELINE_STAGES = [
  {
    name: "Embed",
    title: "Embed log vector",
    detail: "all-MiniLM-L6-v2, 384-d",
  },
  {
    name: "Cluster",
    title: "Cluster anomaly",
    detail: "HDBSCAN, 48 clusters",
  },
  {
    name: "Retrieve",
    title: "Retrieve incident",
    detail: "ChromaDB, quality-biased",
  },
  {
    name: "Generate",
    title: "Generate RCA",
    detail: "openai/gpt-oss-120b",
  },
];

function extractCompany(url: string): string {
  try {
    const hostname = new URL(url).hostname.replace(/^www\./, "");
    if (hostname.includes("github")) return "GitHub";
    if (hostname.includes("cloudflare")) return "Cloudflare";
    if (hostname.includes("slack")) return "Slack";
    if (hostname.includes("aws") || hostname.includes("amazon")) return "AWS";
    if (hostname.includes("google") || hostname.includes("gcp")) return "Google";
    if (hostname.includes("stripe")) return "Stripe";
    if (hostname.includes("gitlab")) return "GitLab";
    if (hostname.includes("heroku")) return "Heroku";
    if (hostname.includes("statuspage")) return hostname.split(".")[0];
    return hostname.split(".")[0] || "Unknown";
  } catch {
    return "Unknown";
  }
}

function parseRemediationSteps(raw: string): string[] {
  if (!raw) return [];
  return raw
    .split(/\n+/)
    .map((s) => s.trim().replace(/^[\d+.\-*]+\s*/, ""))
    .filter(Boolean);
}

function highlightTokens(text: string) {
  const parts = text.split(/(ERROR|FATAL|WARN(?:ING)?|CRITICAL)/gi);
  return parts.map((part, i) => {
    const upper = part.toUpperCase();
    if (upper === "ERROR" || upper === "FATAL" || upper === "CRITICAL") {
      return (
        <span key={i} className="text-[var(--danger)] font-semibold">
          {part}
        </span>
      );
    }
    if (upper === "WARN" || upper === "WARNING") {
      return (
        <span key={i} className="text-[var(--warn)] font-semibold">
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function SimilarityGauge({ value }: { value: number }) {
  const clampVal = Math.max(0, Math.min(1, value));
  const r = 38;
  const cx = 50;
  const cy = 46;
  const arcLength = Math.PI * r;
  const offset = arcLength * (1 - clampVal);

  const tickAngle = Math.PI * (1 - 0.4);
  const tickInnerR = r - 5;
  const tickOuterR = r + 5;
  const x1 = cx + tickInnerR * Math.cos(tickAngle);
  const y1 = cy - tickInnerR * Math.sin(tickAngle);
  const x2 = cx + tickOuterR * Math.cos(tickAngle);
  const y2 = cy - tickOuterR * Math.sin(tickAngle);

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg width="100" height="54" viewBox="0 0 100 54" className="overflow-visible" aria-hidden="true">
        {/* Track */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="var(--border)"
          strokeWidth="6"
          strokeLinecap="round"
        />
        {/* Progress */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={arcLength}
          strokeDashoffset={offset}
          className="transition-all duration-500 ease-out"
        />
        {/* 0.40 Threshold tick */}
        <line
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="var(--muted)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute top-4 flex flex-col items-center">
        <span className="font-mono text-[16px] font-semibold tabular-nums text-[var(--text)]">
          {(clampVal * 100).toFixed(0)}%
        </span>
        <span className="text-[10px] text-[var(--muted)] font-mono uppercase tracking-wider">
          score
        </span>
      </div>
    </div>
  );
}

export default function Home() {
  const [logText, setLogText] = useState(PRESET_EXAMPLES[0].log);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [isColdStart, setIsColdStart] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [timestamp, setTimestamp] = useState<string | null>(null);
  const [checkedSteps, setCheckedSteps] = useState<Record<number, boolean>>({});

  // Health check state
  const [apiStatus, setApiStatus] = useState<"ok" | "waking" | "error">("ok");
  const [lastChecked, setLastChecked] = useState<string>("Initializing");

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  const shouldReduceMotion = useReducedMotion();

  const rawBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  const baseUrl = (() => {
    let url = rawBaseUrl.trim().replace(/\/+$/, "");
    if (url.includes(".hf.space") && !url.includes("/gradio_api")) {
      url = `${url}/gradio_api`;
    }
    return url;
  })();

  // --- Health Check Polling ---
  useEffect(() => {
    let isMounted = true;

    const poll = async () => {
      const t0 = performance.now();
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000);

        const res = await fetch(`${baseUrl}/health`, { signal: controller.signal });
        clearTimeout(timeoutId);
        const elapsed = performance.now() - t0;

        if (!isMounted) return;
        const nowStr = new Date().toLocaleTimeString([], { hour12: false });
        setLastChecked(nowStr);

        if (res.ok) {
          if (elapsed > 3000) {
            setApiStatus("waking");
          } else {
            setApiStatus("ok");
          }
        } else {
          setApiStatus("error");
        }
      } catch {
        if (!isMounted) return;
        setApiStatus("error");
        setLastChecked(new Date().toLocaleTimeString([], { hour12: false }));
      }
    };

    poll();
    const interval = setInterval(poll, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [baseUrl]);

  // --- Synchronized Editor Scroll ---
  const handleScroll = () => {
    if (textareaRef.current) {
      const { scrollTop, scrollLeft } = textareaRef.current;
      if (highlightRef.current) {
        highlightRef.current.scrollTop = scrollTop;
        highlightRef.current.scrollLeft = scrollLeft;
      }
      if (gutterRef.current) {
        gutterRef.current.scrollTop = scrollTop;
      }
    }
  };

  // --- Submission Logic ---
  const handleSubmit = useCallback(async () => {
    if (!logText.trim() || loading) return;

    setLoading(true);
    setIsColdStart(false);
    setError(null);
    setResult(null);
    setCheckedSteps({});
    setLoadingStage(0);

    // Timed simulated stage progression
    const stageTimer1 = setTimeout(() => setLoadingStage(1), 400);
    const stageTimer2 = setTimeout(() => setLoadingStage(2), 1000);
    const stageTimer3 = setTimeout(() => setLoadingStage(3), 1800);

    // Cold-start warning timer if backend takes >3s
    const coldStartTimer = setTimeout(() => {
      setIsColdStart(true);
    }, 3000);

    const startTime = performance.now();

    try {
      const res = await fetch(`${baseUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: logText }),
      });

      const elapsed = Math.round(performance.now() - startTime);
      setLatencyMs(elapsed);
      setTimestamp(new Date().toLocaleTimeString([], { hour12: false }));

      if (!res.ok) {
        let errMessage = `HTTP ${res.status} ${res.statusText}`;
        try {
          const errData = await res.json();
          if (errData.message) {
            errMessage = errData.message;
          } else if (errData.detail) {
            errMessage = typeof errData.detail === "string" ? errData.detail : JSON.stringify(errData.detail);
          }
        } catch {
          // Keep standard status message
        }
        setError({ message: errMessage, statusCode: res.status });
        return;
      }

      const data: AnalyzeResponse = await res.json();
      setResult(data);
      setLoadingStage(4);
    } catch (err: unknown) {
      const elapsed = Math.round(performance.now() - startTime);
      setLatencyMs(elapsed);
      setError({
        message: err instanceof Error ? err.message : "Failed to connect to backend",
      });
    } finally {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      clearTimeout(stageTimer3);
      clearTimeout(coldStartTimer);
      setLoading(false);
    }
  }, [baseUrl, loading, logText]);

  // Keyboard shortcut: Ctrl+Enter or Cmd+Enter to submit
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleSubmit]);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast.success(`${label} copied`);
  };

  const lineCount = logText.split("\n").length;
  const charCount = logText.length;
  const remediationList = result ? parseRemediationSteps(result.remediation_steps) : [];
  const completedStepsCount = Object.values(checkedSteps).filter(Boolean).length;

  const tileVariants = {
    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 6 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        duration: shouldReduceMotion ? 0.01 : 0.18,
        ease: "easeOut" as const,
        delay: shouldReduceMotion ? 0 : i * 0.04,
      },
    }),
  };

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--text)]">
      {/* Top Bar (56px, no tile, bottom border) */}
      <header className="h-14 border-b border-[var(--border)] bg-transparent px-4 sm:px-6 flex items-center justify-between z-10">
        {/* Left: Wordmark + Geometric Logo */}
        <div className="flex items-center gap-2.5">
          <svg
            width="18"
            height="18"
            viewBox="0 0 18 18"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="text-[var(--text)] flex-shrink-0"
            aria-hidden="true"
          >
            <rect x="2" y="2" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
            <rect x="10" y="2" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
            <rect x="2" y="10" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="13" cy="13" r="2.5" fill="currentColor" />
            <line x1="8" y1="5" x2="10" y2="5" stroke="currentColor" strokeWidth="1.5" />
            <line x1="5" y1="8" x2="5" y2="10" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          <span className="font-semibold text-[15px] tracking-tight">TraceMind</span>
        </div>

        {/* Right: API Status, Theme Toggle, GitHub */}
        <div className="flex items-center gap-3">
          {/* API Status Cluster */}
          <Tooltip>
            <TooltipTrigger className="flex items-center gap-2 px-2.5 py-1 rounded-lg border border-[var(--border)] bg-[var(--tile)] text-[12px] cursor-default">
              {apiStatus === "ok" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-[var(--ok)]" />
                  <span className="text-[var(--muted)] font-medium">Operational</span>
                </>
              )}
              {apiStatus === "waking" && (
                <>
                  <Loader2 className="h-3 w-3 animate-spin text-[var(--warn)]" />
                  <span className="text-[var(--warn)] font-medium">Waking up</span>
                </>
              )}
              {apiStatus === "error" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-[var(--danger)]" />
                  <span className="text-[var(--danger)] font-medium">Down</span>
                </>
              )}
            </TooltipTrigger>
            <TooltipContent side="bottom" className="text-[12px] font-mono">
              Last checked: {lastChecked}
            </TooltipContent>
          </Tooltip>

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* GitHub link */}
          <a
            href="https://github.com/nish-debug15/TraceMind"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub Repository"
            className="flex items-center justify-center h-8 w-8 rounded-lg text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] border border-transparent hover:border-[var(--border)] transition-colors"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
              <path d="M9 18c-4.51 2-5-2-7-2" />
            </svg>
          </a>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1320px] w-full mx-auto p-4 sm:p-6 flex flex-col gap-4">
        {/* Error Banner (full-width, danger-bordered) */}
        {error && (
          <div
            role="alert"
            aria-live="assertive"
            className="w-full flex items-center justify-between gap-4 p-4 rounded-[12px] border border-[var(--danger)] bg-[var(--danger)]/5 text-[var(--text)] text-[13px]"
          >
            <div className="flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-[var(--danger)] flex-shrink-0" />
              <div>
                <span className="font-semibold text-[var(--danger)]">Pipeline Error: </span>
                <span>{error.message}</span>
                {error.statusCode && (
                  <Badge variant="outline" className="ml-2 font-mono text-[11px] border-[var(--danger)]/40 text-[var(--danger)]">
                    HTTP {error.statusCode}
                  </Badge>
                )}
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleSubmit}
              className="border-[var(--danger)]/40 text-[var(--danger)] hover:bg-[var(--danger)]/10 h-8 rounded-lg"
            >
              <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
              Retry
            </Button>
          </div>
        )}

        {/* Bento Grid: 12 cols desktop, 6 tablet, 1 mobile */}
        <div className="grid grid-cols-1 md:grid-cols-6 xl:grid-cols-12 gap-4">
          {/* ========================================================= */}
          {/* ROW 1 - TILE A: Incident Log (col-span 7) */}
          {/* ========================================================= */}
          <motion.div
            custom={0}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className="order-1 md:order-1 col-span-1 md:col-span-3 xl:col-span-7 flex flex-col rounded-[16px] border border-[var(--border)] bg-[var(--tile)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none min-h-[460px] overflow-hidden"
          >
            {/* Tile Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Incident Log
                </span>
              </div>
              <div className="flex items-center gap-2">
                {/* Example Dropdown */}
                <DropdownMenu>
                  <DropdownMenuTrigger className="inline-flex items-center justify-center h-7 text-[12px] font-medium text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] rounded-lg px-2 border border-transparent transition-colors">
                    Load example
                    <ChevronDown className="h-3.5 w-3.5 ml-1 opacity-70" />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56 bg-[var(--tile)] border-[var(--border)]">
                    {PRESET_EXAMPLES.map((ex) => (
                      <DropdownMenuItem
                        key={ex.title}
                        onClick={() => {
                          setLogText(ex.log);
                          setResult(null);
                          setError(null);
                        }}
                        className="text-[12px] cursor-pointer hover:bg-[var(--tile-2)]"
                      >
                        {ex.title}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>

                {/* Clear */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setLogText("");
                    setResult(null);
                    setError(null);
                  }}
                  className="h-7 text-[12px] text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] rounded-lg px-2"
                >
                  Clear
                </Button>
              </div>
            </div>

            {/* Editor Body with Gutter */}
            <div className="relative flex-1 flex min-h-[260px] bg-[var(--tile-2)]/50 font-mono text-[13px] overflow-hidden">
              {/* Line number gutter */}
              <div
                ref={gutterRef}
                className="w-10 select-none py-3 text-right pr-3 text-[var(--muted)]/50 border-r border-[var(--border)] overflow-hidden font-mono text-[12px] leading-[22px] flex-shrink-0"
                aria-hidden="true"
              >
                {Array.from({ length: Math.max(lineCount, 12) }).map((_, i) => (
                  <div key={i}>{i + 1}</div>
                ))}
              </div>

              {/* Textarea & Token Highlight container */}
              <div className="relative flex-1 overflow-hidden">
                {/* Syntax Highlight Overlay */}
                <div
                  ref={highlightRef}
                  aria-hidden="true"
                  className="absolute inset-0 p-3 overflow-hidden pointer-events-none whitespace-pre-wrap break-words leading-[22px]"
                >
                  {highlightTokens(logText)}
                </div>

                {/* Actual Editor Textarea */}
                <textarea
                  ref={textareaRef}
                  value={logText}
                  onChange={(e) => setLogText(e.target.value)}
                  onScroll={handleScroll}
                  placeholder="Paste production raw log excerpt or stack trace here..."
                  aria-label="Incident log text"
                  spellCheck={false}
                  className="relative z-10 w-full h-full p-3 bg-transparent text-transparent caret-[var(--accent)] resize-none outline-none whitespace-pre-wrap break-words leading-[22px] font-mono text-[13px] selection:bg-[var(--accent-soft)] selection:text-[var(--accent)]"
                />
              </div>
            </div>

            {/* Preset chips inside Tile A under editor */}
            <div className="px-4 py-2 border-t border-[var(--border)] bg-[var(--tile)] flex items-center gap-1.5 overflow-x-auto">
              <span className="text-[11px] text-[var(--muted)] font-mono uppercase tracking-wider mr-1 flex-shrink-0">
                Chips:
              </span>
              {PRESET_EXAMPLES.map((ex) => (
                <button
                  key={ex.title}
                  type="button"
                  onClick={() => {
                    setLogText(ex.log);
                    setResult(null);
                    setError(null);
                  }}
                  className="text-[12px] px-2.5 py-1 rounded-[6px] border border-[var(--border)] bg-[var(--tile-2)] text-[var(--muted)] hover:text-[var(--text)] hover:border-[var(--border-strong)] transition-colors whitespace-nowrap flex-shrink-0"
                >
                  {ex.title}
                </button>
              ))}
            </div>

            {/* Footer row: stats + analyze button */}
            <div className="flex items-center justify-between px-4 py-3 border-t border-[var(--border)] bg-[var(--tile)]">
              <div className="text-[12px] font-mono text-[var(--muted)] tabular-nums">
                {lineCount} {lineCount === 1 ? "line" : "lines"} · {charCount} chars
              </div>

              <Button
                onClick={handleSubmit}
                disabled={!logText.trim() || loading}
                className="h-10 px-4 rounded-[10px] bg-[var(--accent)] hover:opacity-95 text-white font-medium text-[13px] shadow-sm flex items-center gap-2 disabled:opacity-50 transition-all cursor-pointer"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <span>Analyze incident</span>
                    <kbd className="hidden sm:inline-block text-[11px] font-mono px-1.5 py-0.5 rounded bg-white/20 text-white">
                      ⌘Enter
                    </kbd>
                  </>
                )}
              </Button>
            </div>
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 1 - TILE B: Pipeline (col-span 5) */}
          {/* ========================================================= */}
          <motion.div
            custom={1}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className="order-2 md:order-2 col-span-1 md:col-span-3 xl:col-span-5 flex flex-col rounded-[16px] border border-[var(--border)] bg-[var(--tile)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none min-h-[460px] p-4"
          >
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Pipeline
                </span>
              </div>
              {loading && (
                <span className="text-[11px] font-mono text-[var(--muted)]">
                  (estimated)
                </span>
              )}
              {result && latencyMs !== null && (
                <span className="text-[12px] font-mono text-[var(--ok)] tabular-nums flex items-center gap-1">
                  <Check className="h-3.5 w-3.5" />
                  {latencyMs}ms
                </span>
              )}
            </div>

            {/* Stepper Stages */}
            <div className="flex-1 flex flex-col justify-around py-4">
              {PIPELINE_STAGES.map((stage, idx) => {
                const isFinished = result ? true : loading ? loadingStage > idx : false;
                const isActive = loading && loadingStage === idx;
                const isFailed = !!error && loadingStage === idx;

                const detailText =
                  idx === 3 && result?.trace?.model
                    ? result.trace.model
                    : stage.detail;

                return (
                  <div key={stage.name} className="flex items-start gap-3.5 relative">
                    {/* Connecting vertical line */}
                    {idx < PIPELINE_STAGES.length - 1 && (
                      <div
                        className={`absolute left-[13px] top-[26px] bottom-[-16px] w-[2px] transition-colors ${
                          isFinished ? "bg-[var(--ok)]" : "bg-[var(--border)]"
                        }`}
                        aria-hidden="true"
                      />
                    )}

                    {/* Step Icon Indicator */}
                    <div className="relative z-10 flex items-center justify-center h-7 w-7 rounded-full flex-shrink-0">
                      {isFinished ? (
                        <div className="h-7 w-7 rounded-full bg-[var(--ok)]/10 border border-[var(--ok)] flex items-center justify-center text-[var(--ok)]">
                          <Check className="h-3.5 w-3.5 stroke-[2.5]" />
                        </div>
                      ) : isActive ? (
                        <div className="h-7 w-7 rounded-full bg-[var(--accent-soft)] border border-[var(--accent)] flex items-center justify-center text-[var(--accent)]">
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        </div>
                      ) : isFailed ? (
                        <div className="h-7 w-7 rounded-full bg-[var(--danger)]/10 border border-[var(--danger)] flex items-center justify-center text-[var(--danger)]">
                          <AlertCircle className="h-3.5 w-3.5" />
                        </div>
                      ) : (
                        <div className="h-7 w-7 rounded-full bg-[var(--tile-2)] border border-[var(--border)] flex items-center justify-center text-[var(--muted)] text-[12px] font-mono">
                          {idx + 1}
                        </div>
                      )}
                    </div>

                    {/* Step text */}
                    <div className="flex-1 pt-0.5">
                      <div className="flex items-center justify-between">
                        <span
                          className={`text-[13px] font-medium transition-colors ${
                            isFinished
                              ? "text-[var(--text)]"
                              : isActive
                              ? "text-[var(--accent)]"
                              : isFailed
                              ? "text-[var(--danger)]"
                              : "text-[var(--muted)]"
                          }`}
                        >
                          {stage.title}
                        </span>
                        {isActive && (
                          <span className="text-[11px] font-mono text-[var(--accent)]">running</span>
                        )}
                      </div>
                      <p className="text-[12px] font-mono text-[var(--muted)] mt-0.5">
                        {detailText}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Cold start alert message if elapsed >3s */}
            {isColdStart && loading && (
              <div className="mt-auto p-3 rounded-[8px] bg-[var(--warn)]/10 border border-[var(--warn)]/30 text-[12px] text-[var(--warn)] flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
                <span>Backend is waking up, first request can take up to a minute</span>
              </div>
            )}
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 2 - TILE C: Cluster (col-span 3) */}
          {/* ========================================================= */}
          <motion.div
            custom={2}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className={`order-6 md:order-3 col-span-1 md:col-span-2 xl:col-span-3 flex flex-col justify-between rounded-[16px] border bg-[var(--tile)] p-4 min-h-[170px] ${
              !result && !loading
                ? "border-dashed border-[var(--border)] shadow-none"
                : "border-[var(--border)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none"
            }`}
          >
            <div className="flex items-center justify-between pb-2 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Cluster
                </span>
              </div>
              {result && (
                <Badge
                  variant="outline"
                  className={`text-[11px] font-mono ${
                    result.trace.is_noise
                      ? "border-[var(--warn)] text-[var(--warn)]"
                      : "border-[var(--border)] text-[var(--muted)]"
                  }`}
                >
                  {result.trace.is_noise ? "NOISE" : "ASSIGNED"}
                </Badge>
              )}
            </div>

            {loading ? (
              <div className="space-y-2 py-4">
                <Skeleton className="h-8 w-24 bg-[var(--border)]" />
                <Skeleton className="h-4 w-36 bg-[var(--border)]" />
              </div>
            ) : result ? (
              <div className="py-2">
                <div
                  className={`text-[28px] font-mono font-semibold tabular-nums tracking-tight ${
                    result.trace.is_noise ? "text-[var(--warn)]" : "text-[var(--text)]"
                  }`}
                >
                  {result.trace.is_noise ? "Novel" : `#${result.trace.cluster_id}`}
                </div>
                <div className="text-[12px] text-[var(--muted)] mt-1">
                  {result.trace.is_noise
                    ? "No historical cluster, treat as new"
                    : "Known failure cluster"}
                </div>
              </div>
            ) : (
              <div className="py-6 text-center text-[12px] text-[var(--muted)]">
                Cluster assignment will appear after analysis
              </div>
            )}
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 2 - TILE D: Best Match (col-span 5) */}
          {/* ========================================================= */}
          <motion.div
            custom={3}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className={`order-3 md:order-4 col-span-1 md:col-span-4 xl:col-span-5 flex flex-col justify-between rounded-[16px] border bg-[var(--tile)] p-4 min-h-[170px] ${
              !result && !loading
                ? "border-dashed border-[var(--border)] shadow-none"
                : "border-[var(--border)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none"
            }`}
          >
            <div className="flex items-center justify-between pb-2 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Best Match
                </span>
              </div>
              {result && (
                <div className="flex items-center gap-2">
                  {/* Retrieval Pass Badge */}
                  {result.retrieval_pass && (
                    <Badge
                      variant="outline"
                      className={`text-[11px] font-mono ${
                        result.retrieval_pass === 1
                          ? "border-[var(--accent)]/30 text-[var(--accent)] bg-[var(--accent-soft)]/50"
                          : "border-[var(--warn)]/30 text-[var(--warn)] bg-[var(--warn)]/10"
                      }`}
                    >
                      {result.retrieval_pass === 1 ? "Pass 1: high quality" : "Pass 2: fallback"}
                    </Badge>
                  )}
                  {/* Quality score badge if present */}
                  {(result.trace.retrieved_incident.quality_score !== undefined ||
                    result.trace.retrieved_incident.quality !== undefined) && (
                    <Badge variant="outline" className="text-[11px] font-mono border-[var(--border)] text-[var(--muted)]">
                      Q: {result.trace.retrieved_incident.quality_score ?? result.trace.retrieved_incident.quality}
                    </Badge>
                  )}
                </div>
              )}
            </div>

            {loading ? (
              <div className="flex items-center justify-between py-4">
                <div className="space-y-2">
                  <Skeleton className="h-5 w-32 bg-[var(--border)]" />
                  <Skeleton className="h-4 w-24 bg-[var(--border)]" />
                </div>
                <Skeleton className="h-14 w-24 rounded-full bg-[var(--border)]" />
              </div>
            ) : result ? (
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 py-2">
                {/* Left: Incident info */}
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[14px] font-semibold text-[var(--text)]">
                      {result.trace.retrieved_incident.incident_id}
                    </span>
                    <span className="text-[12px] text-[var(--muted)] font-medium">
                      ({extractCompany(result.trace.retrieved_incident.source_url)})
                    </span>
                  </div>

                  {/* Weak match note if similarity < 0.40 */}
                  {result.trace.similarity_score < 0.4 ? (
                    <p className="text-[11px] text-[var(--warn)] font-medium">
                      Weak match, treat as a hypothesis
                    </p>
                  ) : (
                    <p className="text-[11px] text-[var(--muted)] font-mono">
                      High confidence semantic correlation
                    </p>
                  )}

                  {/* Open Source Button */}
                  {result.trace.retrieved_incident.source_url && (
                    <a
                      href={result.trace.retrieved_incident.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[12px] text-[var(--accent)] hover:underline pt-1"
                    >
                      <span>Open source</span>
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  )}
                </div>

                {/* Right: Semicircular Arc Gauge */}
                <SimilarityGauge value={result.trace.similarity_score} />
              </div>
            ) : (
              <div className="py-6 text-center text-[12px] text-[var(--muted)]">
                Nearest historical incident will be correlated here
              </div>
            )}
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 2 - TILE E: Run Telemetry (col-span 4) */}
          {/* ========================================================= */}
          <motion.div
            custom={4}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className={`order-7 md:order-5 col-span-1 md:col-span-6 xl:col-span-4 flex flex-col justify-between rounded-[16px] border bg-[var(--tile)] p-4 min-h-[170px] ${
              !result && !loading
                ? "border-dashed border-[var(--border)] shadow-none"
                : "border-[var(--border)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none"
            }`}
          >
            <div className="flex items-center justify-between pb-2 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Run Telemetry
                </span>
              </div>
              {result && (
                <Badge
                  variant="outline"
                  className={`text-[10px] font-mono font-semibold ${
                    result.is_mock
                      ? "border-[var(--warn)] text-[var(--warn)] bg-[var(--warn)]/10"
                      : "border-[var(--ok)] text-[var(--ok)] bg-[var(--ok)]/10"
                  }`}
                >
                  {result.is_mock ? "MOCK" : "LIVE"}
                </Badge>
              )}
            </div>

            {loading ? (
              <div className="space-y-2 py-3">
                <Skeleton className="h-4 w-full bg-[var(--border)]" />
                <Skeleton className="h-4 w-3/4 bg-[var(--border)]" />
                <Skeleton className="h-4 w-1/2 bg-[var(--border)]" />
              </div>
            ) : result ? (
              <div className="space-y-2 py-1 text-[12px] font-mono">
                {result.trace.model && (
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--muted)]">Model:</span>
                    <span className="text-[var(--text)] truncate max-w-[180px]">{result.trace.model}</span>
                  </div>
                )}
                {latencyMs !== null && (
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--muted)]">Latency:</span>
                    <span className="text-[var(--text)] tabular-nums">{latencyMs}ms</span>
                  </div>
                )}
                {timestamp && (
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--muted)]">Timestamp:</span>
                    <span className="text-[var(--text)] tabular-nums">{timestamp}</span>
                  </div>
                )}
                {result.incident_id && (
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--muted)]">ID:</span>
                    <span className="text-[var(--text)] tabular-nums">{result.incident_id}</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-6 text-center text-[12px] text-[var(--muted)]">
                Telemetry and execution metadata
              </div>
            )}
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 3 - TILE F: Root Cause (col-span 7) */}
          {/* ========================================================= */}
          <motion.div
            custom={5}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className={`order-4 md:order-6 col-span-1 md:col-span-3 xl:col-span-7 flex flex-col rounded-[16px] border bg-[var(--tile)] min-h-[300px] ${
              !result && !loading
                ? "border-dashed border-[var(--border)] shadow-none"
                : "border-[var(--border)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none"
            }`}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Root Cause
                </span>
              </div>
              {result && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(result.root_cause, "Root cause")}
                  className="h-7 text-[12px] text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] rounded-lg px-2"
                >
                  <Copy className="h-3.5 w-3.5 mr-1" />
                  Copy
                </Button>
              )}
            </div>

            <div className="p-5 flex-1 flex flex-col justify-center">
              {loading ? (
                <div className="space-y-3">
                  <Skeleton className="h-4 w-full bg-[var(--border)]" />
                  <Skeleton className="h-4 w-5/6 bg-[var(--border)]" />
                  <Skeleton className="h-4 w-4/6 bg-[var(--border)]" />
                  <Skeleton className="h-4 w-3/4 bg-[var(--border)]" />
                </div>
              ) : result ? (
                <div className="max-w-[64ch] text-[15px] leading-[1.65] text-[var(--text)] whitespace-pre-line">
                  {result.root_cause}
                </div>
              ) : (
                <div className="text-center text-[12px] text-[var(--muted)]">
                  Root cause synthesis will be generated by the model
                </div>
              )}
            </div>
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 3 - TILE G: Remediation (col-span 5) */}
          {/* ========================================================= */}
          <motion.div
            custom={6}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className={`order-5 md:order-7 col-span-1 md:col-span-3 xl:col-span-5 flex flex-col rounded-[16px] border bg-[var(--tile)] min-h-[300px] ${
              !result && !loading
                ? "border-dashed border-[var(--border)] shadow-none"
                : "border-[var(--border)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none"
            }`}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
              <div className="flex items-center gap-2">
                <ListChecks className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                  Remediation
                </span>
              </div>
              {result && remediationList.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-[12px] font-mono text-[var(--muted)] tabular-nums">
                    {completedStepsCount} of {remediationList.length} done
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(result.remediation_steps, "Remediation steps")}
                    className="h-7 text-[12px] text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] rounded-lg px-2"
                  >
                    <Copy className="h-3.5 w-3.5 mr-1" />
                    Copy all
                  </Button>
                </div>
              )}
            </div>

            <div className="p-4 flex-1 flex flex-col justify-center">
              {loading ? (
                <div className="space-y-3">
                  <Skeleton className="h-5 w-full bg-[var(--border)]" />
                  <Skeleton className="h-5 w-4/5 bg-[var(--border)]" />
                  <Skeleton className="h-5 w-5/6 bg-[var(--border)]" />
                </div>
              ) : result && remediationList.length > 0 ? (
                <div className="space-y-2.5">
                  {remediationList.map((step, idx) => {
                    const isDone = !!checkedSteps[idx];
                    return (
                      <div
                        key={idx}
                        onClick={() =>
                          setCheckedSteps((prev) => ({
                            ...prev,
                            [idx]: !prev[idx],
                          }))
                        }
                        className="flex items-start gap-3 p-2 rounded-lg hover:bg-[var(--tile-2)] transition-colors cursor-pointer group"
                      >
                        <button
                          type="button"
                          aria-label={`Mark step ${idx + 1} as ${isDone ? "incomplete" : "complete"}`}
                          className="mt-0.5 text-[var(--muted)] group-hover:text-[var(--accent)] transition-colors flex-shrink-0"
                        >
                          {isDone ? (
                            <CheckSquare className="h-4 w-4 text-[var(--accent)]" />
                          ) : (
                            <Square className="h-4 w-4" />
                          )}
                        </button>
                        <div
                          className={`text-[13px] leading-relaxed transition-colors ${
                            isDone ? "line-through text-[var(--muted)]" : "text-[var(--text)]"
                          }`}
                        >
                          <span className="font-mono text-[12px] text-[var(--muted)] mr-1.5 tabular-nums">
                            {idx + 1}.
                          </span>
                          {step}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center text-[12px] text-[var(--muted)]">
                  Actionable remediation checklist
                </div>
              )}
            </div>
          </motion.div>

          {/* ========================================================= */}
          {/* ROW 4 - TILE H: Pipeline Trace (col-span 12) */}
          {/* ========================================================= */}
          <motion.div
            custom={7}
            initial="hidden"
            animate="visible"
            variants={tileVariants}
            className={`order-8 md:order-8 col-span-1 md:col-span-6 xl:col-span-12 flex flex-col rounded-[16px] border bg-[var(--tile)] min-h-[280px] overflow-hidden ${
              !result && !loading
                ? "border-dashed border-[var(--border)] shadow-none"
                : "border-[var(--border)] shadow-[0_1px_2px_rgba(16,16,20,0.04)] dark:shadow-none"
            }`}
          >
            {result ? (
              <Tabs defaultValue="timeline" className="w-full flex flex-col flex-1">
                <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border)] bg-[var(--tile)]">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-[var(--muted)]" strokeWidth={1.5} />
                    <span className="text-[12px] font-medium tracking-wide uppercase text-[var(--muted)]">
                      Pipeline Trace
                    </span>
                  </div>

                  <TabsList className="bg-[var(--tile-2)] border border-[var(--border)] h-8 p-0.5 rounded-lg">
                    <TabsTrigger
                      value="timeline"
                      className="text-[12px] h-7 px-3 rounded-md data-[state=active]:bg-[var(--tile)] data-[state=active]:text-[var(--text)]"
                    >
                      Timeline
                    </TabsTrigger>
                    <TabsTrigger
                      value="prompt"
                      className="text-[12px] h-7 px-3 rounded-md data-[state=active]:bg-[var(--tile)] data-[state=active]:text-[var(--text)]"
                    >
                      Prompt
                    </TabsTrigger>
                  </TabsList>
                </div>

                {/* Timeline Tab */}
                <TabsContent value="timeline" className="p-6 m-0 flex-1">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
                    {/* Node 1: Anomaly */}
                    <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--tile-2)]/60 flex flex-col gap-2">
                      <div className="flex items-center justify-between text-[11px] font-mono text-[var(--muted)] uppercase tracking-wider">
                        <span>01 · Anomaly</span>
                        <span className="h-2 w-2 rounded-full bg-[var(--ok)]" />
                      </div>
                      <div className="text-[13px] font-medium text-[var(--text)]">
                        Cluster #{result.trace.cluster_id}
                      </div>
                      <p className="text-[12px] font-mono text-[var(--muted)]">
                        Length: {result.raw_log_excerpt.length} chars
                      </p>
                      <Badge variant="outline" className="w-fit text-[10px] font-mono border-[var(--border)]">
                        {result.trace.is_noise ? "Novel pattern" : "Known pattern"}
                      </Badge>
                    </div>

                    {/* Node 2: Retrieval */}
                    <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--tile-2)]/60 flex flex-col gap-2">
                      <div className="flex items-center justify-between text-[11px] font-mono text-[var(--muted)] uppercase tracking-wider">
                        <span>02 · Retrieval</span>
                        <span className="h-2 w-2 rounded-full bg-[var(--ok)]" />
                      </div>
                      <div className="text-[13px] font-medium text-[var(--text)] truncate">
                        {result.trace.retrieved_incident.incident_id}
                      </div>
                      <p className="text-[12px] font-mono text-[var(--muted)] tabular-nums">
                        Sim: {(result.trace.similarity_score * 100).toFixed(1)}%
                      </p>
                      <Badge
                        variant="outline"
                        className={`w-fit text-[10px] font-mono ${
                          result.retrieval_pass === 1
                            ? "border-[var(--accent)] text-[var(--accent)]"
                            : "border-[var(--warn)] text-[var(--warn)]"
                        }`}
                      >
                        Pass {result.retrieval_pass ?? 1}
                      </Badge>
                    </div>

                    {/* Node 3: Prompt */}
                    <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--tile-2)]/60 flex flex-col gap-2">
                      <div className="flex items-center justify-between text-[11px] font-mono text-[var(--muted)] uppercase tracking-wider">
                        <span>03 · Prompt</span>
                        <span className="h-2 w-2 rounded-full bg-[var(--ok)]" />
                      </div>
                      <div className="text-[13px] font-medium text-[var(--text)]">
                        Instruction Context
                      </div>
                      <p className="text-[12px] font-mono text-[var(--muted)] tabular-nums">
                        {result.trace.prompt_used.length} chars
                      </p>
                      <Badge variant="outline" className="w-fit text-[10px] font-mono border-[var(--border)]">
                        System + Incident
                      </Badge>
                    </div>

                    {/* Node 4: Generation */}
                    <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--tile-2)]/60 flex flex-col gap-2">
                      <div className="flex items-center justify-between text-[11px] font-mono text-[var(--muted)] uppercase tracking-wider">
                        <span>04 · Generation</span>
                        <span className="h-2 w-2 rounded-full bg-[var(--ok)]" />
                      </div>
                      <div className="text-[13px] font-medium text-[var(--text)] truncate">
                        {result.trace.model || "openai/gpt-oss-120b"}
                      </div>
                      <p className="text-[12px] font-mono text-[var(--muted)] tabular-nums">
                        {latencyMs !== null ? `${latencyMs}ms` : "Completed"}
                      </p>
                      <Badge
                        variant="outline"
                        className={`w-fit text-[10px] font-mono ${
                          result.is_mock
                            ? "border-[var(--warn)] text-[var(--warn)]"
                            : "border-[var(--ok)] text-[var(--ok)]"
                        }`}
                      >
                        {result.is_mock ? "Mock Output" : "Live Groq Inference"}
                      </Badge>
                    </div>
                  </div>
                </TabsContent>

                {/* Prompt Tab */}
                <TabsContent value="prompt" className="p-4 m-0 flex-1 flex flex-col">
                  <div className="flex items-center justify-between pb-3">
                    <span className="text-[12px] font-mono text-[var(--muted)]">
                      Exact prompt sent to the model
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(result.trace.prompt_used, "Prompt")}
                      className="h-7 text-[12px] text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--tile-2)] rounded-lg px-2"
                    >
                      <Copy className="h-3.5 w-3.5 mr-1" />
                      Copy prompt
                    </Button>
                  </div>
                  <ScrollArea className="max-h-[360px] rounded-lg border border-[var(--border)] bg-[var(--tile-2)] p-4 font-mono text-[12px] leading-relaxed text-[var(--text)] whitespace-pre-wrap">
                    {result.trace.prompt_used}
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            ) : loading ? (
              <div className="p-6 space-y-4">
                <Skeleton className="h-6 w-48 bg-[var(--border)]" />
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Skeleton className="h-28 rounded-xl bg-[var(--border)]" />
                  <Skeleton className="h-28 rounded-xl bg-[var(--border)]" />
                  <Skeleton className="h-28 rounded-xl bg-[var(--border)]" />
                  <Skeleton className="h-28 rounded-xl bg-[var(--border)]" />
                </div>
              </div>
            ) : (
              <div className="p-10 text-center text-[12px] text-[var(--muted)]">
                Full retrieval trace and prompt details will populate on run
              </div>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}
