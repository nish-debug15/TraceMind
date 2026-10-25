"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { toast } from "sonner";
import {
  AlertTriangle,
  AlertCircle,
  Check,
  ChevronRight,
  ChevronDown,
  Clock,
  Copy,
  ExternalLink,
  Loader2,
  RotateCcw,
  Terminal,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";

// --- Types matching Backend Models strictly ---

type RetrievedIncident = {
  incident_id: string;
  raw_log_excerpt: string;
  root_cause: string;
  remediation_steps: string;
  source_url: string;
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

const LOADING_STAGES = [
  "Embedding log vector",
  "Clustering anomaly signature",
  "Retrieving historical postmortem",
  "Generating root cause & remediation",
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
    .map((s) => s.trim())
    .filter(Boolean);
}

function highlightTokens(text: string) {
  const parts = text.split(/(ERROR|FATAL|WARN(?:ING)?)/gi);
  return parts.map((part, i) => {
    const upper = part.toUpperCase();
    if (upper === "ERROR" || upper === "FATAL") {
      return (
        <span key={i} className="text-[#E5534B] font-semibold">
          {part}
        </span>
      );
    }
    if (upper === "WARN" || upper === "WARNING") {
      return (
        <span key={i} className="text-[#E5A23B] font-semibold">
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export default function Home() {
  const [logText, setLogText] = useState(PRESET_EXAMPLES[0].log);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [checkedSteps, setCheckedSteps] = useState<Record<number, boolean>>({});
  const [promptOpen, setPromptOpen] = useState(false);

  // Health check state
  const [apiStatus, setApiStatus] = useState<"ok" | "waking" | "error">("ok");
  const [lastChecked, setLastChecked] = useState<string>("Initializing");
  const [knownModel, setKnownModel] = useState<string>("openai/gpt-oss-120b");

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

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
    setError(null);
    setResult(null);
    setCheckedSteps({});
    setLoadingStage(0);

    // Simulate stage progress timers
    const stageTimer1 = setTimeout(() => setLoadingStage(1), 500);
    const stageTimer2 = setTimeout(() => setLoadingStage(2), 1200);
    const stageTimer3 = setTimeout(() => setLoadingStage(3), 2000);

    const startTime = performance.now();

    try {
      const res = await fetch(`${baseUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: logText }),
      });

      const elapsed = Math.round(performance.now() - startTime);
      setLatencyMs(elapsed);

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
      if (data.trace?.model) {
        setKnownModel(data.trace.model);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to connect to backend service.";
      setError({ message: msg, statusCode: 502 });
    } finally {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      clearTimeout(stageTimer3);
      setLoading(false);
    }
  }, [logText, loading, baseUrl]);

  // --- Keyboard Shortcuts ---
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  useEffect(() => {
    const globalKeyHandler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      }
    };
    window.addEventListener("keydown", globalKeyHandler);
    return () => window.removeEventListener("keydown", globalKeyHandler);
  }, [handleSubmit]);

  // --- Helper actions ---
  const toggleStep = (idx: number) => {
    setCheckedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const copyRemediation = (steps: string[]) => {
    const text = steps.map((s, i) => `${i + 1}. ${s}`).join("\n");
    navigator.clipboard.writeText(text);
    toast.success("Remediation checklist copied to clipboard");
  };

  const copyPrompt = (prompt: string) => {
    navigator.clipboard.writeText(prompt);
    toast.success("Prompt payload copied to clipboard");
  };

  const lines = logText.split("\n");
  const lineCount = Math.max(lines.length, 1);
  const remediationList = result ? parseRemediationSteps(result.remediation_steps) : [];
  const similarity = result?.trace?.similarity_score ?? 0;
  const passTier = result?.trace?.retrieval_pass ?? result?.retrieval_pass ?? (similarity >= 0.40 ? 1 : 2);
  const isWeakMatch = Boolean(result && similarity < 0.40);
  const company = result ? extractCompany(result.trace?.retrieved_incident?.source_url || result.source_url) : "";

  return (
    <div className="flex flex-col min-h-screen bg-[#0A0A0B] text-[#EDEDEF]">
      {/* 1. TOP BAR (48px) */}
      <header className="h-12 border-b border-[#26262B] bg-[#0A0A0B] px-4 flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center gap-2.5">
          <Terminal className="w-4 h-4 text-[#EDEDEF]" aria-hidden="true" />
          <span className="font-semibold text-[14px] text-[#EDEDEF] tracking-tight">TraceMind</span>
          <span className="text-[11px] font-mono text-[#8B8B94] px-1.5 py-0.5 rounded bg-[#17171A] border border-[#26262B]">
            ops-console
          </span>
        </div>

        {/* Right-aligned status cluster */}
        <div className="flex items-center gap-3">
          {/* API Health Dot */}
          <Tooltip>
            <TooltipTrigger className="flex items-center gap-1.5 cursor-pointer px-2 py-1 rounded hover:bg-[#17171A] transition-colors outline-none focus-visible:ring-1 focus-visible:ring-[#6E7AE8]">
              {apiStatus === "waking" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-[#E5A23B]" />
                  <span className="text-[12px] text-[#E5A23B] font-mono">Backend waking up</span>
                </>
              ) : (
                <>
                  <span
                    className={`w-2 h-2 rounded-full ${
                      apiStatus === "ok" ? "bg-[#3FB67C]" : "bg-[#E5534B]"
                    }`}
                  />
                  <span className="text-[12px] font-mono text-[#8B8B94]">
                    {apiStatus === "ok" ? "API online" : "API offline"}
                  </span>
                </>
              )}
            </TooltipTrigger>
            <TooltipContent side="bottom" className="text-[11px] font-mono bg-[#17171A] border-[#26262B] text-[#EDEDEF]">
              GET {baseUrl}/health · Checked {lastChecked}
            </TooltipContent>
          </Tooltip>

          {/* Model ID */}
          <span className="hidden sm:inline-flex font-mono text-[11px] text-[#8B8B94] px-2 py-0.5 rounded bg-[#111113] border border-[#26262B] tabular-nums">
            {knownModel}
          </span>

          {/* Mode Badge */}
          <Badge
            variant="outline"
            className={`font-mono text-[11px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-[4px] ${
              result?.is_mock
                ? "border-[#E5A23B]/40 bg-[#E5A23B]/10 text-[#E5A23B]"
                : "border-[#3FB67C]/40 bg-[#3FB67C]/10 text-[#3FB67C]"
            }`}
          >
            {result?.is_mock ? "MOCK" : "LIVE"}
          </Badge>
        </div>
      </header>

      {/* 2. MAIN AREA: TWO-PANE SPLIT */}
      <main className="w-full max-w-[1440px] mx-auto flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* LEFT PANE: Log input (45%) */}
        <section className="w-full lg:w-[45%] border-b lg:border-b-0 lg:border-r border-[#26262B] flex flex-col bg-[#0A0A0B] overflow-hidden">
          {/* Pane Header (40px) */}
          <div className="h-10 border-b border-[#26262B] bg-[#111113] px-3.5 flex items-center justify-between shrink-0 select-none">
            <span className="text-[11px] font-semibold text-[#8B8B94] tracking-wider uppercase font-mono">
              LOG INPUT
            </span>

            <div className="flex items-center gap-2">
              {/* Preset Dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger className="inline-flex items-center justify-center h-7 text-[12px] px-2.5 bg-[#17171A] border border-[#26262B] text-[#EDEDEF] hover:bg-[#26262B] hover:text-[#FFFFFF] rounded-[6px] font-medium transition-colors select-none outline-none focus-visible:ring-1 focus-visible:ring-[#6E7AE8]">
                  Load example
                  <ChevronDown className="w-3.5 h-3.5 ml-1 text-[#8B8B94]" />
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  className="bg-[#17171A] border-[#26262B] text-[#EDEDEF] rounded-[6px] shadow-[0_4px_12px_rgba(0,0,0,0.5)] p-1 min-w-[180px]"
                >
                  {PRESET_EXAMPLES.map((ex, idx) => (
                    <DropdownMenuItem
                      key={idx}
                      onClick={() => setLogText(ex.log)}
                      className="text-[12px] font-mono cursor-pointer hover:bg-[#26262B] focus:bg-[#26262B] focus:text-[#FFFFFF] rounded-[4px] px-2.5 py-1.5"
                    >
                      {ex.title}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Clear Button */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setLogText("")}
                disabled={!logText}
                className="h-7 text-[12px] px-2 text-[#8B8B94] hover:text-[#EDEDEF] hover:bg-[#17171A] rounded-[6px] disabled:opacity-40"
              >
                Clear
              </Button>
            </div>
          </div>

          {/* Code-style Editor with synchronized Gutter & Highlight Layer */}
          <div className="flex-1 min-h-[340px] flex overflow-hidden bg-[#0A0A0B] relative">
            {/* Line numbers gutter */}
            <div
              ref={gutterRef}
              aria-hidden="true"
              className="w-11 py-3 bg-[#111113] border-r border-[#26262B] text-right pr-2.5 text-[12px] font-mono text-[#52525B] select-none overflow-hidden shrink-0"
            >
              {Array.from({ length: lineCount }).map((_, i) => (
                <div key={i} className="leading-[20px] h-[20px] tabular-nums">
                  {i + 1}
                </div>
              ))}
            </div>

            {/* Editor Textarea with Syntax Highlight Overlay */}
            <div className="relative flex-1 min-w-0 h-full overflow-hidden">
              {/* Highlight Overlay Layer */}
              <div
                ref={highlightRef}
                aria-hidden="true"
                className="absolute inset-0 pointer-events-none select-none p-3 font-mono text-[12.5px] leading-[20px] whitespace-pre overflow-hidden text-[#EDEDEF]"
              >
                {lines.map((line, idx) => (
                  <div key={idx} className="h-[20px] leading-[20px]">
                    {highlightTokens(line) || <span>&nbsp;</span>}
                  </div>
                ))}
              </div>

              {/* Functional Transparent Textarea */}
              <textarea
                ref={textareaRef}
                value={logText}
                onChange={(e) => setLogText(e.target.value)}
                onScroll={handleScroll}
                onKeyDown={handleKeyDown}
                placeholder="Paste infrastructure error logs or stack traces here..."
                spellCheck={false}
                className="absolute inset-0 w-full h-full p-3 font-mono text-[12.5px] leading-[20px] whitespace-pre overflow-auto bg-transparent text-transparent caret-[#EDEDEF] resize-none outline-none selection:bg-[#6E7AE8]/30 selection:text-transparent"
              />
            </div>
          </div>

          {/* Footer Bar */}
          <div className="h-10 border-t border-[#26262B] bg-[#111113] px-3.5 flex items-center justify-between shrink-0 select-none">
            <span className="text-[12px] font-mono text-[#8B8B94] tabular-nums">
              {logText.length.toLocaleString()} chars · {lineCount} lines
            </span>

            <Button
              disabled={!logText.trim() || loading}
              onClick={handleSubmit}
              className="bg-[#6E7AE8] hover:bg-[#5C68D4] text-[#FFFFFF] h-7 px-3 text-[12px] font-medium rounded-[6px] transition-colors focus-visible:ring-1 focus-visible:ring-[#6E7AE8] disabled:opacity-40"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                  Analyzing
                </>
              ) : (
                <>
                  Analyze
                  <kbd className="ml-1.5 text-[10px] bg-[#0A0A0B]/40 border border-[#26262B] px-1 py-0.5 rounded text-[#EDEDEF]/80 font-mono">
                    ⌘↵
                  </kbd>
                </>
              )}
            </Button>
          </div>
        </section>

        {/* RIGHT PANE: Result (55%) */}
        <section className="w-full lg:w-[55%] flex flex-col bg-[#0A0A0B] overflow-hidden">
          {/* Pane Header (40px) */}
          <div className="h-10 border-b border-[#26262B] bg-[#111113] px-3.5 flex items-center justify-between shrink-0 select-none">
            <span className="text-[11px] font-semibold text-[#8B8B94] tracking-wider uppercase font-mono">
              ANALYSIS RESULT
            </span>

            {latencyMs !== null && result && (
              <span className="text-[12px] font-mono text-[#8B8B94] tabular-nums flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-[#8B8B94]" />
                {latencyMs}ms
              </span>
            )}
          </div>

          {/* Pane Body Area */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
            {/* STATE 1: IDLE */}
            {!loading && !result && !error && (
              <div className="h-full min-h-[300px] flex flex-col items-center justify-center text-center p-6 space-y-4">
                <p className="text-[13px] text-[#8B8B94] max-w-sm">
                  Paste infrastructure logs or load a preset example on the left to analyze root cause and remediation.
                </p>

                {/* Compact 4-step pipeline line */}
                <div className="flex items-center gap-2 text-[12px] font-mono text-[#8B8B94] bg-[#111113] px-3.5 py-1.5 rounded-[6px] border border-[#26262B]">
                  <span>Cluster</span>
                  <ChevronRight className="w-3.5 h-3.5 text-[#52525B]" />
                  <span>Retrieve</span>
                  <ChevronRight className="w-3.5 h-3.5 text-[#52525B]" />
                  <span>Generate</span>
                  <ChevronRight className="w-3.5 h-3.5 text-[#52525B]" />
                  <span>Audit</span>
                </div>
              </div>
            )}

            {/* STATE 2: LOADING */}
            {loading && (
              <div className="space-y-6 py-2">
                {/* Stage Progress List */}
                <div className="border border-[#26262B] bg-[#111113] rounded-[6px] p-4 space-y-3">
                  <div className="text-[11px] font-mono uppercase tracking-wider text-[#8B8B94] flex items-center justify-between">
                    <span>Pipeline Progress</span>
                    <span className="text-[11px] text-[#52525B]">Simulated execution</span>
                  </div>

                  <div className="space-y-2.5">
                    {LOADING_STAGES.map((stage, idx) => {
                      const isDone = idx < loadingStage;
                      const isCurrent = idx === loadingStage;
                      return (
                        <div key={idx} className="flex items-center gap-2.5 text-[12.5px] font-mono">
                          {isDone ? (
                            <Check className="w-3.5 h-3.5 text-[#3FB67C] shrink-0" />
                          ) : isCurrent ? (
                            <Loader2 className="w-3.5 h-3.5 text-[#6E7AE8] animate-spin shrink-0" />
                          ) : (
                            <span className="w-3.5 h-3.5 flex items-center justify-center">
                              <span className="w-1.5 h-1.5 rounded-full bg-[#26262B]" />
                            </span>
                          )}
                          <span
                            className={
                              isDone
                                ? "text-[#8B8B94]"
                                : isCurrent
                                ? "text-[#EDEDEF] font-medium"
                                : "text-[#52525B]"
                            }
                          >
                            {stage}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Skeletons */}
                <div className="space-y-3 pt-2">
                  <Skeleton className="h-4 w-28 bg-[#17171A]" />
                  <Skeleton className="h-16 w-full bg-[#111113] border border-[#26262B]" />
                  <Skeleton className="h-4 w-36 bg-[#17171A] mt-4" />
                  <Skeleton className="h-24 w-full bg-[#111113] border border-[#26262B]" />
                </div>
              </div>
            )}

            {/* STATE 3: ERROR */}
            {error && !loading && (
              <div className="border border-[#E5534B]/40 bg-[#E5534B]/5 rounded-[6px] p-4 space-y-3">
                <div className="flex items-center gap-2 text-[#E5534B] font-medium text-[13px]">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Inference Error ({error.statusCode || "502"})</span>
                </div>
                <p className="text-[12px] font-mono text-[#EDEDEF] leading-relaxed break-words">
                  {error.message}
                </p>
                <div className="pt-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSubmit}
                    className="h-7 text-[12px] border-[#E5534B]/40 text-[#EDEDEF] hover:bg-[#E5534B]/10 hover:text-[#FFFFFF] rounded-[6px]"
                  >
                    <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
                    Retry
                  </Button>
                </div>
              </div>
            )}

            {/* STATE 4: RESULT */}
            {result && !loading && (
              <div className="space-y-4">
                {/* Result Header Incident Summary Bar */}
                <div className="border border-[#26262B] bg-[#111113] rounded-[6px] p-3 flex flex-wrap items-center gap-2.5">
                  {/* Cluster Badge */}
                  {result.trace.is_noise ? (
                    <Badge className="bg-[#E5A23B]/10 text-[#E5A23B] border border-[#E5A23B]/30 font-mono text-[11px] rounded-[4px]">
                      NOVEL
                    </Badge>
                  ) : (
                    <Badge className="bg-[#17171A] text-[#EDEDEF] border border-[#26262B] font-mono text-[11px] rounded-[4px]">
                      Cluster #{result.trace.cluster_id}
                    </Badge>
                  )}

                  {/* Similarity Badge with 4px thin bar */}
                  <div className="flex items-center gap-2 px-2 py-1 rounded bg-[#17171A] border border-[#26262B] font-mono text-[11px] text-[#EDEDEF] tabular-nums">
                    <span>Sim {similarity.toFixed(2)}</span>
                    <div className="h-1 w-12 bg-[#26262B] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#6E7AE8]"
                        style={{ width: `${Math.min(100, Math.max(0, similarity * 100))}%` }}
                      />
                    </div>
                  </div>

                  {/* Pass Tier */}
                  <Badge className="bg-[#17171A] text-[#8B8B94] border border-[#26262B] font-mono text-[11px] rounded-[4px]">
                    {passTier === 1 ? "Pass 1 (high-quality)" : "Pass 2 (fallback)"}
                  </Badge>

                  {/* Company */}
                  {company && (
                    <Badge className="bg-[#17171A] text-[#8B8B94] border border-[#26262B] font-mono text-[11px] rounded-[4px] uppercase">
                      {company}
                    </Badge>
                  )}

                  {/* Mock Flag */}
                  {result.is_mock && (
                    <Badge className="bg-[#E5A23B]/10 text-[#E5A23B] border border-[#E5A23B]/30 font-mono text-[11px] rounded-[4px]">
                      MOCK
                    </Badge>
                  )}
                </div>

                {/* Weak Match Warning Banner */}
                {isWeakMatch && (
                  <div className="border border-[#E5A23B]/40 bg-[#E5A23B]/10 rounded-[6px] px-3.5 py-2.5 text-[12px] text-[#E5A23B] flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>Weak match (&lt;0.40) · Treat retrieved postmortem as a hypothesis, not ground truth.</span>
                  </div>
                )}

                {/* Tabs: Analysis vs Pipeline Trace */}
                <Tabs defaultValue="analysis" className="w-full">
                  <TabsList className="bg-[#111113] border border-[#26262B] p-0.5 rounded-[6px] h-8 w-fit mb-3">
                    <TabsTrigger
                      value="analysis"
                      className="text-[12px] font-mono px-3 py-1 data-[state=active]:bg-[#17171A] data-[state=active]:text-[#6E7AE8] data-[state=active]:shadow-none rounded-[4px]"
                    >
                      Analysis
                    </TabsTrigger>
                    <TabsTrigger
                      value="trace"
                      className="text-[12px] font-mono px-3 py-1 data-[state=active]:bg-[#17171A] data-[state=active]:text-[#6E7AE8] data-[state=active]:shadow-none rounded-[4px]"
                    >
                      Pipeline trace
                    </TabsTrigger>
                  </TabsList>

                  {/* TAB 1: ANALYSIS */}
                  <TabsContent value="analysis" className="space-y-4 mt-0">
                    {/* Root Cause */}
                    <div className="border border-[#26262B] bg-[#111113] rounded-[6px] p-4 space-y-2">
                      <div className="text-[11px] font-mono text-[#8B8B94] uppercase tracking-wider font-semibold">
                        Root cause
                      </div>
                      <div className="text-[14px] leading-[1.6] max-w-[68ch] text-[#EDEDEF] whitespace-pre-wrap font-sans">
                        {result.root_cause}
                      </div>
                    </div>

                    {/* Remediation Steps Checklist */}
                    <div className="border border-[#26262B] bg-[#111113] rounded-[6px] p-4 space-y-3">
                      <div className="flex items-center justify-between pb-1">
                        <div className="text-[11px] font-mono text-[#8B8B94] uppercase tracking-wider font-semibold">
                          Remediation checklist ({remediationList.length})
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyRemediation(remediationList)}
                          className="h-6 px-2 text-[11px] text-[#8B8B94] hover:text-[#EDEDEF] hover:bg-[#17171A] rounded-[4px]"
                        >
                          <Copy className="w-3 h-3 mr-1" />
                          Copy all
                        </Button>
                      </div>

                      <div className="divide-y divide-[#26262B]">
                        {remediationList.map((step, idx) => {
                          const isChecked = !!checkedSteps[idx];
                          return (
                            <div key={idx} className="flex items-start gap-3 py-2.5">
                              <Checkbox
                                id={`step-${idx}`}
                                checked={isChecked}
                                onCheckedChange={() => toggleStep(idx)}
                                className="mt-0.5 border-[#3F3F46] data-[state=checked]:bg-[#6E7AE8] data-[state=checked]:border-[#6E7AE8] rounded-[3px]"
                              />
                              <label
                                htmlFor={`step-${idx}`}
                                className={`text-[13px] leading-relaxed cursor-pointer select-none transition-colors ${
                                  isChecked ? "text-[#8B8B94] line-through" : "text-[#EDEDEF]"
                                }`}
                              >
                                {step}
                              </label>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </TabsContent>

                  {/* TAB 2: PIPELINE TRACE */}
                  <TabsContent value="trace" className="mt-0">
                    <div className="border border-[#26262B] bg-[#111113] rounded-[6px] p-5">
                      {/* Vertical timeline */}
                      <div className="relative pl-6 space-y-6 before:absolute before:left-[7px] before:top-2 before:bottom-2 before:w-[1px] before:bg-[#26262B]">
                        {/* 1. Anomaly Detection */}
                        <div className="relative">
                          <span className="absolute -left-[27px] top-1 w-2 h-2 rounded-full bg-[#6E7AE8] ring-4 ring-[#111113]" />
                          <div className="text-[11px] font-mono text-[#8B8B94] uppercase tracking-wider font-semibold mb-1">
                            1. Anomaly Detection (HDBSCAN)
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-[12px] font-mono bg-[#0A0A0B] p-2.5 rounded-[4px] border border-[#26262B]">
                            <div>
                              <span className="text-[#8B8B94]">Cluster ID: </span>
                              <span className="text-[#EDEDEF]">#{result.trace.cluster_id}</span>
                            </div>
                            <div>
                              <span className="text-[#8B8B94]">Noise status: </span>
                              <span className={result.trace.is_noise ? "text-[#E5A23B]" : "text-[#3FB67C]"}>
                                {result.trace.is_noise ? "true (novel pattern)" : "false (known cluster)"}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* 2. Retrieval */}
                        <div className="relative">
                          <span className="absolute -left-[27px] top-1 w-2 h-2 rounded-full bg-[#6E7AE8] ring-4 ring-[#111113]" />
                          <div className="text-[11px] font-mono text-[#8B8B94] uppercase tracking-wider font-semibold mb-1">
                            2. Vector Retrieval (ChromaDB 2-Pass)
                          </div>
                          <div className="space-y-2 text-[12px] font-mono bg-[#0A0A0B] p-2.5 rounded-[4px] border border-[#26262B]">
                            <div className="flex items-center justify-between">
                              <div>
                                <span className="text-[#8B8B94]">Matched incident: </span>
                                <span className="text-[#EDEDEF]">{result.trace.retrieved_incident.incident_id}</span>
                              </div>
                              <div>
                                <span className="text-[#8B8B94]">Score: </span>
                                <span className="text-[#EDEDEF] tabular-nums">
                                  {result.trace.similarity_score.toFixed(4)}
                                </span>
                              </div>
                            </div>
                            <div>
                              <span className="text-[#8B8B94]">Quality tier: </span>
                              <span className="text-[#EDEDEF]">
                                {passTier === 1 ? "Pass 1 (high quality)" : "Pass 2 (fallback tier)"}
                              </span>
                            </div>
                            {result.trace.retrieved_incident.source_url && (
                              <div className="pt-1 flex items-center gap-1.5 text-[11px]">
                                <span className="text-[#8B8B94]">Postmortem URL: </span>
                                <a
                                  href={result.trace.retrieved_incident.source_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[#6E7AE8] hover:underline flex items-center gap-1 truncate max-w-[340px]"
                                >
                                  {result.trace.retrieved_incident.source_url}
                                  <ExternalLink className="w-3 h-3 shrink-0" />
                                </a>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* 3. Prompt Payload */}
                        <div className="relative">
                          <span className="absolute -left-[27px] top-1 w-2 h-2 rounded-full bg-[#6E7AE8] ring-4 ring-[#111113]" />
                          <div className="text-[11px] font-mono text-[#8B8B94] uppercase tracking-wider font-semibold mb-1">
                            3. Prompt Assembly
                          </div>
                          <Collapsible open={promptOpen} onOpenChange={setPromptOpen}>
                            <div className="bg-[#0A0A0B] p-2.5 rounded-[4px] border border-[#26262B] space-y-2">
                              <div className="flex items-center justify-between text-[11px] font-mono">
                                <span className="text-[#8B8B94]">
                                  Payload length: {result.trace.prompt_used?.length.toLocaleString() || 0} chars
                                </span>
                                <div className="flex items-center gap-2">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => copyPrompt(result.trace.prompt_used)}
                                    className="h-5 px-1.5 text-[11px] text-[#8B8B94] hover:text-[#EDEDEF]"
                                  >
                                    <Copy className="w-3 h-3 mr-1" />
                                    Copy
                                  </Button>
                                  <CollapsibleTrigger className="inline-flex items-center justify-center h-5 px-1.5 text-[11px] text-[#8B8B94] hover:text-[#EDEDEF] hover:bg-[#17171A] rounded-[4px] font-mono transition-colors outline-none cursor-pointer">
                                    {promptOpen ? "Hide" : "Expand"}
                                  </CollapsibleTrigger>
                                </div>
                              </div>
                              <CollapsibleContent>
                                <ScrollArea className="max-h-[360px] w-full rounded border border-[#26262B] bg-[#111113] p-2.5 mt-2">
                                  <pre className="text-[11.5px] font-mono text-[#EDEDEF] whitespace-pre-wrap leading-relaxed">
                                    {result.trace.prompt_used}
                                  </pre>
                                </ScrollArea>
                              </CollapsibleContent>
                            </div>
                          </Collapsible>
                        </div>

                        {/* 4. Generation */}
                        <div className="relative">
                          <span className="absolute -left-[27px] top-1 w-2 h-2 rounded-full bg-[#6E7AE8] ring-4 ring-[#111113]" />
                          <div className="text-[11px] font-mono text-[#8B8B94] uppercase tracking-wider font-semibold mb-1">
                            4. LLM Generation
                          </div>
                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[12px] font-mono bg-[#0A0A0B] p-2.5 rounded-[4px] border border-[#26262B]">
                            <div>
                              <span className="text-[#8B8B94]">Model: </span>
                              <span className="text-[#EDEDEF]">{result.trace.model || knownModel}</span>
                            </div>
                            <div>
                              <span className="text-[#8B8B94]">Latency: </span>
                              <span className="text-[#EDEDEF] tabular-nums">{latencyMs ? `${latencyMs}ms` : "—"}</span>
                            </div>
                            <div>
                              <span className="text-[#8B8B94]">Mock mode: </span>
                              <span className={result.is_mock ? "text-[#E5A23B]" : "text-[#3FB67C]"}>
                                {result.is_mock ? "true" : "false"}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* 3. FOOTER (32px) */}
      <footer className="h-8 border-t border-[#26262B] bg-[#0A0A0B] px-4 flex items-center justify-between text-[12px] text-[#8B8B94] shrink-0 select-none">
        <span>TraceMind, NLP course project</span>
        <a
          href="https://github.com/nish-debug15/TraceMind"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-[#EDEDEF] transition-colors flex items-center gap-1"
        >
          <span>github.com/nish-debug15/TraceMind</span>
          <ExternalLink className="w-3 h-3" />
        </a>
      </footer>
    </div>
  );
}
