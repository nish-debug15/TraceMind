"use client";

import { useState } from "react";

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
  prompt_used: string;
};

type AnalyzeResponse = {
  incident_id: string;
  raw_log_excerpt: string;
  root_cause: string;
  remediation_steps: string;
  source_url: string;
  trace: PipelineTrace;
  is_mock?: boolean;
};

export default function Home() {
  const [logText, setLogText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTrace, setShowTrace] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!logText.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Adjusted to use 127.0.0.1 to avoid IPv6 vs IPv4 localhost resolution issues on some machines
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${baseUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log_text: logText }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data: AnalyzeResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to fetch. Is the backend running on http://127.0.0.1:8000?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-12 py-16 px-6">
      <header className="flex flex-col md:flex-row md:items-baseline justify-between gap-4">
        <div>
          <h1 className="text-4xl font-serif text-[#332f2c] tracking-tight">TraceMind RCA</h1>
          <p className="text-[#6B655D] mt-2 font-sans text-sm tracking-wide uppercase">AI-Assisted Root Cause Analysis</p>
        </div>
        {result?.is_mock && (
          <div className="px-4 py-1.5 rounded bg-[#FDF4F2] text-[#B84B31] text-sm font-medium border border-[#F5D8D3] self-start md:self-auto">
            Mock Response — LLM not configured
          </div>
        )}
      </header>

      <section className="bg-white rounded-xl border border-[#EAE5D9] p-8 md:p-10">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="logInput" className="block text-sm font-medium text-[#4A4541] mb-3">
              Input Raw Log Excerpt
            </label>
            <textarea
              id="logInput"
              className="w-full h-40 p-4 border border-[#EAE5D9] rounded-lg bg-[#FAF9F5] text-[#332f2c] placeholder-[#9F9992] focus:ring-1 focus:ring-[#D97757] focus:border-[#D97757] outline-none transition-colors font-mono text-sm leading-relaxed resize-y"
              placeholder="Paste infrastructure logs or error traces here..."
              value={logText}
              onChange={(e) => setLogText(e.target.value)}
            />
          </div>
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading || !logText.trim()}
              className="bg-[#D97757] text-white px-8 py-2.5 rounded-md font-medium text-sm hover:bg-[#C26245] disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              {loading ? "Analyzing..." : "Analyze Incident"}
            </button>
          </div>
        </form>
        {error && (
          <div className="mt-6 p-5 text-[#B84B31] bg-[#FDF4F2] border border-[#F5D8D3] rounded-lg text-sm">
            {error}
          </div>
        )}
      </section>

      {loading && (
        <div className="text-center py-16 flex flex-col items-center justify-center space-y-4">
          <div className="inline-block animate-spin w-8 h-8 border-[3px] border-[#D97757] border-t-transparent rounded-full"></div>
          <p className="text-[#6B655D] text-sm font-medium tracking-wide">Retrieving historical context...</p>
        </div>
      )}

      {result && !loading && (
        <section className="space-y-8 animate-in fade-in duration-500">
          <div className="bg-white rounded-xl border border-[#EAE5D9] p-8 md:p-10 space-y-10">
            <h2 className="text-3xl font-serif text-[#332f2c] border-b border-[#EAE5D9] pb-4">Analysis Results</h2>
            
            <div className="space-y-4">
              <h3 className="text-sm font-semibold tracking-wide text-[#6B655D] uppercase">Root Cause</h3>
              <div className="text-[#332f2c] text-lg leading-relaxed whitespace-pre-wrap font-serif">
                {result.root_cause}
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-sm font-semibold tracking-wide text-[#6B655D] uppercase">Remediation Steps</h3>
              <div className="text-[#332f2c] text-lg leading-relaxed whitespace-pre-wrap font-serif">
                {result.remediation_steps}
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-[#EAE5D9] p-8 md:p-10">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-serif text-[#332f2c]">Pipeline Trace</h2>
              <button
                type="button"
                onClick={() => setShowTrace(!showTrace)}
                className="text-sm font-medium text-[#D97757] hover:text-[#C26245] bg-[#FDF9F7] hover:bg-[#FCEEEA] px-4 py-2 rounded-md border border-[#F5D8D3] transition-colors"
              >
                {showTrace ? "Hide Audit Trace" : "Show Audit Trace"}
              </button>
            </div>

            {showTrace && (
              <div className="mt-8 space-y-8 text-[#4A4541]">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-[#FAF9F5] p-5 rounded-lg border border-[#EAE5D9]">
                    <span className="block text-[#6B655D] text-xs font-semibold uppercase tracking-wider mb-2">Similarity Score</span>
                    <span className="font-mono text-[#4A4541] text-xl">
                      {result.trace.similarity_score.toFixed(4)}
                    </span>
                  </div>
                  <div className="bg-[#FAF9F5] p-5 rounded-lg border border-[#EAE5D9]">
                    <span className="block text-[#6B655D] text-xs font-semibold uppercase tracking-wider mb-2">Matched Incident ID</span>
                    <span className="font-mono text-[#4A4541] text-xl">
                      {result.trace.retrieved_incident.incident_id}
                    </span>
                  </div>
                </div>

                <div>
                  <h4 className="text-[#6B655D] text-xs font-semibold uppercase tracking-wider mb-3">Retrieved Historical Context</h4>
                  <div className="bg-[#FAF9F5] p-6 rounded-lg border border-[#EAE5D9] space-y-4 text-sm leading-relaxed">
                    <p><strong className="text-[#332f2c] font-medium">Root Cause:</strong> {result.trace.retrieved_incident.root_cause}</p>
                    <p><strong className="text-[#332f2c] font-medium">Remediation:</strong> {result.trace.retrieved_incident.remediation_steps}</p>
                    <div className="pt-2">
                      <strong className="text-[#332f2c] font-medium mr-2">Source:</strong>
                      <a 
                        href={result.trace.retrieved_incident.source_url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-[#D97757] hover:text-[#C26245] underline decoration-[#F5D8D3] underline-offset-4 break-all"
                      >
                        {result.trace.retrieved_incident.source_url}
                      </a>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-[#6B655D] text-xs font-semibold uppercase tracking-wider mb-3">Raw Prompt Payload</h4>
                  <pre className="bg-[#332f2c] text-[#EAE5D9] p-6 rounded-lg overflow-x-auto text-xs leading-relaxed font-mono whitespace-pre-wrap">
                    {result.trace.prompt_used}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
