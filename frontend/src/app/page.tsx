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
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="flex items-center justify-between">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">TraceMind RCA</h1>
          {result?.is_mock && (
            <div className="px-3 py-1 rounded-full bg-amber-100 text-amber-800 text-sm font-semibold border border-amber-300">
              Mock Response — LLM not configured
            </div>
          )}
        </header>

        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="logInput" className="block text-sm font-medium text-gray-700 mb-2">
                Raw Log Excerpt
              </label>
              <textarea
                id="logInput"
                className="w-full h-32 p-3 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                placeholder="Paste your raw infrastructure logs here..."
                value={logText}
                onChange={(e) => setLogText(e.target.value)}
              />
            </div>
            <button
              type="submit"
              disabled={loading || !logText.trim()}
              className="bg-blue-600 text-white px-6 py-2 rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Analyzing..." : "Analyze Incident"}
            </button>
          </form>
          {error && (
            <div className="mt-4 p-4 text-red-700 bg-red-50 border border-red-200 rounded-md">
              {error}
            </div>
          )}
        </section>

        {loading && (
          <div className="text-center p-12">
            <div className="inline-block animate-spin w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full"></div>
            <p className="mt-4 text-gray-600">Retrieving historical context and generating RCA...</p>
          </div>
        )}

        {result && !loading && (
          <section className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-6">
              <h2 className="text-2xl font-semibold text-gray-900 border-b pb-2">Analysis Results</h2>
              
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-2">Root Cause</h3>
                <div className="bg-gray-50 p-4 rounded-md border border-gray-100 whitespace-pre-wrap">
                  {result.root_cause}
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-2">Remediation Steps</h3>
                <div className="bg-gray-50 p-4 rounded-md border border-gray-100 whitespace-pre-wrap">
                  {result.remediation_steps}
                </div>
              </div>
            </div>

            <div className="bg-slate-900 rounded-xl shadow-sm border border-slate-700 text-slate-300 p-6 overflow-hidden">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">Pipeline Trace (Auditable RAG)</h2>
                <button
                  type="button"
                  onClick={() => setShowTrace(!showTrace)}
                  className="text-sm bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-1.5 rounded border border-slate-600 transition-colors"
                >
                  {showTrace ? "Hide Pipeline Trace" : "Show Pipeline Trace"}
                </button>
              </div>

              {showTrace && (
                <div className="space-y-6 mt-6 border-t border-slate-700 pt-6 text-sm">
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-slate-800 p-3 rounded border border-slate-700">
                      <span className="block text-slate-400 text-xs mb-1">Similarity Score</span>
                      <span className="font-mono text-emerald-400 text-lg">
                        {result.trace.similarity_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-slate-800 p-3 rounded border border-slate-700">
                      <span className="block text-slate-400 text-xs mb-1">Incident ID</span>
                      <span className="font-mono text-slate-200 text-lg">
                        {result.trace.retrieved_incident.incident_id}
                      </span>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-slate-400 mb-2 uppercase tracking-wider text-xs font-semibold">Retrieved Historical Incident</h4>
                    <div className="bg-slate-800 p-4 rounded border border-slate-700 space-y-3">
                      <p><strong className="text-slate-200">Root Cause:</strong> {result.trace.retrieved_incident.root_cause}</p>
                      <p><strong className="text-slate-200">Remediation:</strong> {result.trace.retrieved_incident.remediation_steps}</p>
                      <p>
                        <strong className="text-slate-200">Source:</strong>{" "}
                        <a 
                          href={result.trace.retrieved_incident.source_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300 underline break-all"
                        >
                          {result.trace.retrieved_incident.source_url}
                        </a>
                      </p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-slate-400 mb-2 uppercase tracking-wider text-xs font-semibold">Prompt Sent to LLM</h4>
                    <pre className="bg-slate-800 p-4 rounded border border-slate-700 whitespace-pre-wrap font-mono text-xs overflow-x-auto text-slate-300">
                      {result.trace.prompt_used}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
