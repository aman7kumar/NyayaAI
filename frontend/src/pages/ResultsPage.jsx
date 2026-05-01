// frontend/src/pages/ResultsPage.jsx
import React, { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import "./ResultsPage.css";

export default function ResultsPage() {
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("summary");

  useEffect(() => {
    const stored = sessionStorage.getItem("nyayaai_result");
    if (!stored) {
      navigate("/analyze");
      return;
    }
    setResult(JSON.parse(stored));
  }, [navigate]);

  if (!result) return null;

  const {
    query_type, detected_language, translated_query,
    entities, ipc_sections, crpc_sections, rag_context,
    explanation, roadmap, summary,
  } = result;

  const allSections = [...(ipc_sections || []), ...(crpc_sections || [])];

  const TABS = [
    { id: "summary",     label: "📋 Summary" },
    { id: "sections",    label: `⚖️ Sections (${allSections.length})` },
    { id: "explanation", label: "🔍 Explanation" },
    { id: "roadmap",     label: `🗺️ Roadmap (${roadmap?.length || 0} steps)` },
    { id: "entities",    label: "🏷️ Entities" },
  ];

  const domainLabels = {
    criminal: { label: "Criminal", color: "badge--red" },
    civil:    { label: "Civil",    color: "badge--blue" },
    consumer: { label: "Consumer", color: "badge--green" },
    family:   { label: "Family",   color: "badge--gold" },
    cyber:    { label: "Cyber",    color: "badge--navy" },
    default:  { label: "General",  color: "badge--gold" },
  };

  const domain = domainLabels[query_type] || domainLabels.default;

  return (
    <div className="results-page section">
      <div className="container">
        {/* ── Header ─────────────────────────────────────── */}
        <div className="results-header">
          <div className="results-header__meta">
            <span className={`badge ${domain.color}`}>{domain.label} Matter</span>
            <span className="badge badge--gold">
              {detected_language === "hi" ? "🇮🇳 Hindi Detected" : "🇬🇧 English"}
            </span>
          </div>
          <h1 className="results-title">Analysis Results</h1>
          {translated_query && (
            <div className="translated-note">
              <strong>Translated query:</strong> {translated_query}
            </div>
          )}
        </div>

        {/* ── Quick stats ─────────────────────────────────── */}
        <div className="results-stats">
          <div className="results-stat">
            <span className="results-stat__val">{ipc_sections?.length || 0}</span>
            <span className="results-stat__label">IPC Sections</span>
          </div>
          <div className="results-stat">
            <span className="results-stat__val">{crpc_sections?.length || 0}</span>
            <span className="results-stat__label">CrPC Sections</span>
          </div>
          <div className="results-stat">
            <span className="results-stat__val">{roadmap?.length || 0}</span>
            <span className="results-stat__label">Action Steps</span>
          </div>
          <div className="results-stat">
            <span className="results-stat__val">{rag_context?.length || 0}</span>
            <span className="results-stat__label">Laws Retrieved</span>
          </div>
        </div>

        {/* ── Tabs ────────────────────────────────────────── */}
        <div className="results-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`results-tab ${activeTab === t.id ? "results-tab--active" : ""}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* ── Tab content ─────────────────────────────────── */}
        <div className="results-content">

          {/* SUMMARY */}
          {activeTab === "summary" && (
            <div className="tab-panel">
              <div className="card card--gold summary-card">
                <h2 className="summary-card__title">⚖️ Legal Summary</h2>
                <div className="divider--gold divider" style={{ margin: "16px 0" }} />
                <div className="summary-text markdown-body">
                  <ReactMarkdown>
                    {summary || "Analysis complete. View other tabs for detailed results."}
                  </ReactMarkdown>
                </div>
              </div>
              <div className="summary-quick-actions">
                <button
                  className="btn btn--primary"
                  onClick={() => setActiveTab("roadmap")}
                >
                  See Your Action Roadmap →
                </button>
                <button
                  className="btn btn--outline"
                  onClick={() => setActiveTab("sections")}
                >
                  View Applicable Laws
                </button>
                <Link to="/analyze" className="btn btn--outline">
                  ← New Analysis
                </Link>
              </div>
            </div>
          )}

          {/* IPC / CrPC SECTIONS */}
          {activeTab === "sections" && (
            <div className="tab-panel">
              {allSections.length === 0 ? (
                <div className="empty-state">
                  No specific sections identified with high confidence.
                  Try providing more detail about the incident.
                </div>
              ) : (
                <div className="sections-grid">
                  {allSections.map((s, i) => (
                    <div className="section-card card" key={i}>
                      <div className="section-card__header">
                        <span className="section-card__tag badge badge--navy">{s.section}</span>
                        <span className="section-card__confidence">
                          {(s.confidence * 100).toFixed(0)}% match
                        </span>
                      </div>
                      <h3 className="section-card__title">{s.title}</h3>
                      <p className="section-card__desc">{s.description}</p>
                      {s.punishment && (
                        <div className="section-card__punishment">
                          <strong>⚖️ Punishment:</strong> {s.punishment}
                        </div>
                      )}
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${s.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* EXPLANATION */}
          {activeTab === "explanation" && (
            <div className="tab-panel">
              {explanation ? (
                <div className="card explanation-card">
                  <div className="explanation-text markdown-body">
                    <ReactMarkdown>{explanation}</ReactMarkdown>
                  </div>
                </div>
              ) : (
                <div className="empty-state">Explanation not generated for this analysis.</div>
              )}
            </div>
          )}

          {/* ROADMAP */}
          {activeTab === "roadmap" && (
            <div className="tab-panel">
              {!roadmap || roadmap.length === 0 ? (
                <div className="empty-state">No roadmap generated.</div>
              ) : (
                <div className="roadmap-timeline">
                  {roadmap.map((step, i) => (
                    <div className="roadmap-step" key={i}>
                      <div className="roadmap-step__num">{step.step_number}</div>
                      <div className="roadmap-step__body card">
                        <div className="roadmap-step__header">
                          <h3 className="roadmap-step__action">{step.action}</h3>
                          <span className="badge badge--blue">{step.timeline}</span>
                        </div>
                        <div className="roadmap-step__approach">
                          <strong>👤 Whom to approach:</strong> {step.whom_to_approach}
                        </div>
                        {step.documents_needed?.length > 0 && (
                          <div className="roadmap-step__docs">
                            <strong>📋 Documents needed:</strong>
                            <ul>
                              {step.documents_needed.map((d, j) => (
                                <li key={j}>{d}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {step.tips && (
                          <div className="roadmap-step__tips">
                            💡 {step.tips}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ENTITIES */}
          {activeTab === "entities" && (
            <div className="tab-panel">
              <div className="entities-grid">
                {Object.entries(entities || {}).map(([key, vals]) =>
                  Array.isArray(vals) && vals.length > 0 ? (
                    <div className="entity-group card" key={key}>
                      <h4 className="entity-group__title">
                        {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                      </h4>
                      <div className="entity-tags">
                        {vals.map((v, i) => (
                          <span key={i} className="badge badge--gold">{v}</span>
                        ))}
                      </div>
                    </div>
                  ) : null
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── RAG Context (collapsed by default) ──────────── */}
        <details className="rag-context">
          <summary>📚 Retrieved Legal Context ({rag_context?.length || 0} sources)</summary>
          {rag_context?.map((chunk, i) => (
            <div className="rag-chunk" key={i}>
              <span className="rag-chunk__num">{i + 1}</span>
              <p>{chunk}</p>
            </div>
          ))}
        </details>
      </div>
    </div>
  );
}
