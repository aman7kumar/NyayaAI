// frontend/src/pages/RoadmapPage.jsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./RoadmapPage.css";

const EMERGENCY_CONTACTS = [
  { name: "National Emergency",        phone: "112",           icon: "🚨", color: "#c0392b" },
  { name: "Women's Helpline",          phone: "181",           icon: "👩", color: "#8e44ad" },
  { name: "Police",                    phone: "100",           icon: "👮", color: "#2980b9" },
  { name: "Ambulance",                 phone: "102",           icon: "🚑", color: "#27ae60" },
  { name: "Cyber Crime",               phone: "1930",          icon: "💻", color: "#e67e22" },
  { name: "Legal Aid (NALSA)",         phone: "15100",         icon: "⚖️",  color: "#1a4a7a" },
  { name: "Consumer Helpline",         phone: "1800-11-4000",  icon: "🛒", color: "#16a085" },
  { name: "Child Helpline",            phone: "1098",          icon: "👶", color: "#d35400" },
  { name: "Domestic Violence",         phone: "181",           icon: "🏠", color: "#8e44ad" },
  { name: "Anti-Corruption (CVC)",     phone: "1964",          icon: "🏛️", color: "#2c3e50" },
];

const LEGAL_RIGHTS = [
  {
    icon: "📝",
    right: "Right to FIR Registration",
    detail: "Police CANNOT refuse to register an FIR for a cognizable offence. You can complain to the SP or approach a Magistrate under CrPC 156(3).",
    law: "CrPC Section 154",
  },
  {
    icon: "📄",
    right: "Right to Free FIR Copy",
    detail: "You are entitled to a free copy of the FIR immediately after registration.",
    law: "CrPC Section 154(2)",
  },
  {
    icon: "⚖️",
    right: "Right to Free Legal Aid",
    detail: "Every person has the right to free legal aid — available for women, SC/ST, persons with disabilities, and those below poverty line.",
    law: "Article 39A, Constitution + Legal Services Authorities Act",
  },
  {
    icon: "🔕",
    right: "Right to Remain Silent",
    detail: "You cannot be compelled to be a witness against yourself. You may remain silent when questioned.",
    law: "Article 20(3), Constitution",
  },
  {
    icon: "🏥",
    right: "Right to Medical Treatment",
    detail: "Government hospitals MUST provide free emergency treatment to all, regardless of police complaint status.",
    law: "Article 21, Constitution",
  },
  {
    icon: "🌐",
    right: "Right to Know Grounds of Arrest",
    detail: "If arrested, police must immediately inform you of the grounds of arrest and your right to bail.",
    law: "Article 22, Constitution + CrPC Section 50",
  },
];

export default function RoadmapPage() {
  const [result, setResult] = useState(null);
  const [roadmap, setRoadmap] = useState([]);

  useEffect(() => {
    const stored = sessionStorage.getItem("nyayaai_result");
    if (stored) {
      const parsed = JSON.parse(stored);
      setResult(parsed);
      setRoadmap(parsed.roadmap || []);
    }
  }, []);

  return (
    <div className="roadmap-page section">
      <div className="container">
        <div className="roadmap-header">
          <div className="ornament">✦ ✦ ✦</div>
          <h1 className="section-title" style={{ marginTop: 12 }}>Your Legal Action Roadmap</h1>
          <p className="section-subtitle">
            Step-by-step guidance on what to do, whom to approach, and when.
          </p>
          {!result && (
            <div className="alert-info">
              👆 <Link to="/analyze">Analyze your case first</Link> to get a personalized roadmap.
              Below is a general guide to your fundamental rights.
            </div>
          )}
        </div>

        {/* ── Emergency Contacts ───────────────────────────── */}
        <section className="roadmap-section">
          <h2 className="roadmap-section__title">🚨 Emergency Helpline Numbers</h2>
          <div className="emergency-grid">
            {EMERGENCY_CONTACTS.map((c) => (
              <a
                key={c.phone}
                href={`tel:${c.phone}`}
                className="emergency-card"
                style={{ "--accent": c.color }}
              >
                <span className="emergency-card__icon">{c.icon}</span>
                <span className="emergency-card__phone">{c.phone}</span>
                <span className="emergency-card__name">{c.name}</span>
              </a>
            ))}
          </div>
        </section>

        {/* ── Personalized roadmap (if available) ─────────── */}
        {roadmap.length > 0 && (
          <section className="roadmap-section">
            <h2 className="roadmap-section__title">🗺️ Your Personalized Action Plan</h2>
            <div className="roadmap-timeline">
              {roadmap.map((step, i) => (
                <div className="roadmap-item" key={i}>
                  <div className="roadmap-item__connector">
                    <div className="roadmap-item__dot">{step.step_number}</div>
                    {i < roadmap.length - 1 && <div className="roadmap-item__line" />}
                  </div>
                  <div className="roadmap-item__content card">
                    <div className="roadmap-item__header">
                      <h3>{step.action}</h3>
                      <span className="badge badge--blue">{step.timeline}</span>
                    </div>
                    <p className="roadmap-item__approach">
                      <strong>👤 Approach:</strong> {step.whom_to_approach}
                    </p>
                    {step.documents_needed?.length > 0 && (
                      <div className="roadmap-item__docs">
                        <strong>📋 Documents:</strong>
                        <ul>
                          {step.documents_needed.map((d, j) => <li key={j}>{d}</li>)}
                        </ul>
                      </div>
                    )}
                    {step.tips && (
                      <div className="roadmap-item__tip">💡 {step.tips}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ── Know Your Rights ─────────────────────────────── */}
        <section className="roadmap-section">
          <h2 className="roadmap-section__title">🏛️ Know Your Fundamental Rights</h2>
          <div className="rights-grid">
            {LEGAL_RIGHTS.map((r) => (
              <div className="right-card card" key={r.right}>
                <div className="right-card__icon">{r.icon}</div>
                <h3 className="right-card__title">{r.right}</h3>
                <p className="right-card__detail">{r.detail}</p>
                <span className="badge badge--gold right-card__law">{r.law}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ── CTA ──────────────────────────────────────────── */}
        <div className="roadmap-cta card card--gold">
          <div className="roadmap-cta__text">
            <h3>Need a personalized roadmap for your case?</h3>
            <p>Describe your specific situation and get tailored legal guidance.</p>
          </div>
          <Link to="/analyze" className="btn btn--primary btn--lg">
            Analyze My Case →
          </Link>
        </div>
      </div>
    </div>
  );
}
