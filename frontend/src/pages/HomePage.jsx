// frontend/src/pages/HomePage.jsx
import React from "react";
import { Link, useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import "./HomePage.css";

const FEATURES = [
  {
    icon: "📄",
    title: "FIR & Document Analysis",
    desc: "Upload handwritten FIRs, PDFs, or typed complaints. Our OCR engine extracts and understands the text.",
  },
  {
    icon: "⚖️",
    title: "IPC/CrPC Section Prediction",
    desc: "Our fine-tuned BERT model maps your situation to the correct sections of Indian Penal Code and CrPC.",
  },
  {
    icon: "🗺️",
    title: "Step-by-Step Roadmap",
    desc: "Know exactly what to do, whom to approach, what documents to carry, and when to act.",
  },
  {
    icon: "🔍",
    title: "Explainable AI",
    desc: "Every prediction comes with a full legal reasoning chain — no black-box answers.",
  },
  {
    icon: "🌐",
    title: "Hindi + English Support",
    desc: "Ask in Hindi or English. Our multilingual pipeline handles both scripts seamlessly.",
  },
  {
    icon: "🔒",
    title: "100% Local & Private",
    desc: "All models run on your own server. No data is sent to any external AI service.",
  },
];

const STEPS = [
  { num: "01", title: "Describe your situation", desc: "Type your legal query, paste FIR text, or upload a document/image." },
  { num: "02", title: "AI analyzes the case",    desc: "RAG retrieves relevant laws. BERT classifier predicts applicable IPC/CrPC sections." },
  { num: "03", title: "Receive your roadmap",    desc: "Get a plain-language summary, applicable laws, and a step-by-step action plan." },
];

export default function HomePage() {
  const navigate = useNavigate();

  const handleStartCaseChat = () => {
    const hasCaseContext = Boolean(sessionStorage.getItem("nyayaai_result"));
    if (hasCaseContext) {
      navigate("/case-chat");
      return;
    }
    toast("Please analyze your case first, then start chat.", {
      duration: 3500,
      style: { background: "#fef9ec", color: "#7a5a00" },
    });
    navigate("/analyze");
  };

  return (
    <div className="home">
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero__bg-pattern" aria-hidden="true" />
        <div className="container hero__content">
          <div className="hero__eyebrow">
            <span className="badge badge--gold">⚖ AI-Powered Legal Assistance</span>
          </div>
          <h1 className="hero__title">
            Understand Your Legal Rights<br />
            <span className="hero__title-accent">in Plain Language</span>
          </h1>
          <p className="hero__desc">
            NyayaAI bridges the gap between India's complex legal system and 
            the common citizen. Describe your situation — we'll tell you which laws apply, 
            what your rights are, and exactly what steps to take.
          </p>
          <div className="hero__actions">
            <Link to="/analyze" className="btn btn--gold btn--lg">
              Analyze My Case →
            </Link>
            <a href="#how-it-works" className="btn btn--outline btn--lg">
              How It Works
            </a>
          </div>
          <div className="hero__disclaimer">
            ⚠ For informational purposes only. Consult a lawyer for legal advice.
          </div>
        </div>
      </section>

      {/* ── Stats bar ────────────────────────────────────────── */}
      <div className="stats-bar">
        <div className="container stats-bar__inner">
          {[
            { val: "500+", label: "IPC/CrPC Sections Covered" },
            { val: "4",    label: "Legal Domains" },
            { val: "2",    label: "Languages (EN + HI)" },
            { val: "100%", label: "Private & Local" },
          ].map((s) => (
            <div className="stats-bar__item" key={s.label}>
              <span className="stats-bar__val">{s.val}</span>
              <span className="stats-bar__label">{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── How It Works ─────────────────────────────────────── */}
      <section className="section how-it-works" id="how-it-works">
        <div className="container">
          <div className="ornament">✦ ✦ ✦</div>
          <h2 className="section-title" style={{ textAlign: "center", marginTop: 16 }}>
            How NyayaAI Works
          </h2>
          <p className="section-subtitle" style={{ textAlign: "center", margin: "8px auto 48px" }}>
            Three simple steps to legal clarity
          </p>
          <div className="how-grid">
            {STEPS.map((step) => (
              <div className="how-card" key={step.num}>
                <div className="how-card__num">{step.num}</div>
                <h3 className="how-card__title">{step.title}</h3>
                <p className="how-card__desc">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────── */}
      <section className="section features" id="features">
        <div className="container">
          <h2 className="section-title">Everything You Need</h2>
          <p className="section-subtitle">
            A complete AI legal analysis system — no external APIs, fully self-trained.
          </p>
          <div className="case-chat-cta card card--gold">
            <div>
              <h3 className="case-chat-cta__title">Case-Based Chat Assistant</h3>
              <p className="case-chat-cta__desc">
                Continue your case with follow-up legal questions, clarifications, and guided next steps.
              </p>
            </div>
            <button className="btn btn--primary" onClick={handleStartCaseChat}>
              Start Case Chat
            </button>
          </div>
          <div className="features-grid">
            {FEATURES.map((f) => (
              <div className="feature-card card" key={f.title}>
                <div className="feature-card__icon">{f.icon}</div>
                <h3 className="feature-card__title">{f.title}</h3>
                <p className="feature-card__desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────── */}
      <section className="cta-section">
        <div className="container cta-section__inner">
          <h2>Ready to understand your rights?</h2>
          <p>Describe your situation and get instant AI-powered legal guidance.</p>
          <Link to="/analyze" className="btn btn--gold btn--lg">
            Start Free Analysis →
          </Link>
        </div>
      </section>
    </div>
  );
}
