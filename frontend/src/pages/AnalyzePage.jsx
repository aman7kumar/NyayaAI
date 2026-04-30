// frontend/src/pages/AnalyzePage.jsx
import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import axios from "axios";
import "./AnalyzePage.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const INPUT_MODES = [
  { id: "text",  label: "📝 Type Query",         desc: "Describe your situation in text" },
  { id: "image", label: "📷 Upload FIR Image",   desc: "Handwritten or scanned FIR" },
  { id: "pdf",   label: "📄 Upload PDF Document", desc: "Legal document or complaint PDF" },
];

export default function AnalyzePage() {
  const navigate = useNavigate();
  const [mode, setMode]         = useState("text");
  const [query, setQuery]       = useState("");
  const [language, setLanguage] = useState("auto");
  const [file, setFile]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [ocrText, setOcrText]   = useState("");
  const [ocrLoading, setOcrLoading] = useState(false);

  // ── Dropzone ────────────────────────────────────────────
  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) {
      setFile(accepted[0]);
      setOcrText("");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: mode === "image"
      ? { "image/*": [".jpg", ".jpeg", ".png", ".webp"] }
      : { "application/pdf": [".pdf"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  // ── OCR extraction ───────────────────────────────────────
  const handleOCR = async () => {
    if (!file) return;
    setOcrLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${API_BASE}/ocr`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Handle response
      const extractedText = res.data.raw_text || "";
      const engine        = res.data.engine_used || "unknown";
      const confidence    = res.data.confidence  || 0;
      const wordCount     = res.data.word_count  || 0;

      if (!extractedText || extractedText.trim().length === 0) {
        toast.error(
          "No text extracted. Try a clearer image with good lighting."
        );
        setOcrLoading(false);
        return;
      }

      setOcrText(extractedText);
      setQuery(extractedText);
      toast.success(
        `✅ OCR complete! Engine: ${engine} | ` +
        `Confidence: ${(confidence * 100).toFixed(0)}% | ` +
        `Words: ${wordCount}`
      );
    } catch (err) {
      const msg = err.response?.data?.detail || "OCR failed. Try a clearer image.";
      toast.error(msg);
    } finally {
      setOcrLoading(false);
    }
  };

  // ── Submit analysis ──────────────────────────────────────
 const handleAnalyze = async () => {
    const finalQuery = (query || "").trim();
    if (!finalQuery || finalQuery.length < 10) {
      toast.error("Please describe your situation in at least a few words.");
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/analyze`, {
        query: finalQuery,
        language,
        include_explanation: true,
        include_roadmap: true,
      });
      // Store result and navigate
      sessionStorage.setItem("nyayaai_result", JSON.stringify(res.data));
      navigate("/results");
    } catch (err) {
      const msg = err.response?.data?.detail || "Analysis failed. Is the backend running?";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyze-page section">
      <div className="container">
        <div className="analyze-header">
          <div className="ornament">✦ ✦ ✦</div>
          <h1 className="section-title" style={{ marginTop: 12 }}>Analyze Your Legal Situation</h1>
          <p className="section-subtitle">
            Describe your issue, upload a document, or paste FIR text.
            Our AI will identify applicable laws and guide you.
          </p>
        </div>

        <div className="analyze-layout">
          {/* ── Left: Input Form ──────────────────────────── */}
          <div className="analyze-form">
            {/* Mode selector */}
            <div className="mode-selector">
              {INPUT_MODES.map((m) => (
                <button
                  key={m.id}
                  className={`mode-btn ${mode === m.id ? "mode-btn--active" : ""}`}
                  onClick={() => { setMode(m.id); setFile(null); setOcrText(""); }}
                >
                  <span className="mode-btn__label">{m.label}</span>
                  <span className="mode-btn__desc">{m.desc}</span>
                </button>
              ))}
            </div>

            {/* Text input */}
            {mode === "text" && (
              <div className="form-group">
                <label className="form-label">
                  Describe your situation
                  <span className="form-hint">Hindi या English में लिखें</span>
                </label>
                <textarea
                  className="textarea"
                  placeholder={
                    "Examples:\n" +
                    "• My neighbour attacked me with a rod and I was injured.\n" +
                    "• Someone threatened to kill me if I don't give him money.\n" +
                    "• मेरे पड़ोसी ने मुझ पर हमला किया।\n" +
                    "• My employer has not paid my salary for 3 months."
                  }
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={8}
                />
                <div className="char-count">{query.length} characters</div>
              </div>
            )}

            {/* Image/PDF upload */}
            {(mode === "image" || mode === "pdf") && (
              <div className="form-group">
                <div
                  {...getRootProps()}
                  className={`dropzone ${isDragActive ? "dropzone--active" : ""} ${file ? "dropzone--filled" : ""}`}
                >
                  <input {...getInputProps()} />
                  {file ? (
                    <div className="dropzone__file">
                      <span className="dropzone__file-icon">
                        {mode === "image" ? "🖼️" : "📄"}
                      </span>
                      <span className="dropzone__file-name">{file.name}</span>
                      <span className="dropzone__file-size">
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                  ) : (
                    <div className="dropzone__placeholder">
                      <span className="dropzone__icon">{mode === "image" ? "📷" : "📄"}</span>
                      <p>Drag & drop here, or <strong>click to browse</strong></p>
                      <p className="dropzone__hint">
                        {mode === "image"
                          ? "JPG, PNG up to 10MB"
                          : "PDF up to 10MB"}
                      </p>
                    </div>
                  )}
                </div>

                {file && (
                  <button
                    className="btn btn--outline"
                    style={{ marginTop: 12 }}
                    onClick={handleOCR}
                    disabled={ocrLoading}
                  >
                    {ocrLoading ? (
                      <><span className="spinner" /> Extracting text…</>
                    ) : (
                      "🔍 Extract Text with OCR"
                    )}
                  </button>
                )}

                {ocrText && (
                  <div className="ocr-preview">
                    <div className="ocr-preview__label">Extracted Text (editable):</div>
                    <textarea
                      className="textarea"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      rows={6}
                    />
                  </div>
                )}
              </div>
            )}

            {/* Language selector */}
            <div className="form-group form-row">
              <div>
                <label className="form-label">Language</label>
                <select
                  className="input"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  <option value="auto">Auto-detect</option>
                  <option value="en">English</option>
                  <option value="hi">Hindi</option>
                </select>
              </div>
            </div>

            {/* Submit */}
            <button
              className="btn btn--gold btn--lg analyze-submit"
              onClick={handleAnalyze}
              disabled={loading || !query.trim()}
            >
              {loading ? (
                <><span className="spinner" /> Analyzing…</>
              ) : (
                "⚖️ Analyze My Case →"
              )}
            </button>
          </div>

          {/* ── Right: Info panel ─────────────────────────── */}
          <aside className="analyze-sidebar">
            <div className="card card--gold sidebar-card">
              <h3>🔒 Your Privacy</h3>
              <p>
                All analysis is performed on our local server. 
                No data is sent to OpenAI, Google, or any external service.
              </p>
            </div>

            <div className="card sidebar-card">
              <h3>⚠️ Disclaimer</h3>
              <p>
                NyayaAI provides legal <strong>information</strong>, not legal advice.
                Always consult a qualified lawyer before taking legal action.
              </p>
            </div>

            <div className="card sidebar-card">
              <h3>📞 Emergency Contacts</h3>
              <ul className="contact-list">
                {[
                  { num: "112",        label: "National Emergency" },
                  { num: "181",        label: "Women Helpline" },
                  { num: "1930",       label: "Cyber Crime" },
                  { num: "15100",      label: "Legal Aid (NALSA)" },
                  { num: "1800-11-4000", label: "Consumer Helpline" },
                ].map((c) => (
                  <li key={c.num} className="contact-item">
                    <a href={`tel:${c.num}`} className="contact-num">{c.num}</a>
                    <span className="contact-label">{c.label}</span>
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
