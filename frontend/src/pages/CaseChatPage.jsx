import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import toast from "react-hot-toast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./CaseChatPage.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";
const API_ROOT = API_BASE.replace(/\/api\/v1\/?$/, "");

const getCaseMatterLabel = (caseContext) => {
  const ipcCount = caseContext?.ipc_sections?.length || 0;
  const crpcCount = caseContext?.crpc_sections?.length || 0;
  if (ipcCount + crpcCount > 0) return "CRIMINAL";

  const queryType = (caseContext?.query_type || "").toLowerCase();
  if (queryType === "rights" || queryType === "constitutional") return "CONSTITUTIONAL / RIGHTS";

  const textBlob = [caseContext?.summary || "", caseContext?.explanation || "", caseContext?.translated_query || ""]
    .join(" ")
    .toLowerCase();
  if (/(article\s*14|article\s*19|article\s*21|constitution|fundamental right|writ)/i.test(textBlob)) {
    return "CONSTITUTIONAL / RIGHTS";
  }

  return (caseContext?.query_type || "general").toUpperCase();
};

export default function CaseChatPage() {
  const navigate = useNavigate();
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(true);
  const [caseContext, setCaseContext] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [firFile, setFirFile] = useState(null);
  const [firUploading, setFirUploading] = useState(false);
  const [supportsFirUpload, setSupportsFirUpload] = useState(false);
  const [supportsChatContext, setSupportsChatContext] = useState(false);
  const [attachedFirText, setAttachedFirText] = useState("");
  const [showExtractedPreview, setShowExtractedPreview] = useState(false);
  const [manualFirText, setManualFirText] = useState("");

  useEffect(() => {
    const bootstrap = async () => {
      const storedResult = sessionStorage.getItem("nyayaai_result");
      if (!storedResult) {
        toast.error("Analyze your case first before starting chat.");
        navigate("/analyze");
        return;
      }

      const parsed = JSON.parse(storedResult);
      setCaseContext(parsed);

      const existingSession = sessionStorage.getItem("nyayaai_case_chat_session");
      if (existingSession) {
        setSessionId(existingSession);
        try {
          const res = await axios.get(`${API_BASE}/chat/${existingSession}`);
          setMessages(res.data.messages || []);
        } catch (_err) {
          sessionStorage.removeItem("nyayaai_case_chat_session");
        } finally {
          setStarting(false);
        }
        return;
      }

      try {
        const res = await axios.post(`${API_BASE}/chat/start`, {
          caseContext: parsed,
        });
        const sid = res.data.sessionId;
        setSessionId(sid);
        sessionStorage.setItem("nyayaai_case_chat_session", sid);
      } catch (err) {
        toast.error(err.response?.data?.detail || "Failed to start chat session.");
      } finally {
        setStarting(false);
      }
    };
    bootstrap();
  }, [navigate]);

  useEffect(() => {
    const detectCapabilities = async () => {
      try {
        const res = await axios.get(`${API_ROOT}/openapi.json`);
        const paths = res.data?.paths || {};
        setSupportsFirUpload(Boolean(paths["/api/v1/fir/upload"]?.post));
        setSupportsChatContext(Boolean(paths["/api/v1/chat/context"]?.post));
      } catch (_err) {
        setSupportsFirUpload(false);
        setSupportsChatContext(false);
      }
    };
    detectCapabilities();
  }, []);

  const caseSummary = useMemo(() => {
    if (!caseContext) return "No case context loaded.";
    const sectionCount = (caseContext.ipc_sections?.length || 0) + (caseContext.crpc_sections?.length || 0);
    return `${getCaseMatterLabel(caseContext)} matter | ${sectionCount} likely sections`;
  }, [caseContext]);

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || !sessionId) return;

    const composedMessage =
      attachedFirText && !supportsChatContext
        ? `User question: ${message}\n\nAttached FIR context (OCR):\n${attachedFirText.slice(0, 3500)}`
        : message;

    setLoading(true);
    const optimistic = [...messages, { role: "user", content: message, timestamp: new Date().toISOString() }];
    setMessages(optimistic);
    setInput("");

    try {
      const res = await axios.post(`${API_BASE}/chat/message`, {
        sessionId,
        message: composedMessage,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.data.assistantMessage,
          timestamp: new Date().toISOString(),
        },
      ]);
      setSuggestions(res.data.suggestions || []);
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to send message.");
      setMessages(messages);
    } finally {
      setLoading(false);
    }
  };

  const handleFIRUpload = async () => {
    if (!firFile || !sessionId) return;
    setFirUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", firFile);
      let uploadRes;
      if (supportsFirUpload) {
        uploadRes = await axios.post(`${API_BASE}/fir/upload`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      } else {
        const ocrRes = await axios.post(`${API_BASE}/ocr`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        uploadRes = {
          data: {
            extracted_text: ocrRes.data.cleaned_text || ocrRes.data.raw_text || "",
            fir_analysis: {},
          },
        };
        toast("Using OCR fallback route. Restart backend to enable /fir/upload.", {
          duration: 4500,
          style: { background: "#fef9ec", color: "#7a5a00" },
        });
      }

      const extracted = uploadRes.data.extracted_text || "";
      setAttachedFirText(extracted);

      if (supportsChatContext) {
        await axios.post(`${API_BASE}/chat/context`, {
          sessionId,
          fir_text: extracted,
          fir_analysis: uploadRes.data.fir_analysis || {},
        });
        toast.success("FIR uploaded and added to chat context.");
      } else {
        toast("Context API unavailable on current backend. Using message fallback.", {
          duration: 4500,
          style: { background: "#fef9ec", color: "#7a5a00" },
        });
      }
      toast.success("FIR attached. Ask your question in the chat box.");
    } catch (err) {
      toast.error(err.response?.data?.detail || "FIR upload failed.");
    } finally {
      setFirUploading(false);
    }
  };

  const handleUseManualFIRText = async () => {
    if (!sessionId || !manualFirText.trim()) return;
    setAttachedFirText(manualFirText.trim());
    try {
      if (supportsChatContext) {
        await axios.post(`${API_BASE}/chat/context`, {
          sessionId,
          fir_text: manualFirText.trim(),
        });
        toast.success("FIR text added to chat context.");
      } else {
        toast("Context API unavailable on current backend. Using message fallback.", {
          duration: 4500,
          style: { background: "#fef9ec", color: "#7a5a00" },
        });
      }
      toast.success("Manual FIR text attached. Ask your question now.");
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to add FIR text.");
    }
  };

  if (starting) {
    return (
      <div className="case-chat-page section">
        <div className="container"><div className="card">Starting case chat...</div></div>
      </div>
    );
  }

  return (
    <div className="case-chat-page section">
      <div className="container">
        <div className="case-chat-header card card--gold">
          <h1 className="section-title" style={{ marginBottom: 8 }}>Case-Based Legal Assistant</h1>
          <p className="case-chat-summary">{caseSummary}</p>
          <p className="case-chat-warning">
            This is AI guidance, consult a lawyer for final action.
          </p>
        </div>

        <div className="case-chat-window card">
          <div className="fir-upload-panel">
            <div className="fir-upload-row">
              <input
                type="file"
                accept="image/*,.pdf"
                onChange={(e) => setFirFile(e.target.files?.[0] || null)}
              />
              <button
                className="btn btn--outline"
                onClick={handleFIRUpload}
                disabled={!firFile || firUploading}
              >
                {firUploading ? "Uploading FIR..." : "Upload FIR Image/PDF"}
              </button>
            </div>
            {attachedFirText && (
              <div className="fir-attached-banner">
                <span className="badge badge--green">FIR attached for context</span>
                <button className="btn btn--outline" onClick={() => setShowExtractedPreview((v) => !v)}>
                  {showExtractedPreview ? "Hide Extracted Text" : "Preview Extracted Text"}
                </button>
              </div>
            )}
            {showExtractedPreview && attachedFirText && (
              <div className="fir-preview">{attachedFirText}</div>
            )}
            <details>
              <summary>OCR unclear? Paste FIR text manually</summary>
              <div className="fir-manual-box">
                <textarea
                  className="textarea"
                  rows={4}
                  placeholder="Paste FIR text manually (optional)..."
                  value={manualFirText}
                  onChange={(e) => setManualFirText(e.target.value)}
                />
                <button className="btn btn--outline" onClick={handleUseManualFIRText} disabled={!manualFirText.trim()}>
                  Attach Manual FIR Text
                </button>
              </div>
            </details>
          </div>

          <div className="case-chat-messages">
            {messages.length === 0 && (
              <div className="case-chat-empty">
                Ask a follow-up about FIR details, next legal step, evidence strategy, or section clarification.
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={`${idx}-${msg.timestamp || ""}`} className={`chat-row ${msg.role === "user" ? "chat-row--user" : "chat-row--assistant"}`}>
                <div className={`chat-bubble ${msg.role === "user" ? "chat-bubble--user" : "chat-bubble--assistant"}`}>
                  {msg.role === "assistant" ? (
                    <div className="chat-markdown markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}
          </div>

          {suggestions.length > 0 && (
            <div className="chat-suggestions">
              {suggestions.slice(0, 2).map((s) => (
                <span key={s} className="badge badge--gold">{s}</span>
              ))}
            </div>
          )}

          <div className="chat-input-row">
            <textarea
              className="textarea"
              rows={3}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a case-specific follow-up..."
            />
            <button className="btn btn--primary" onClick={sendMessage} disabled={loading || !input.trim()}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
