// frontend/src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import HomePage from "./pages/HomePage";
import AnalyzePage from "./pages/AnalyzePage";
import ResultsPage from "./pages/ResultsPage";
import RoadmapPage from "./pages/RoadmapPage";
import CaseChatPage from "./pages/CaseChatPage";
import Navbar from "./components/Navbar";
import "./styles/global.css";

function App() {
  return (
    <Router>
      <div className="app-root">
        <Navbar />
        <main className="app-main">
          <Routes>
            <Route path="/"         element={<HomePage />} />
            <Route path="/analyze"  element={<AnalyzePage />} />
            <Route path="/results"  element={<ResultsPage />} />
            <Route path="/roadmap"  element={<RoadmapPage />} />
            <Route path="/case-chat" element={<CaseChatPage />} />
          </Routes>
        </main>
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: "var(--surface)",
              color: "var(--text-primary)",
              border: "1px solid var(--border)",
              fontFamily: "var(--font-body)",
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
