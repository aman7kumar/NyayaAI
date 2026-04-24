"""
backend/models/roadmap_engine.py
==================================
Generates a personalized, step-by-step legal action roadmap for citizens.

Based on:
  - Detected query type (criminal / civil / consumer / family / cyber)
  - Predicted IPC/CrPC sections
  - Extracted entities (location, time, parties)

Roadmap includes:
  - Immediate steps
  - Whom to approach (Police / Magistrate / Consumer Forum / NHRC etc.)
  - Timeline for each action
  - Documents to carry
  - Legal aid contacts
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Roadmap Templates ──────────────────────────────────────────────────────────
# Each query type has a base roadmap that is customized at runtime.

ROADMAP_TEMPLATES: dict[str, list[dict]] = {

    "criminal": [
        {
            "step_number": 1,
            "action": "Ensure your safety first. Move to a safe location if the threat is active.",
            "whom_to_approach": "Emergency Services — Dial 112 (National Emergency)",
            "timeline": "Immediately",
            "documents_needed": [],
            "tips": "Call 112 immediately if you or someone is in danger. Police must respond.",
        },
        {
            "step_number": 2,
            "action": "File a First Information Report (FIR) at the nearest Police Station.",
            "whom_to_approach": "Local Police Station — Officer in Charge (SHO)",
            "timeline": "Within 24 hours of the incident",
            "documents_needed": [
                "Your Aadhar Card / ID proof",
                "Written complaint (can be verbal too — police must record it)",
                "Any evidence: photos, screenshots, witness names",
                "Medical report if injured",
            ],
            "tips": (
                "Police CANNOT refuse to register a cognizable offence FIR (CrPC 154). "
                "If they refuse, write to the Superintendent of Police or file a complaint "
                "before the Magistrate under CrPC 156(3)."
            ),
        },
        {
            "step_number": 3,
            "action": "Collect and preserve all evidence. Do not tamper with the crime scene.",
            "whom_to_approach": "Self / Trusted witnesses",
            "timeline": "As soon as possible after step 2",
            "documents_needed": [
                "Photographs / videos of the scene",
                "CCTV footage (request preservation immediately)",
                "Medical examination report",
                "Witness statements (names + contact numbers)",
            ],
            "tips": "CCTV footage is typically overwritten within 24-72 hours. Request preservation urgently.",
        },
        {
            "step_number": 4,
            "action": "If police are not acting, escalate to the Superintendent of Police (SP) or SP's office.",
            "whom_to_approach": "Superintendent of Police (SP) / Senior Police Officers",
            "timeline": "If no action within 3-5 days of FIR",
            "documents_needed": [
                "Copy of FIR (you have a legal right to a free copy — CrPC 154(2))",
                "Written complaint to SP",
            ],
            "tips": "You can also file an online complaint on your state police portal or the NCRB portal.",
        },
        {
            "step_number": 5,
            "action": "Consult a lawyer and consider filing a private complaint before the Magistrate.",
            "whom_to_approach": "Judicial Magistrate (via CrPC 200) or District & Sessions Court",
            "timeline": "If police file closure report or inaction continues beyond 2 weeks",
            "documents_needed": [
                "All previous FIR/complaint copies",
                "Evidence collected",
                "Legal representation (consult a lawyer)",
            ],
            "tips": "Free legal aid is available under the Legal Services Authorities Act. Contact DLSA.",
        },
    ],

    "civil": [
        {
            "step_number": 1,
            "action": "Send a formal Legal Notice to the opposing party via a lawyer.",
            "whom_to_approach": "Advocate / Lawyer — prepare and send via registered post",
            "timeline": "Within 1 week of the dispute",
            "documents_needed": [
                "All agreements / contracts",
                "Communication records (emails, messages)",
                "Payment receipts or proof of loss",
                "Your ID proof",
            ],
            "tips": "A legal notice often resolves disputes without going to court. Give 15-30 days for response.",
        },
        {
            "step_number": 2,
            "action": "Attempt mediation or Lok Adalat for faster, cost-free resolution.",
            "whom_to_approach": "District Legal Services Authority (DLSA) — Lok Adalat",
            "timeline": "After sending legal notice, within 30 days",
            "documents_needed": [
                "Copy of legal notice",
                "All supporting documents",
            ],
            "tips": "Lok Adalat awards are final and binding. No court fees. Fast resolution — same day often.",
        },
        {
            "step_number": 3,
            "action": "File a civil suit in the appropriate court.",
            "whom_to_approach": "Civil Court (Munsiff / Civil Judge) — based on claim amount",
            "timeline": "If mediation fails, within limitation period (usually 3 years)",
            "documents_needed": [
                "Plaint (complaint document prepared by lawyer)",
                "All evidence",
                "Court fee (based on claim amount)",
                "ID proof",
            ],
            "tips": "Claim up to ₹3 lakh → Munsiff Court. ₹3-20 lakh → Civil Judge. Above → District Court.",
        },
    ],

    "consumer": [
        {
            "step_number": 1,
            "action": "Send a complaint email/letter to the company's customer care and keep a record.",
            "whom_to_approach": "Company Customer Care / Grievance Officer",
            "timeline": "Immediately after the issue",
            "documents_needed": [
                "Purchase receipt / invoice",
                "Product/service description",
                "Communication records",
            ],
            "tips": "Keep all emails and chat transcripts. This creates an evidence trail.",
        },
        {
            "step_number": 2,
            "action": "File a complaint on the National Consumer Helpline portal.",
            "whom_to_approach": "National Consumer Helpline — 1800-11-4000 or consumerhelpline.gov.in",
            "timeline": "Within 1 week of company non-response",
            "documents_needed": [
                "Invoice / receipt",
                "Company complaint acknowledgment",
            ],
            "tips": "NCH mediates between consumer and company. Many cases resolved at this stage.",
        },
        {
            "step_number": 3,
            "action": "File a case before the Consumer Disputes Redressal Commission.",
            "whom_to_approach": "District Consumer Forum (claim ≤ ₹50 lakh) / State Commission / National Commission",
            "timeline": "Within 2 years of the cause of action",
            "documents_needed": [
                "Complaint form",
                "All purchase documents",
                "Evidence of deficiency in service",
                "Demand for compensation (specify amount)",
            ],
            "tips": (
                "Filing fee is minimal (₹50-₹200 for District Forum). "
                "No lawyer required — you can represent yourself."
            ),
        },
    ],

    "family": [
        {
            "step_number": 1,
            "action": "Seek mediation through family members or a trusted community elder.",
            "whom_to_approach": "Family / Community Mediation",
            "timeline": "Immediately",
            "documents_needed": [],
            "tips": "Informal resolution is faster and less stressful. Attempt this first.",
        },
        {
            "step_number": 2,
            "action": "Consult a family lawyer or legal aid authority.",
            "whom_to_approach": "Family Lawyer / District Legal Services Authority (DLSA)",
            "timeline": "Within 1-2 weeks",
            "documents_needed": [
                "Marriage certificate (for matrimonial disputes)",
                "Birth certificates (for custody disputes)",
                "Income documents (for maintenance claims)",
                "All relevant agreements",
            ],
            "tips": "Free legal aid is available for women, children, and economically weaker sections.",
        },
        {
            "step_number": 3,
            "action": "File a petition in the Family Court.",
            "whom_to_approach": "Family Court (available in most districts)",
            "timeline": "After consulting lawyer, within limitation period",
            "documents_needed": [
                "Petition (prepared by lawyer)",
                "All supporting documents",
                "ID proofs of all parties",
            ],
            "tips": "Family Courts are designed for faster resolution. Cases usually heard within 6-18 months.",
        },
    ],

    "cyber": [
        {
            "step_number": 1,
            "action": "Preserve all digital evidence immediately. Take screenshots.",
            "whom_to_approach": "Self — preserve evidence before reporting",
            "timeline": "Immediately",
            "documents_needed": [
                "Screenshots of offending content (with timestamps visible)",
                "URLs / links",
                "Email headers if relevant",
            ],
            "tips": "Do NOT delete any messages or block the person yet — this destroys evidence.",
        },
        {
            "step_number": 2,
            "action": "Report to the National Cyber Crime Reporting Portal.",
            "whom_to_approach": "cybercrime.gov.in — or Cyber Crime Cell of your district",
            "timeline": "Within 24 hours of discovering the crime",
            "documents_needed": [
                "All screenshots",
                "Device used (phone/laptop)",
                "Your email/account details",
            ],
            "tips": "For child-related cyber crimes, use the 'Report & Track' option on cybercrime.gov.in.",
        },
        {
            "step_number": 3,
            "action": "File FIR at local police station under IT Act / IPC sections.",
            "whom_to_approach": "Local Police Station — Cyber Crime Cell",
            "timeline": "Within 48-72 hours",
            "documents_needed": [
                "All digital evidence (screenshots, emails)",
                "Your ID proof",
                "Written complaint",
            ],
            "tips": "Relevant sections: IT Act 66, 66C, 66E, 67 and IPC 499, 503, 509.",
        },
    ],

    "dowry_harassment": [
        {
            "step_number": 1,
            "action": "Leave the unsafe environment. Go to a safe place — parent's home or shelter.",
            "whom_to_approach": "Family / Women's Shelter — call 181 (Women's Helpline)",
            "timeline": "Immediately if in danger",
            "documents_needed": [],
            "tips": "Dial 181 (Women Helpline) or 1091 (Police Women Helpline). These are 24/7 free services.",
        },
        {
            "step_number": 2,
            "action": "File a complaint at the nearest police station under IPC 498A and Dowry Prohibition Act.",
            "whom_to_approach": "Local Police Station — Women's Cell",
            "timeline": "As soon as safe to do so",
            "documents_needed": [
                "Marriage certificate",
                "Evidence of dowry demands (messages, witnesses)",
                "Medical report if physically harmed",
                "List of dowry items given",
            ],
            "tips": "Police must register FIR for IPC 498A. It is a cognizable and non-bailable offence.",
        },
        {
            "step_number": 3,
            "action": "Apply for protection order under Protection of Women from Domestic Violence Act, 2005.",
            "whom_to_approach": "Magistrate Court / Protection Officer (appointed in each district)",
            "timeline": "Simultaneously with police complaint",
            "documents_needed": [
                "Application form (DIR — Domestic Incident Report)",
                "Evidence of violence/harassment",
            ],
            "tips": "Protection Officer helps you file DIR for free. Magistrate can issue protection order same day.",
        },
    ],

    "default": [
        {
            "step_number": 1,
            "action": "Document everything related to your legal issue with dates, times, and details.",
            "whom_to_approach": "Self — create a written record",
            "timeline": "Immediately",
            "documents_needed": ["All relevant documents, receipts, messages, photos"],
            "tips": "A clear written timeline of events is invaluable for any legal proceeding.",
        },
        {
            "step_number": 2,
            "action": "Consult a lawyer for a professional assessment of your situation.",
            "whom_to_approach": "Lawyer / District Legal Services Authority (DLSA) for free legal aid",
            "timeline": "Within 1 week",
            "documents_needed": ["All documents related to the issue"],
            "tips": "Free legal aid is available for women, SC/ST, persons with disabilities, and people below poverty line.",
        },
        {
            "step_number": 3,
            "action": "File a complaint with the appropriate authority based on your issue.",
            "whom_to_approach": "Police / Court / Consumer Forum / Regulatory Authority",
            "timeline": "As advised by lawyer",
            "documents_needed": ["As per your lawyer's guidance"],
            "tips": "Always keep copies of every document you submit anywhere.",
        },
    ],
}


URGENCY_MAP = {
    "criminal": "immediate",
    "dowry_harassment": "immediate",
    "cyber": "immediate",
    "civil": "within_week",
    "consumer": "within_week",
    "family": "within_week",
    "default": "within_week",
}


LEGAL_AID_CONTACTS = [
    {
        "name": "National Legal Services Authority (NALSA)",
        "phone": "15100",
        "website": "nalsa.gov.in",
        "description": "Free legal aid for eligible citizens",
    },
    {
        "name": "Women's Helpline",
        "phone": "181",
        "website": "ncw.nic.in",
        "description": "24/7 helpline for women in distress",
    },
    {
        "name": "National Emergency",
        "phone": "112",
        "website": "112.gov.in",
        "description": "Police, Fire, Ambulance — unified emergency number",
    },
    {
        "name": "Cyber Crime Reporting",
        "phone": "1930",
        "website": "cybercrime.gov.in",
        "description": "Report cyber fraud and cyber crimes",
    },
    {
        "name": "Consumer Helpline",
        "phone": "1800-11-4000",
        "website": "consumerhelpline.gov.in",
        "description": "Consumer complaints and grievances",
    },
    {
        "name": "Child Helpline",
        "phone": "1098",
        "website": "childlineindia.org",
        "description": "Children in need of care and protection",
    },
]


class RoadmapEngine:
    """Generates personalized legal action roadmaps."""

    def generate_roadmap(
        self,
        query: str,
        query_type: str,
        entities: dict,
        ipc_sections: list[dict],
    ) -> list[dict]:
        """
        Generate a step-by-step roadmap based on query type and predicted sections.
        Customizes templates with entity/section-specific details.
        """
        # Determine template
        template_key = query_type.lower()

        # Special case: if IPC 498A predicted → use dowry_harassment
        predicted_keys = [s.get("label_key", "") for s in ipc_sections]
        if "IPC_498A" in predicted_keys:
            template_key = "dowry_harassment"

        template = ROADMAP_TEMPLATES.get(
            template_key,
            ROADMAP_TEMPLATES["default"]
        )

        # Deep copy and customize
        import copy
        steps = copy.deepcopy(template)

        # Inject predicted section info into relevant steps
        if ipc_sections:
            section_str = ", ".join(
                f"{s.get('section','')} — {s.get('title','')}"
                for s in ipc_sections[:3]
            )
            # Add section context to step 2 tips (FIR filing step usually)
            for step in steps:
                if "FIR" in step["action"] or "police" in step["action"].lower():
                    step["tips"] += f"\n\nApplicable sections: {section_str}."
                    break

        return steps

    def assess_urgency(self, query: str, query_type: str) -> str:
        """Return urgency level: 'immediate' / 'within_24h' / 'within_week'"""
        q = query.lower()
        if any(w in q for w in ["assault", "rape", "murder", "kidnap", "attack", "dying", "hurt"]):
            return "immediate"
        if any(w in q for w in ["threat", "harassment", "blackmail", "stalking"]):
            return "immediate"
        return URGENCY_MAP.get(query_type.lower(), "within_week")

    def get_legal_aid_contacts(self) -> list[dict]:
        return LEGAL_AID_CONTACTS
