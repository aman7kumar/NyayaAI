"""
backend/models/roadmap_engine.py
Generates personalized legal roadmaps for BOTH victims AND accused persons.
"""

from __future__ import annotations
import copy
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── VICTIM Roadmap Templates ───────────────────────────────────────────────────

VICTIM_ROADMAPS: dict[str, list[dict]] = {

    "criminal": [
        {
            "step_number": 1,
            "action": "Ensure your safety first. Move to a safe location if the threat is active.",
            "whom_to_approach": "Emergency Services — Dial 112 (National Emergency)",
            "timeline": "Immediately",
            "documents_needed": [],
            "tips": "Call 112 immediately if you or someone is in danger. Police must respond.",
            "type": "victim",
        },
        {
            "step_number": 2,
            "action": "File a First Information Report (FIR) at the nearest Police Station.",
            "whom_to_approach": "Local Police Station — Officer in Charge (SHO)",
            "timeline": "Within 24 hours of the incident",
            "documents_needed": [
                "Your Aadhar Card / ID proof",
                "Written complaint (can be verbal — police must record it)",
                "Any evidence: photos, screenshots, witness names",
                "Medical report if injured",
            ],
            "tips": "Police CANNOT refuse to register a cognizable offence FIR (CrPC 154). If they refuse, write to the SP or approach Magistrate under CrPC 156(3).",
            "type": "victim",
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
            "type": "victim",
        },
        {
            "step_number": 4,
            "action": "If police are not acting, escalate to the Superintendent of Police (SP).",
            "whom_to_approach": "Superintendent of Police (SP) / Senior Police Officers",
            "timeline": "If no action within 3-5 days of FIR",
            "documents_needed": [
                "Copy of FIR (you have a legal right to a free copy — CrPC 154(2))",
                "Written complaint to SP",
            ],
            "tips": "You can also file an online complaint on your state police portal or the NCRB portal.",
            "type": "victim",
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
            "tips": "Free legal aid is available under Legal Services Authorities Act. Contact DLSA (call 15100).",
            "type": "victim",
        },
    ],

    "consumer": [
        {
            "step_number": 1,
            "action": "Send a formal complaint email/letter to the company's Grievance Officer and keep records.",
            "whom_to_approach": "Company Customer Care / Grievance Officer",
            "timeline": "Immediately after the issue",
            "documents_needed": ["Purchase receipt / invoice", "Product description", "Communication records"],
            "tips": "Keep all emails and chat transcripts as evidence.",
            "type": "victim",
        },
        {
            "step_number": 2,
            "action": "File a complaint on the National Consumer Helpline portal.",
            "whom_to_approach": "National Consumer Helpline — 1800-11-4000 or consumerhelpline.gov.in",
            "timeline": "Within 1 week of company non-response",
            "documents_needed": ["Invoice/receipt", "Company complaint acknowledgment"],
            "tips": "NCH mediates between consumer and company. Many cases resolved at this stage.",
            "type": "victim",
        },
        {
            "step_number": 3,
            "action": "File a case before the Consumer Disputes Redressal Commission.",
            "whom_to_approach": "District Consumer Forum (claim up to Rs 50 lakh)",
            "timeline": "Within 2 years of the cause of action",
            "documents_needed": ["Complaint form", "All purchase documents", "Evidence of deficiency"],
            "tips": "Filing fee is minimal. No lawyer required — you can represent yourself.",
            "type": "victim",
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
            "type": "victim",
        },
        {
            "step_number": 2,
            "action": "Consult a family lawyer or the District Legal Services Authority.",
            "whom_to_approach": "Family Lawyer / DLSA (call 15100)",
            "timeline": "Within 1-2 weeks",
            "documents_needed": ["Marriage certificate", "Birth certificates", "Income documents"],
            "tips": "Free legal aid is available for women, children, and economically weaker sections.",
            "type": "victim",
        },
        {
            "step_number": 3,
            "action": "File a petition in the Family Court.",
            "whom_to_approach": "Family Court (available in most districts)",
            "timeline": "After consulting lawyer",
            "documents_needed": ["Petition", "All supporting documents", "ID proofs"],
            "tips": "Family Courts are designed for faster resolution.",
            "type": "victim",
        },
    ],

    "dowry_harassment": [
        {
            "step_number": 1,
            "action": "Leave the unsafe environment immediately. Go to a safe place.",
            "whom_to_approach": "Family / Women's Shelter — call 181 (Women's Helpline)",
            "timeline": "Immediately if in danger",
            "documents_needed": [],
            "tips": "Dial 181 (Women Helpline) or 1091 (Police Women Helpline). Available 24/7.",
            "type": "victim",
        },
        {
            "step_number": 2,
            "action": "File a complaint at the nearest police station under IPC 498A and Dowry Prohibition Act.",
            "whom_to_approach": "Local Police Station — Women's Cell",
            "timeline": "As soon as safe to do so",
            "documents_needed": ["Marriage certificate", "Evidence of dowry demands", "Medical report if harmed"],
            "tips": "Police must register FIR for IPC 498A. It is a cognizable and non-bailable offence.",
            "type": "victim",
        },
        {
            "step_number": 3,
            "action": "Apply for protection order under Protection of Women from Domestic Violence Act 2005.",
            "whom_to_approach": "Magistrate Court / Protection Officer (appointed in each district)",
            "timeline": "Simultaneously with police complaint",
            "documents_needed": ["Application form (DIR)", "Evidence of violence/harassment"],
            "tips": "Protection Officer helps you file DIR for free. Magistrate can issue protection order same day.",
            "type": "victim",
        },
    ],

    "cyber": [
        {
            "step_number": 1,
            "action": "Preserve all digital evidence immediately. Take screenshots of everything.",
            "whom_to_approach": "Self — preserve evidence before reporting",
            "timeline": "Immediately",
            "documents_needed": ["Screenshots with timestamps", "URLs/links", "Email headers if relevant"],
            "tips": "Do NOT delete any messages or block the person yet — this destroys evidence.",
            "type": "victim",
        },
        {
            "step_number": 2,
            "action": "Report to the National Cyber Crime Reporting Portal.",
            "whom_to_approach": "cybercrime.gov.in or Cyber Crime Cell — call 1930",
            "timeline": "Within 24 hours",
            "documents_needed": ["All screenshots", "Device details", "Your account details"],
            "tips": "For financial fraud, call 1930 immediately — banks can freeze fraudulent transactions.",
            "type": "victim",
        },
        {
            "step_number": 3,
            "action": "File FIR at local police station under IT Act / IPC sections.",
            "whom_to_approach": "Local Police Station — Cyber Crime Cell",
            "timeline": "Within 48-72 hours",
            "documents_needed": ["All digital evidence", "Your ID proof", "Written complaint"],
            "tips": "Relevant sections: IT Act 66, 66C, 66E, 67 and IPC 499, 503, 509.",
            "type": "victim",
        },
    ],
}


# ── ACCUSED Roadmap Templates ──────────────────────────────────────────────────

ACCUSED_ROADMAPS: dict[str, list[dict]] = {

    "criminal_accused": [
        {
            "step_number": 1,
            "action": "DO NOT speak to police without a lawyer present. Exercise your right to silence.",
            "whom_to_approach": "Do not approach police alone — contact a lawyer FIRST",
            "timeline": "IMMEDIATELY — before any police interaction",
            "documents_needed": [],
            "tips": (
                "Under Article 20(3) of the Constitution, you cannot be compelled to be a witness "
                "against yourself. You have the right to remain silent. Anything you say CAN and "
                "WILL be used against you in court. Say only: 'I want to speak to my lawyer first.'"
            ),
            "warning": "DO NOT confess, even partially. Even a partial admission is admissible in court.",
            "type": "accused",
        },
        {
            "step_number": 2,
            "action": "Immediately contact a criminal defence lawyer. This is your most important step.",
            "whom_to_approach": "Criminal Defence Lawyer / District Legal Services Authority (DLSA) for free legal aid",
            "timeline": "Within the next 1-2 hours",
            "documents_needed": [
                "Any documents related to the alleged incident",
                "Your ID proof (Aadhar Card)",
                "Contact details of witnesses who can support you",
            ],
            "tips": (
                "Free legal aid is available if you cannot afford a lawyer (call DLSA: 15100). "
                "Your lawyer can: negotiate bail, challenge evidence, prepare your defence, "
                "and advise you on whether to cooperate with police."
            ),
            "warning": "Do not hire a lawyer recommended by police — they may not act in your interest.",
            "type": "accused",
        },
        {
            "step_number": 3,
            "action": "Understand your arrest rights if police come to arrest you.",
            "whom_to_approach": "Know your rights — inform a family member immediately upon arrest",
            "timeline": "If police arrive for arrest",
            "documents_needed": [],
            "tips": (
                "Under CrPC Section 50, police MUST inform you: (1) The grounds of arrest; "
                "(2) Your right to bail (if bailable offence); (3) Your right to be produced before "
                "a Magistrate within 24 hours (Article 22, Constitution). "
                "You have the right to inform a family member or friend about your arrest (CrPC 50A)."
            ),
            "warning": (
                "Do not resist arrest physically — this adds Section 353 IPC (assault on public servant). "
                "Cooperate physically but stay silent legally."
            ),
            "type": "accused",
        },
        {
            "step_number": 4,
            "action": "Apply for anticipatory bail immediately if you fear arrest (Section 438 CrPC).",
            "whom_to_approach": "Sessions Court or High Court — through your lawyer",
            "timeline": "Before arrest — as soon as possible",
            "documents_needed": [
                "Anticipatory bail application (prepared by lawyer)",
                "Your ID and address proof",
                "Evidence of your good conduct / character certificates",
                "Surety details (person who will stand guarantee)",
            ],
            "tips": (
                "Anticipatory bail protects you from arrest. If granted, police cannot arrest you "
                "without court permission. This is crucial if the offence is non-bailable."
            ),
            "warning": "This must be filed BEFORE arrest. After arrest, you need regular bail (Section 437/439 CrPC).",
            "type": "accused",
        },
        {
            "step_number": 5,
            "action": "After arrest: Apply for regular bail at the appropriate court.",
            "whom_to_approach": "Magistrate Court (bailable offences) or Sessions Court (non-bailable)",
            "timeline": "Within 24 hours of arrest — you must be produced before Magistrate",
            "documents_needed": [
                "Bail application (prepared by lawyer)",
                "ID and address proof",
                "Surety / bail bond",
                "Evidence of ties to community (job, family, property)",
            ],
            "tips": (
                "For bailable offences (like IPC 379-theft): bail is a right, police must grant it. "
                "For non-bailable offences (like IPC 302-murder): only court can grant bail. "
                "Bail conditions typically include: not leaving city, reporting to police weekly, "
                "not contacting witnesses."
            ),
            "type": "accused",
        },
        {
            "step_number": 6,
            "action": "Prepare your defence — understand the charges and build your case with your lawyer.",
            "whom_to_approach": "Criminal Defence Lawyer",
            "timeline": "Ongoing — from day 1 until case concludes",
            "documents_needed": [
                "Copy of FIR (you have a right to access it)",
                "Charge sheet when filed by police",
                "All evidence in your favour (alibi, CCTV, witnesses)",
                "Character witnesses",
            ],
            "tips": (
                "Your lawyer will: Challenge the FIR if false/exaggerated; Cross-examine witnesses; "
                "File for discharge before framing of charges; Negotiate settlement/plea if appropriate. "
                "The prosecution must PROVE guilt beyond reasonable doubt — you do not need to prove innocence."
            ),
            "warning": (
                "Do NOT destroy evidence, tamper with witnesses, or flee. These are additional offences "
                "(IPC 201 — causing disappearance of evidence, IPC 195A — threatening witness) "
                "and will severely damage your case."
            ),
            "type": "accused",
        },
    ],

    "theft_accused": [
        {
            "step_number": 1,
            "action": "Do NOT admit to anything. Exercise your right to silence immediately.",
            "whom_to_approach": "Contact a lawyer before speaking to anyone",
            "timeline": "Immediately",
            "documents_needed": [],
            "tips": "Say only: 'I want to speak to my lawyer.' Nothing else.",
            "warning": "A confession made to police is NOT admissible in court (Indian Evidence Act Section 25), but statements made to a Magistrate are. Do not go to Magistrate without a lawyer.",
            "type": "accused",
        },
        {
            "step_number": 2,
            "action": "Hire a criminal defence lawyer immediately.",
            "whom_to_approach": "Criminal Lawyer / DLSA for free legal aid (call 15100)",
            "timeline": "Within 1-2 hours",
            "documents_needed": ["Your ID proof", "Details of the alleged incident", "Names of witnesses in your favour"],
            "tips": "For IPC 379 (theft) — it is cognizable and non-bailable. Police can arrest without warrant. Bail can be obtained from court.",
            "type": "accused",
        },
        {
            "step_number": 3,
            "action": "Apply for anticipatory bail before arrest, or regular bail after arrest.",
            "whom_to_approach": "Magistrate Court through your lawyer",
            "timeline": "Before arrest: anticipatory bail. After arrest: within 24 hours",
            "documents_needed": ["Bail application", "ID proof", "Surety details"],
            "tips": (
                "Theft (IPC 379) is punishable up to 3 years. First-time offenders often get bail. "
                "If you return the stolen property, this may help in bail and sentencing."
            ),
            "warning": "Do not flee or hide — this creates additional charges and damages your bail application.",
            "type": "accused",
        },
        {
            "step_number": 4,
            "action": "Explore settlement / compensation to the victim through your lawyer.",
            "whom_to_approach": "Through lawyer — civil settlement or Lok Adalat",
            "timeline": "After consulting lawyer",
            "documents_needed": ["Settlement agreement", "Proof of compensation paid"],
            "tips": (
                "In property offences like theft, if stolen property is returned and compensation paid, "
                "courts may take a lenient view during sentencing. Compounding (compromise) is possible "
                "in some offences with the victim's consent."
            ),
            "warning": "Never approach the victim directly to settle — this may be seen as witness tampering (IPC 195A).",
            "type": "accused",
        },
        {
            "step_number": 5,
            "action": "Prepare your defence with your lawyer — challenge evidence and prosecution case.",
            "whom_to_approach": "Criminal Defence Lawyer",
            "timeline": "Ongoing",
            "documents_needed": ["FIR copy", "Charge sheet", "Evidence in your favour", "Alibi if any"],
            "tips": (
                "Your lawyer will examine: Was the identification of accused proper? "
                "Was evidence collected following proper procedure? "
                "Were there any witnesses who can support your version? "
                "Prosecution must prove guilt beyond reasonable doubt."
            ),
            "type": "accused",
        },
    ],

    "civil_accused": [
        {
            "step_number": 1,
            "action": "Read the legal notice carefully and do not ignore it.",
            "whom_to_approach": "Consult a civil lawyer immediately",
            "timeline": "Within 24-48 hours of receiving notice",
            "documents_needed": ["The legal notice received", "All documents related to the dispute"],
            "tips": "Ignoring a legal notice does not make it go away — it can lead to court proceedings against you.",
            "type": "accused",
        },
        {
            "step_number": 2,
            "action": "Consult a civil/commercial lawyer and prepare a reply to the legal notice.",
            "whom_to_approach": "Civil Defence Lawyer",
            "timeline": "Within the time period mentioned in the notice (usually 15-30 days)",
            "documents_needed": ["Legal notice", "Contracts/agreements", "Payment records", "Communication records"],
            "tips": "A well-drafted reply preserves your legal position and shows good faith.",
            "type": "accused",
        },
        {
            "step_number": 3,
            "action": "Explore mediation or settlement before the matter reaches court.",
            "whom_to_approach": "Lok Adalat / Mediation Centre / through lawyers",
            "timeline": "Before court filing",
            "documents_needed": ["Settlement terms", "Evidence supporting your position"],
            "tips": "Settlement saves time, money, and reputation. Lok Adalat awards are final and binding with no court fees.",
            "type": "accused",
        },
        {
            "step_number": 4,
            "action": "If case goes to court, file a written statement defending your position.",
            "whom_to_approach": "Civil Court through your lawyer",
            "timeline": "Within 30 days of summons (can be extended)",
            "documents_needed": ["Written statement prepared by lawyer", "All supporting evidence"],
            "tips": "You have the right to present your side. The burden of proof is on the plaintiff (the person suing you).",
            "type": "accused",
        },
    ],
}


# ── Pros/Cons Awareness ────────────────────────────────────────────────────────

ACCUSED_PROS_CONS = {
    "cooperation": {
        "pros": [
            "May get bail more easily if cooperative",
            "Court may view favourably during sentencing",
            "Avoids additional charges like obstruction of justice",
        ],
        "cons": [
            "Statements may be used against you",
            "Over-cooperation without legal advice can harm defence",
        ],
    },
    "confession": {
        "pros": [
            "May reduce sentence if genuine remorse shown",
            "Can help negotiate plea bargain (Section 265B CrPC)",
        ],
        "cons": [
            "Cannot be taken back once made before Magistrate",
            "Directly proves guilt — prosecution's job becomes easy",
            "May lead to maximum punishment",
        ],
    },
    "fleeing": {
        "pros": ["Short-term avoidance of arrest"],
        "cons": [
            "Makes you look guilty to court",
            "Additional offence: non-bailable warrant issued",
            "Property can be attached under Section 83 CrPC",
            "Bail becomes very difficult to obtain",
            "Adds Section 174A IPC (failure to appear) charge",
        ],
    },
    "settlement": {
        "pros": [
            "Faster resolution",
            "Avoids criminal record in compoundable offences",
            "Saves legal costs and time",
            "Less stress and uncertainty",
        ],
        "cons": [
            "Must pay compensation to victim",
            "Not all offences can be compounded (IPC 320)",
            "Court must approve the settlement",
        ],
    },
}


# ── Urgency Map ────────────────────────────────────────────────────────────────

URGENCY_MAP = {
    "criminal_accused":  "immediate",
    "theft_accused":     "immediate",
    "civil_accused":     "within_week",
    "criminal":          "immediate",
    "dowry_harassment":  "immediate",
    "cyber":             "immediate",
    "civil":             "within_week",
    "consumer":          "within_week",
    "family":            "within_week",
    "default":           "within_week",
}

LEGAL_AID_CONTACTS = [
    {"name": "National Legal Services Authority (NALSA)", "phone": "15100",
     "website": "nalsa.gov.in", "description": "Free legal aid for eligible citizens"},
    {"name": "Women's Helpline",    "phone": "181",
     "website": "ncw.nic.in",   "description": "24/7 helpline for women in distress"},
    {"name": "National Emergency",  "phone": "112",
     "website": "112.gov.in",   "description": "Police, Fire, Ambulance"},
    {"name": "Cyber Crime",         "phone": "1930",
     "website": "cybercrime.gov.in", "description": "Report cyber fraud"},
    {"name": "Consumer Helpline",   "phone": "1800-11-4000",
     "website": "consumerhelpline.gov.in", "description": "Consumer complaints"},
    {"name": "Child Helpline",      "phone": "1098",
     "website": "childlineindia.org", "description": "Children in need"},
]


# ── Main Engine ────────────────────────────────────────────────────────────────

class RoadmapEngine:

    def generate_roadmap(
        self,
        query:        str,
        query_type:   str,
        entities:     dict,
        ipc_sections: list[dict],
        user_role:    str = "victim",
    ) -> list[dict]:
        """
        Generate roadmap based on user role.
        If accused → lawyer-style defence roadmap with pros/cons.
        If victim  → action roadmap to seek justice.
        """
        predicted_keys = [s.get("label_key", "") for s in ipc_sections]

        if user_role == "accused":
            return self._accused_roadmap(query, query_type, predicted_keys)
        else:
            return self._victim_roadmap(query, query_type, predicted_keys)

    def _victim_roadmap(
        self,
        query:          str,
        query_type:     str,
        predicted_keys: list[str],
    ) -> list[dict]:
        """Standard victim roadmap."""
        template_key = query_type.lower()

        if "IPC_498A" in predicted_keys:
            template_key = "dowry_harassment"

        template = VICTIM_ROADMAPS.get(template_key, VICTIM_ROADMAPS["criminal"])
        steps    = copy.deepcopy(template)

        if ipc_sections := [s for s in predicted_keys if s]:
            section_str = ", ".join(
                s.replace("_", " ").replace("IPC ", "IPC ").replace("CrPC ", "CrPC ")
                for s in ipc_sections[:3]
            )
            for step in steps:
                if "FIR" in step["action"] or "police" in step["action"].lower():
                    step["tips"] += f"\n\nApplicable sections: {section_str}."
                    break

        return steps

    def _accused_roadmap(
        self,
        query:          str,
        query_type:     str,
        predicted_keys: list[str],
    ) -> list[dict]:
        """Defence roadmap for accused persons."""
        if any(k in predicted_keys for k in ["IPC_379", "IPC_380", "IPC_392", "IPC_395"]):
            template_key = "theft_accused"
        elif query_type in ("civil",):
            template_key = "civil_accused"
        else:
            template_key = "criminal_accused"

        steps = copy.deepcopy(ACCUSED_ROADMAPS.get(template_key, ACCUSED_ROADMAPS["criminal_accused"]))

        steps.append({
            "step_number": len(steps) + 1,
            "action": "Understand the pros and cons of each decision you make in this case.",
            "whom_to_approach": "Discuss all options with your lawyer before deciding",
            "timeline": "Before taking any action",
            "documents_needed": [],
            "tips": self._format_pros_cons(),
            "warning": "Every decision in a criminal case has consequences. Make informed choices with legal guidance.",
            "type": "accused",
        })

        return steps

    def _format_pros_cons(self) -> str:
        lines = ["Key decisions and their implications:"]
        for decision, data in ACCUSED_PROS_CONS.items():
            lines.append(f"\n{decision.upper()}:")
            lines.append("  Pros: " + "; ".join(data["pros"]))
            lines.append("  Cons: " + "; ".join(data["cons"]))
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # detect_user_role
    # ──────────────────────────────────────────────────────────────────────────
    def detect_user_role(self, query: str) -> str:
        """
        Detect whether the narrator is the victim or the accused.

        Strategy (highest priority first):
        1. Hard accused-posture regex  → weight 5 each  (legal reality: FIR/case against me)
        2. Hard victim-posture regex   → weight 5 each  (I filed FIR / I am victim)
        3. Keyword accused signals     → weight 2 each
        4. Keyword victim signals      → weight 2 each
        5. Innocence cues ONLY reduce
           keyword-level accused score, never posture score.

        Returns "accused" or "victim".
        """
        text = query.lower()

        # ── 1. Hard posture patterns ────────────────────────────────────────
        # These override everything — they directly state the narrator's legal
        # position regardless of guilt or innocence.

        ACCUSED_POSTURE_PATTERNS = [
            # FIR / case / complaint ON or AGAINST me
            r"\b(fir|case|complaint|charge|charges)\s[\w\s,]{0,30}(against|on)\s[\w\s]{0,10}me\b",
            r"\b(filed|registered|lodged|put)\s[\w\s]{0,20}(fir|case|complaint)\s[\w\s]{0,20}(against|on)\s[\w\s]{0,10}me\b",
            # false / fake case against me
            r"\b(false|fake|fabricated|wrong|झूठा|झूठी)\s[\w\s]{0,12}(fir|case|complaint|maamla)\b",
            # bail / anticipatory bail
            r"\b(anticipatory\s+bail|regular\s+bail|bail\s+application|apply\s+for\s+bail|need\s+bail|get\s+bail)\b",
            # summoned / arrested / accused / respondent
            r"\b(summons\s+received|i\s+am\s+accused|i\s+am\s+arrested|arrested\s+me|police\s+arrested|i\s+was\s+arrested|i\s+have\s+been\s+arrested)\b",
            r"\b(respondent|accused\s+person|the\s+accused)\b",
            # Hindi posture signals
            r"\b(mere\s+khilaf|mujh\s+par\s+case|mujhpe\s+case|mujhe\s+pakda|police\s+ne\s+pakda)\b",
            # ADD this entry to ACCUSED_POSTURE_PATTERNS list:
            r"\b(i|we)\s+(manipulated|diverted|embezzled|misappropriated|falsified|altered|forged|"
            r"deleted|siphoned|laundered|bribed|defrauded|created\s+fake|transferred\s+money|"
            r"opened\s+fake|made\s+fake)\b",
            # ADD this — catches first-person past-tense financial crime narratives
            r"\b(i|we)\s+[\w\s]{0,20}(manipulated|diverted|embezzled|falsified|"
            r"misappropriated|siphoned|laundered|defrauded)\b",

            # ADD this — catches "the fraud remained", "the scheme", self-authored crime story
            r"\b(the\s+fraud|the\s+scheme|my\s+scheme|my\s+fraud)\b",

            # ADD this — audit discovery framing (narrator is the one caught)
            r"\b(audit|auditors?)\s+[\w\s]{0,30}(discovered|identified|found)\b",
            # ── NEW: physical harm / accident caused by narrator ─────────────────────
            # Hit-and-run, drunk driving, vehicular assault
            r"\bstruck\s+[\w\s]{0,15}(person|rider|pedestrian|cyclist|man|woman|child|victim|someone|delivery)\b",
            r"\b(i|we)\s+[\w\s]{0,15}(drove\s+away|ran\s+away|fled\s+the\s+scene|left\s+the\s+scene|escaped\s+from|fled\s+from)\b",
            r"\b(i\s+feared\s+arrest|fear(ed)?\s+of\s+arrest|to\s+avoid\s+arrest|fearing\s+arrest)\b",
            r"\b(despite\s+being\s+intoxicated|drunk(en)?\s+driv(ing|e)|driving\s+(after|while|under)\s+(drinking|intoxicated|drunk|influence))\b",
            r"\b(hit\s+and\s+run|drove\s+away\s+from\s+the\s+scene|immediately\s+drove\s+away|left\s+the\s+injured|without\s+helping)\b",
            r"\b(i\s+lost\s+control|i\s+was\s+speeding|while\s+speeding)\b",
            r"\b(cctv|camera).{0,60}(my\s+(vehicle|car|bike)|vehicle\s+number|my\s+number)\b",
            r"\bi\s+panicked.{0,40}(drove|fled|ran|left)\b",
        ]

        VICTIM_POSTURE_PATTERNS = [
            # I filed / I lodged FIR
            r"\b(i|we)\s[\w\s]{0,10}(filed|lodged|registered|reported)\s[\w\s]{0,20}(fir|complaint|case)\b",
            # FIR / complaint against him/her/them
            r"\b(fir|complaint|case)\s[\w\s]{0,20}(against|on)\s[\w\s]{0,10}(him|her|them|accused)\b",
            # explicit victim declaration
            r"\b(i\s+am\s+(?:a\s+)?victim|i\s+was\s+(?:the\s+)?victim|i\s+am\s+(?:a\s+)?survivor)\b",
            # "he/she/they attacked/threatened/assaulted/harassed me"
            r"\b(he|she|they|manager|boss|neighbour|husband|wife|partner)\b.{0,50}\b(attacked|threatened|assaulted|harassed|abused|beat|beaten|raped|molested|cheated|robbed|snatched|stole\s+from)\b.{0,30}\bme\b",
        ]

        accused_posture_score = 0
        for pat in ACCUSED_POSTURE_PATTERNS:
            if re.search(pat, text):
                accused_posture_score += 5

        victim_posture_score = 0
        for pat in VICTIM_POSTURE_PATTERNS:
            if re.search(pat, text):
                victim_posture_score += 5

        # ── 2. Keyword signals ──────────────────────────────────────────────
        ACCUSED_KEYWORDS = [
            # Original entries
            "i stole", "i took", "i hit", "i beat", "i killed", "i attacked",
            "i threatened", "i cheated", "i committed", "i was caught",
            "i ran away", "i fled", "i broke into", "i snatched", "i robbed",
            "i assaulted", "i molested", "i raped", "i blackmailed",
            "i forged", "i embezzled", "we stole", "we beat", "we attacked",
            "maine mara", "maine churaya", "mujhe arrest",
            # Financial fraud / white-collar
            "i manipulated", "i diverted", "i altered", "i deleted",
            "i transferred", "i created fake", "i falsified", "i siphoned",
            "i misappropriated", "i laundered", "i defrauded", "i bribed",
            "i opened fake", "i made fake", "i concealed", "i destroyed",
            "we manipulated", "we diverted", "we falsified",
            # Admission of scheme
            "the scheme", "my scheme", "the fraud", "my fraud",
            "fake vendor", "fake accounts", "fake invoices",
            "i began facing", "encouraged by the success",
            # Past-tense self-reporting
            "i had stolen", "i had taken", "i had hit", "i had manipulated",
            "i had diverted", "i had embezzled", "i had cheated",
            # Hit-and-run / vehicular
            "i struck", "i hit a", "i ran over", "i crashed into", "i lost control",
            "i drove away", "i fled the scene", "i panicked and",
            "despite being intoxicated", "while intoxicated",
            "i was speeding", "while speeding",
        ]

        VICTIM_KEYWORDS = [
            "someone attacked me", "i was attacked", "i was beaten", "i was robbed",
            "my phone was stolen", "they hit me", "he hit me", "she hit me",
            "i was cheated", "they cheated me", "i was threatened", "threatened me",
            "without consent", "he touched me", "sexual harassment",
            "workplace harassment", "internal complaints committee",
            "icc", "posh", "i reported", "i filed a complaint",
            "i need help", "what should i do", "i am a victim",
        ]

        # Innocence cues — only affect keyword-level accused score, not posture
        INNOCENCE_CUES = [
            "did not commit", "didn't commit", "falsely accused",
            "wrongly accused", "i denied", "i am innocent", "i did nothing",
        ]

        accused_kw = sum(2 for kw in ACCUSED_KEYWORDS if kw in text)
        victim_kw  = sum(2 for kw in VICTIM_KEYWORDS  if kw in text)
        innocence  = sum(1 for kw in INNOCENCE_CUES   if kw in text)

        # Innocence reduces keyword accused score only (not posture)
        accused_kw = max(0, accused_kw - innocence)

        # ── 3. Final tally ──────────────────────────────────────────────────
        total_accused = accused_posture_score + accused_kw
        total_victim  = victim_posture_score  + victim_kw

        logger.debug(
            f"Role detection — accused_posture={accused_posture_score}, "
            f"accused_kw={accused_kw}, victim_posture={victim_posture_score}, "
            f"victim_kw={victim_kw} → "
            f"total_accused={total_accused}, total_victim={total_victim}"
        )

        # Tie-break: prefer victim (safer default for people seeking help)
        if total_accused > total_victim:
            return "accused"
        return "victim"

    def assess_urgency(self, query: str, query_type: str) -> str:
        q = query.lower()
        if any(w in q for w in ["murder", "rape", "kidnap", "dying", "arrested"]):
            return "immediate"
        if any(w in q for w in ["threat", "harassment", "blackmail", "caught"]):
            return "immediate"
        return URGENCY_MAP.get(query_type.lower(), "within_week")

    def get_legal_aid_contacts(self) -> list[dict]:
        return LEGAL_AID_CONTACTS