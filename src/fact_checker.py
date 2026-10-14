"""
fact_checker.py - Grounded Misinformation Flagging & Fact-Check Verification Engine.
Pairs sentiment classification with a lightweight verification engine.
When anti-vaccine sentiment is detected, cross-references text against a curated
vector database of verified WHO/CDC fact-checking FAQs using sentence embeddings
(sentence-transformers/all-MiniLM-L6-v2) or an intelligent lexical fallback.

Author: Krrish P Raju (PES1UG24CS708)
"""

import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class FactEntry:
    topic: str
    claim_keywords: List[str]
    sample_claims: List[str]
    grounded_fact: str
    source: str


# Curated vector database of verified WHO and CDC fact-checking FAQs
VERIFIED_FACT_DATABASE: List[FactEntry] = [
    FactEntry(
        topic="DNA Alteration",
        claim_keywords=["dna", "gene", "genes", "genetic", "alter", "genome", "改变", "基因", "遗传"],
        sample_claims=[
            "mRNA vaccines alter human DNA and genetic code",
            "COVID-19 vaccine changes your genes and modifies genome",
            "mRNA vaccine permanently alters your DNA",
            "疫苗会改变人体DNA和基因结构",
            "接种疫苗破坏人类遗传基因"
        ],
        grounded_fact="CDC finding: mRNA vaccines do not alter human DNA. mRNA never enters the cell nucleus where DNA is kept, and the cell breaks down the mRNA shortly after using the instructions.",
        source="CDC (Centers for Disease Control and Prevention)"
    ),
    FactEntry(
        topic="Infertility & Reproduction",
        claim_keywords=["infertility", "fertility", "sterile", "sterilization", "pregnancy", "placenta", "syncytin", "不孕", "不育", "流产", "胎盘"],
        sample_claims=[
            "COVID-19 vaccines cause female or male infertility and sterilization",
            "Vaccines attack placenta protein syncytin-1 causing miscarriage",
            "The vaccine makes young women infertile",
            "打新冠疫苗会导致女性不孕不育",
            "疫苗攻击胎盘导致绝育"
        ],
        grounded_fact="WHO & CDC statement: COVID-19 vaccines do not cause infertility or affect future pregnancies. Clinical studies found no cross-reactive antibodies targeting placental syncytin-1.",
        source="WHO / CDC Joint Briefing"
    ),
    FactEntry(
        topic="Microchips & Surveillance",
        claim_keywords=["microchip", "chip", "5g", "tracking", "surveillance", "magnetic", "magnet", "芯片", "跟踪", "监控", "磁性"],
        sample_claims=[
            "Vaccines contain tracking microchips and 5G nanobots",
            "COVID vaccine implants a magnetic surveillance chip to control people",
            "They are putting tracking devices in vaccines",
            "疫苗里面有微芯片和5G纳米机器人用于人体监控",
            "接种疫苗身体产生磁性被芯片控制"
        ],
        grounded_fact="CDC finding: COVID-19 vaccines do not contain microchips, metals, or tracking technology. Ingredients are publicly disclosed: mRNA/antigen, lipids, salts, and sugars.",
        source="CDC Official Fact Sheet"
    ),
    FactEntry(
        topic="Vaccine Shedding",
        claim_keywords=["shed", "shedding", "transmit", "sheds", "contagious", "散播", "传染", "排放", "活病毒"],
        sample_claims=[
            "Vaccinated people shed the spike protein or virus to unvaccinated people",
            "Vaccine shedding makes others sick through close contact",
            "Vaccinated individuals are shedding contagious virus particles",
            "打过疫苗的人会散播活病毒给其他人",
            "疫苗排放刺突蛋白伤害周围未接种人群"
        ],
        grounded_fact="CDC finding: Vaccine shedding can only occur when a vaccine contains a weakened live virus; authorized COVID-19 vaccines contain no live virus and cannot shed.",
        source="CDC Vaccine Safety Division"
    ),
    FactEntry(
        topic="Rushed Trials & Safety Protocols",
        claim_keywords=["rush", "rushed", "untested", "experimental", "guinea pig", "trials", "实验", "小白鼠", "未经试验", "匆忙"],
        sample_claims=[
            "Vaccines were rushed without safety trials and people are experimental guinea pigs",
            "No clinical testing was conducted before rolling out vaccines",
            "These vaccines are completely untested experimental chemicals",
            "疫苗未经正规临床试验就把大众当小白鼠",
            "仓促上市没有经过任何安全检测"
        ],
        grounded_fact="WHO confirmation: mRNA platforms had been researched for decades; COVID-19 trials completed all standard Phase 1, 2, and 3 clinical safety protocols with tens of thousands of participants.",
        source="WHO / Global Advisory Committee on Vaccine Safety"
    ),
    FactEntry(
        topic="Natural Immunity Superiority",
        claim_keywords=["natural immunity", "better than vaccine", "useless", "immune", "natural", "天然免疫", "自愈", "天然抗体", "无用"],
        sample_claims=[
            "Natural infection gives 100% immunity so vaccines are totally useless",
            "No need for vaccines because getting sick naturally is completely safe and better",
            "Natural immunity makes vaccines obsolete and harmful",
            "得过新冠获得天然抗体完全不需要接种疫苗",
            "天然免疫力比疫苗好一千倍打疫苗有害无利"
        ],
        grounded_fact="WHO finding: Vaccines provide a more reliable and safer immune response than natural infection without the risk of severe illness, long COVID, or hospitalization.",
        source="WHO Science in 5"
    ),
    FactEntry(
        topic="Severe Neurological Collapse",
        claim_keywords=["bell's palsy", "palsy", "paralysis", "neurological", "seizure", "collapse", "瘫痪", "面瘫", "神经系统", "抽搐"],
        sample_claims=[
            "Vaccines cause widespread permanent facial paralysis and Bell's palsy",
            "The shot causes nervous system destruction and permanent paralysis",
            "COVID vaccine destroys the central nervous system",
            "疫苗导致大面积永久性面瘫和中枢神经瘫痪",
            "接种后引发全身瘫痪和癫痫发作"
        ],
        grounded_fact="CDC & FDA analysis: Cases of Bell's palsy in clinical trials occurred at rates consistent with the expected background rate in the general population, showing no causal relationship.",
        source="FDA Briefing Document / CDC"
    )
]


@dataclass
class FactCheckResult:
    sentiment_label: str
    sentiment_conf: float
    risk_level: str
    grounded_fact: str
    topic: str
    similarity_score: float

    def format_output(self) -> str:
        """
        Formats result matching the required output schema:
        Sentiment: Negative (0.94)
        Flag: Misinformation Risk: Elevated
        Grounded Fact: "CDC finding: mRNA vaccines do not alter human DNA."
        """
        lines = [
            f"Sentiment: {self.sentiment_label} ({self.sentiment_conf:.2f})",
            f"Flag: Misinformation Risk: {self.risk_level}",
        ]
        if self.grounded_fact:
            lines.append(f'Grounded Fact: "{self.grounded_fact}"')
        else:
            lines.append("Grounded Fact: None (No verified misinformation narrative detected.)")
        return "\n".join(lines)


class FallbackLexicalEmbedder:
    """
    Lightweight character- and word-level n-gram embedder with sublinear TF-IDF weighting
    and cosine similarity. Guarantees 0-dependency execution when sentence-transformers
    is not installed.
    """
    def __init__(self, entries: List[FactEntry]):
        self.entries = entries
        self.corpus_terms: List[Dict[str, float]] = []
        self._build_index()

    def _extract_ngrams(self, text: str) -> Dict[str, float]:
        text = text.lower()
        counts: Dict[str, float] = {}
        # Words
        words = re.findall(r"\b\w+\b", text)
        for w in words:
            counts[w] = counts.get(w, 0.0) + 1.0
        # Character bigrams/trigrams for CJK characters and substrings
        for i in range(len(text) - 1):
            bi = text[i:i+2].strip()
            if bi:
                counts[bi] = counts.get(bi, 0.0) + 1.5
        for i in range(len(text) - 2):
            tri = text[i:i+3].strip()
            if tri:
                counts[tri] = counts.get(tri, 0.0) + 2.0
        # Sublinear TF scaling
        vec = {k: 1.0 + math.log(v) for k, v in counts.items()}
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _build_index(self):
        for entry in self.entries:
            combined_text = " ".join(entry.sample_claims + entry.claim_keywords)
            vec = self._extract_ngrams(combined_text)
            self.corpus_terms.append(vec)

    def compute_similarity(self, query: str) -> List[float]:
        q_vec = self._extract_ngrams(query)
        sims = []
        for doc_vec in self.corpus_terms:
            dot = sum(val * doc_vec.get(term, 0.0) for term, val in q_vec.items())
            sims.append(dot)
        return sims


class FactCheckVerifier:
    """
    Lightweight Fact-Check Verification Engine.
    Cross-references microblog posts against WHO/CDC fact-checking FAQs using
    sentence embeddings (sentence-transformers/all-MiniLM-L6-v2) or fallback lexical embeddings.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.database = VERIFIED_FACT_DATABASE
        self.st_model = None
        self.encoded_db = None
        self.fallback_embedder = FallbackLexicalEmbedder(self.database)

        # Attempt to load sentence-transformers if installed
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
            all_claims = [" ".join(e.sample_claims) for e in self.database]
            self.encoded_db = self.st_model.encode(all_claims, normalize_embeddings=True)
        except Exception:
            self.st_model = None
            self.encoded_db = None

    def calculate_similarities(self, text: str) -> List[float]:
        """Calculates semantic similarity against all database entries."""
        if self.st_model is not None and self.encoded_db is not None:
            try:
                import numpy as np
                q_emb = self.st_model.encode([text], normalize_embeddings=True)[0]
                sims = np.dot(self.encoded_db, q_emb).tolist()
                return sims
            except Exception:
                pass
        return self.fallback_embedder.compute_similarity(text)

    def verify(
        self,
        text: str,
        sentiment_pred: int,
        sentiment_conf: float
    ) -> FactCheckResult:
        """
        Evaluates post for misinformation risk:
        - sentiment_pred: 0 for Negative, 1 for Positive
        - sentiment_conf: confidence score (0.0 to 1.0)
        """
        sentiment_label = "Positive" if sentiment_pred == 1 else "Negative"
        sims = self.calculate_similarities(text)

        best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
        best_sim = sims[best_idx]
        best_entry = self.database[best_idx]

        # Keyword booster: check if specific misinformation keywords appear in query
        lower_text = text.lower()
        keyword_hits = sum(1 for kw in best_entry.claim_keywords if kw in lower_text)
        if keyword_hits >= 2:
            best_sim = max(best_sim, 0.55 + 0.1 * min(keyword_hits, 3))
        elif keyword_hits == 1:
            best_sim = max(best_sim, 0.42)

        # Misinformation risk evaluation logic:
        # Requires anti-vaccine / negative sentiment + high semantic similarity to a debunked claim
        if sentiment_pred == 0 and sentiment_conf >= 0.55:
            if best_sim >= 0.38:
                risk = "Elevated"
                fact = best_entry.grounded_fact
                topic = best_entry.topic
            elif best_sim >= 0.25:
                risk = "Moderate"
                fact = best_entry.grounded_fact
                topic = best_entry.topic
            else:
                risk = "Low"
                fact = ""
                topic = "General Negative / Personal Discomfort"
        else:
            # Positive or ambiguous posts carry low misinformation risk
            risk = "Low"
            fact = ""
            topic = "General / Positive Sentiment"

        return FactCheckResult(
            sentiment_label=sentiment_label,
            sentiment_conf=sentiment_conf,
            risk_level=risk,
            grounded_fact=fact,
            topic=topic,
            similarity_score=float(best_sim)
        )
