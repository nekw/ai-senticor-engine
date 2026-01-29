"""Load sample sector news into RAG vector database.

This module is automatically called during app initialization to populate
the vector database with sample news articles.
"""
# flake8: noqa

import os
import sys

# Ensure core module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.rag_engine import RAGEngine  # noqa: E402


def load_all_sample_news():
    """Load all sample sector news into the vector database."""
    print("🔄 Initializing RAG database with sample sector news...")

    rag = RAGEngine(llm_provider=None)

    # Check if database exists before clearing
    db_exists = os.path.exists(rag.persist_directory)

    if db_exists:
        print("📂 Existing database found. Clearing and reloading...")
        rag.clear_vector_database()
    else:
        print("📂 No existing database found. Creating new database...")

    # Add Technology sector news
    _add_technology_news(rag)

    # Add Semiconductor sector news
    _add_semiconductor_news(rag)

    # Add Financial Services sector news
    _add_financial_services_news(rag)

    # Add Commodities & Crypto sector news
    _add_commodities_crypto_news(rag)

    print("✅ Sample sector news loaded successfully!")
    return rag


def _add_technology_news(rag: RAGEngine):
    """Add Technology sector news."""
    # AAPL
    rag.add_sector_news(
        sector="Technology",
        headline="Apple Reports Record Services Revenue as iPhone 16 AI Features Drive Upgrades",
        content="Apple announced record-breaking Services revenue of $25.3 billion in Q4 2025, driven by "
        "Apple Intelligence adoption across 550 million devices. iPhone 16 sales exceeded expectations "
        "with AI-powered features like advanced Siri capabilities and on-device image generation "
        "attracting premium customers. Vision Pro sales reached 1.2 million units with strong "
        "enterprise adoption in healthcare and industrial design. CFO Luca Maestri highlighted "
        "Services gross margins of 74%, compared to 36% for hardware. The company announced a new "
        "$120 billion share buyback program and raised dividend by 8%. However, China revenue "
        "declined 3% due to local competition from Huawei's Mate 70 series.",
        ticker="AAPL",
        date="2026-01-24",
    )

    # MSFT
    rag.add_sector_news(
        sector="Technology",
        headline="Microsoft Azure OpenAI Service Revenue Surpasses $15B Annual Run Rate",
        content="Microsoft CEO Satya Nadella revealed Azure OpenAI Service has exceeded $15 billion in "
        "annual recurring revenue, making it the fastest-growing product in company history. "
        "Commercial Cloud revenue grew 29% year-over-year to $38.9 billion, with Azure infrastructure "
        "up 31%. Microsoft 365 Copilot now has 86 million paid seats across enterprises, with Fortune "
        "500 adoption exceeding 90%. The company's gaming division showed 61% revenue growth following "
        "Activision Blizzard integration. However, AI infrastructure costs increased capex to $14 billion "
        "for the quarter. Analysts remain bullish on long-term margin expansion as AI workloads scale.",
        ticker="MSFT",
        date="2026-01-26",
    )

    # GOOGL
    rag.add_sector_news(
        sector="Technology",
        headline="Google Gemini 2.0 Flash Achieves 94% on MMLU-Pro, Outperforms GPT-4o",
        content="Alphabet unveiled Gemini 2.0 Flash, achieving state-of-the-art 94.2% accuracy on MMLU-Pro "
        "benchmark and native multimodal understanding across text, images, audio, and video. The model "
        "powers enhanced Google Search with AI Overviews reaching 1 billion users monthly. YouTube "
        "advertising revenue grew 12% to $9.2 billion despite AI-driven content changes. However, "
        "Search ad revenue growth slowed to 7%, down from 11% in Q3, as generative answers reduce "
        "traditional ad clicks. Google Cloud revenue jumped 35% to $11.4 billion, with AI workloads "
        "accounting for 40% of new customer deals. The company faces antitrust pressure with DOJ "
        "seeking forced divestiture of Chrome browser.",
        ticker="GOOGL",
        date="2026-01-25",
    )

    # DIS
    rag.add_sector_news(
        sector="Entertainment",
        headline="Disney+ Reaches 175M Subscribers, Streaming Achieves First Annual Profit",
        content="The Walt Disney Company reported Disney+ subscriber count reached 175 million globally, "
        "with streaming division posting first full-year profit of $1.2 billion. ESPN+ sports streaming "
        "grew 18% with exclusive NFL and college football rights driving engagement. Theme parks revenue "
        "increased 8% with new AI-powered Star Wars experience at Galaxy's Edge attracting record crowds. "
        "Disney announced plans to integrate generative AI into content production, potentially reducing "
        "animation costs by 30%. Box office performance showed recovery with 'Avatar 3' grossing $2.4 "
        "billion worldwide. However, traditional cable networks (ABC, ESPN) saw 12% revenue decline as "
        "cord-cutting accelerates.",
        ticker="DIS",
        date="2026-01-23",
    )


def _add_semiconductor_news(rag: RAGEngine):
    """Add Semiconductor sector news."""
    # NVDA
    rag.add_sector_news(
        sector="Semiconductors",
        headline="NVIDIA Blackwell B200 GPUs Ship to Hyperscalers, $80B Datacenter Revenue Projected",
        content="NVIDIA CEO Jensen Huang announced Blackwell B200 GPU shipments to Microsoft Azure, AWS, "
        "Google Cloud, and Oracle with production ramping to 200,000 units per month by March 2026. "
        "The chips deliver 2.5x performance improvement over H100 for AI training workloads while "
        "reducing power consumption by 40%. Datacenter revenue is projected to reach $80 billion in "
        "fiscal 2026, up from $47 billion in 2025. NVIDIA also unveiled GB200 NVL72 rack-scale systems "
        "for large language model inference, with Meta ordering 350,000 units. Gross margins remain "
        "strong at 75% despite competitive pressure. Gaming revenue stabilized at $3.2 billion quarterly "
        "with RTX 50-series launch pending.",
        ticker="NVDA",
        date="2026-01-27",
    )

    # MU
    rag.add_sector_news(
        sector="Semiconductors",
        headline="Micron HBM3E Memory Achieves Record 1.2TB/s Bandwidth for AI Training",
        content="Micron Technology announced its 12-Hi HBM3E (High Bandwidth Memory) has achieved industry-leading "
        "1.2TB/s bandwidth and 36GB capacity per stack, surpassing Samsung and SK Hynix specifications. "
        "NVIDIA qualified Micron HBM3E for Blackwell GPUs with volume shipments beginning Q2 2026. The "
        "company expects HBM revenue to exceed $8 billion in fiscal 2026, up from $2 billion in 2025. "
        "DDR5 memory prices stabilized after 18-month decline, with AI server demand offsetting PC market "
        "weakness. Micron's Idaho and New York fabs are expanding capacity with $15 billion in U.S. CHIPS "
        "Act funding. Analysts project operating margins improving to 32% by Q4 2026 as HBM mix increases.",
        ticker="MU",
        date="2026-01-26",
    )

    # LITE
    rag.add_sector_news(
        sector="Semiconductors",
        headline="Lumentum Secures $2.5B in AI Datacenter Optical Module Contracts",
        content="Lumentum Holdings announced contracts worth $2.5 billion for high-speed optical transceivers "
        "used in AI datacenter networking, with deployments spanning 2026-2028. The company's 800G and "
        "1.6T optical modules enable GPU-to-GPU communication in AI clusters, critical for training large "
        "language models. Major customers include Microsoft, Meta, and Amazon Web Services. Revenue guidance "
        "raised to $1.9 billion for fiscal 2026, up 35% year-over-year. However, traditional 3D sensing "
        "business for smartphones declined 12% as Apple reduces Face ID component orders. Management indicated "
        "datacenter optical networking will represent 60% of revenue by 2027, up from 38% in 2025.",
        ticker="LITE",
        date="2026-01-22",
    )

    # Industry context
    rag.add_sector_news(
        sector="Semiconductors",
        headline="AI Chip Demand Drives $720B Global Semiconductor Market, Memory Prices Stabilize",
        content="The Semiconductor Industry Association reported global chip sales reached $720 billion in 2025, "
        "with AI accelerators and HBM memory accounting for $210 billion (+45% YoY). DRAM and NAND flash "
        "prices stabilized in Q4 2025 after 18-month downturn, with AI server demand offsetting PC/smartphone "
        "weakness. Automotive chip revenue grew 8% despite EV market slowdown, driven by ADAS and infotainment "
        "systems. Analysts project 2026 semiconductor market growth of 12-15%, primarily from datacenter AI "
        "buildouts by hyperscalers. Geopolitical risks remain as U.S.-China tech restrictions expand to include "
        "advanced packaging and chipmaking equipment.",
        ticker="",
        date="2026-01-25",
    )


def _add_financial_services_news(rag: RAGEngine):
    """Add Financial Services sector news."""
    # V
    rag.add_sector_news(
        sector="Financial Services",
        headline="Visa Processes Record 276B Transactions in 2025, Tap-to-Pay Reaches 80% Penetration",
        content="Visa Inc. reported processing 276 billion transactions globally in 2025, up 11% year-over-year, "
        "with payment volume reaching $14.2 trillion. Contactless 'tap-to-pay' transactions now represent "
        "80% of face-to-face purchases in developed markets, accelerating checkout speeds and reducing fraud. "
        "Visa Direct person-to-person payment volume surged 35% to $1.8 trillion, competing directly with "
        "Zelle and Venmo. Cross-border transaction revenue grew 17% as international travel recovered to "
        "pre-pandemic levels. The company announced new AI-powered fraud detection systems that reduced false "
        "declines by 25%. However, interchange fee regulations in Europe compressed margins by 40 basis points. "
        "New value-added services including identity verification and tokenization grew 24%.",
        ticker="V",
        date="2026-01-26",
    )

    # MA
    rag.add_sector_news(
        sector="Financial Services",
        headline="Mastercard Partners with 15 Central Banks on CBDC Trials, Digital Payments Grow 19%",
        content="Mastercard announced partnerships with 15 central banks globally to pilot Central Bank Digital "
        "Currency (CBDC) infrastructure, positioning itself as a key player in the future of digital money. "
        "The company's Multi-Token Network platform processed $3.2 billion in tokenized asset transactions "
        "in 2025. Total payment volume reached $8.9 trillion, up 12% year-over-year. Mastercard's Cyber & "
        "Intelligence division grew 28% as banks increased spending on fraud prevention and data analytics. "
        "Click-to-Pay adoption accelerated with 250 million enrolled users, streamlining online checkout "
        "experiences. Management highlighted switching services revenue up 21%, driven by data insights and "
        "consulting. Competition from Apple Pay and Google Wallet intensifies, though co-opetition model "
        "allows Mastercard to facilitate these services.",
        ticker="MA",
        date="2026-01-25",
    )


def _add_commodities_crypto_news(rag: RAGEngine):
    """Add Commodities & Crypto sector news."""
    # SLV
    rag.add_sector_news(
        sector="Commodities",
        headline="Silver Prices Surge to $34/oz on Industrial Demand, Solar Panel Production Soars",
        content="Silver prices rallied to $34 per ounce, up 28% year-to-date, driven by unprecedented industrial "
        "demand from solar panel manufacturing and EV electronics. Global solar installations reached 580 "
        "gigawatts in 2025, requiring 180 million ounces of silver annually. The Silver Institute reported "
        "physical deficit of 215 million ounces, the largest shortage in a decade, as mine production "
        "declined 3% while industrial applications grew 12%. iShares Silver Trust (SLV) assets under "
        "management increased to $16.2 billion with institutional investors treating silver as both commodity "
        "and inflation hedge. Analysts project prices could reach $40/oz if supply constraints persist through "
        "2026. However, silver's volatility (annualized 35%) remains significantly higher than gold.",
        ticker="SLV",
        date="2026-01-27",
    )

    # IBIT
    rag.add_sector_news(
        sector="Cryptocurrency",
        headline="iShares Bitcoin Trust Surpasses $50B AUM, SEC Approves Options Trading",
        content="BlackRock's iShares Bitcoin Trust (IBIT) reached $50 billion in assets under management, making "
        "it the fastest ETF in history to achieve this milestone in just 11 months since launch. The SEC "
        "approved options trading on spot Bitcoin ETFs, unlocking institutional hedging strategies and "
        "potentially reducing volatility. IBIT daily trading volume averages $2.8 billion, with 65% of "
        "flows from registered investment advisors and pension funds. Bitcoin price stabilized around "
        "$98,000 after reaching all-time high of $112,000 in December 2025. Grayscale's GBTC experienced "
        "$12 billion in outflows as investors rotated to lower-fee alternatives like IBIT (0.25% vs 1.5%). "
        "The crypto market capitalization exceeded $3.8 trillion with improving regulatory clarity.",
        ticker="IBIT",
        date="2026-01-26",
    )

    # ETH
    rag.add_sector_news(
        sector="Cryptocurrency",
        headline="Ethereum Reaches 15M Transactions Daily After Dencun Upgrade, ETH Price at $4,200",
        content="Ethereum network processed record 15 million transactions daily following the Dencun upgrade, "
        "which reduced Layer 2 rollup costs by 90% through EIP-4844 blob transactions. Ether (ETH) price "
        "rallied to $4,200, up 85% year-to-date, driven by institutional adoption and Ethereum ETF launches. "
        "Total value locked in DeFi protocols reached $180 billion, with Ethereum commanding 62% market share. "
        "Spot Ethereum ETFs attracted $8.5 billion in net inflows in first six months, though lagging Bitcoin "
        "ETF adoption. Ethereum's transition to proof-of-stake reduced energy consumption by 99.95%, addressing "
        "ESG concerns from institutional investors. However, competition from Solana and emerging L1 blockchains "
        "intensifies as developers seek lower transaction costs. Ethereum Foundation announced plans for 'The "
        "Merge 2.0' to further improve scalability.",
        ticker="ETH",
        date="2026-01-24",
    )

    # TQQQ/SQQQ
    rag.add_sector_news(
        sector="Leveraged ETFs",
        headline="ProShares Ultra QQQ (TQQQ) Hits $30B AUM Amid Retail Trading Surge, SQQQ Shrinks",
        content="ProShares UltraPro QQQ (TQQQ), the 3x leveraged Nasdaq-100 ETF, reached $30 billion in assets "
        "under management as retail traders doubled down on tech sector momentum. The fund returned 127% in "
        "2025, amplifying the Nasdaq-100's 42% gain, though volatility averaged 65% annualized. Conversely, "
        "ProShares UltraPro Short QQQ (SQQQ), the inverse 3x bear ETF, saw assets decline to $1.8 billion "
        "as persistent tech rally punished short positions. Financial advisors warned retail investors about "
        "compounding effects and daily rebalancing risks that can erode long-term returns. TQQQ's popularity "
        "concentrated in self-directed brokerage accounts, with Robinhood and Fidelity reporting the fund "
        "among top holdings. Regulators expressed concern about leverage concentration among inexperienced "
        "traders. Fund prospectus emphasizes these are short-term tactical tools, not buy-and-hold investments.",
        ticker="TQQQ",
        date="2026-01-23",
    )


if __name__ == "__main__":
    load_all_sample_news()
