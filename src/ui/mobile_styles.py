"""Mobile-responsive CSS styles for the Streamlit application."""

MOBILE_STYLES = """
<style>
    /* ========== Mobile Responsive Breakpoints ========== */

    /* Base styles - Desktop first approach */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Responsive font sizes */
    @media screen and (max-width: 768px) {
        /* Mobile devices */
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        /* Reduce heading sizes on mobile */
        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.5rem !important;
        }

        h3 {
            font-size: 1.3rem !important;
        }

        /* Make metric cards stack better */
        [data-testid="stMetric"] {
            padding: 0.5rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }
    }

    /* Small mobile devices */
    @media screen and (max-width: 480px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }

        h1 {
            font-size: 1.5rem !important;
        }

        h2 {
            font-size: 1.3rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        /* Further reduce metric sizes */
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
    }

    /* ========== Sidebar Responsive ========== */
    @media screen and (max-width: 768px) {
        /* Make sidebar collapsible on mobile */
        section[data-testid="stSidebar"] {
            width: 280px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 280px !important;
        }
    }

    @media screen and (max-width: 480px) {
        section[data-testid="stSidebar"] {
            width: 250px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 250px !important;
        }
    }

    /* ========== Button Responsive ========== */
    @media screen and (max-width: 768px) {
        .stButton > button {
            width: 100%;
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }

        .stDownloadButton > button {
            width: 100%;
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }

    /* ========== Charts Responsive ========== */
    @media screen and (max-width: 768px) {
        /* Make charts fill container */
        .js-plotly-plot {
            width: 100% !important;
        }

        .plotly {
            width: 100% !important;
        }

        /* Adjust chart heights for mobile */
        .js-plotly-plot .plotly {
            min-height: 400px;
        }
    }

    @media screen and (max-width: 480px) {
        .js-plotly-plot .plotly {
            min-height: 300px;
        }
    }

    /* ========== Tabs Responsive ========== */
    @media screen and (max-width: 768px) {
        /* Make tabs scrollable on mobile */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            -webkit-overflow-scrolling: touch;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.85rem;
            padding: 0.5rem 0.75rem;
        }
    }

    @media screen and (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.75rem;
            padding: 0.4rem 0.6rem;
        }
    }

    /* ========== Trade Advisory Box Responsive ========== */
    @media screen and (max-width: 768px) {
        /* Adjust padding and font sizes for mobile */
        .trade-advisory-box {
            padding: 15px !important;
        }

        .trade-advisory-box h3 {
            font-size: 1.2rem !important;
        }

        .trade-advisory-box p {
            font-size: 0.95rem !important;
        }
    }

    @media screen and (max-width: 480px) {
        .trade-advisory-box {
            padding: 10px !important;
        }

        .trade-advisory-box h3 {
            font-size: 1.1rem !important;
        }

        .trade-advisory-box p {
            font-size: 0.85rem !important;
        }
    }

    /* ========== Table Responsive ========== */
    @media screen and (max-width: 768px) {
        /* Make tables scrollable */
        .stDataFrame {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        /* Reduce table font size */
        .stDataFrame table {
            font-size: 0.85rem;
        }
    }

    @media screen and (max-width: 480px) {
        .stDataFrame table {
            font-size: 0.75rem;
        }
    }

    /* ========== Expander Responsive ========== */
    @media screen and (max-width: 768px) {
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
        }
    }

    /* ========== Columns Layout - Force Stack on Mobile ========== */
    @media screen and (max-width: 768px) {
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }

    /* ========== Input Fields Responsive ========== */
    @media screen and (max-width: 768px) {
        .stTextInput > div > div {
            font-size: 0.9rem;
        }

        .stSelectbox > div > div {
            font-size: 0.9rem;
        }

        .stTextArea > div > div {
            font-size: 0.9rem;
        }
    }

    /* ========== Progress Bar Responsive ========== */
    @media screen and (max-width: 768px) {
        .stProgress > div {
            height: 0.5rem !important;
        }
    }

    /* ========== Alert/Info Box Responsive ========== */
    @media screen and (max-width: 768px) {
        .stAlert {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
        }
    }

    @media screen and (max-width: 480px) {
        .stAlert {
            padding: 0.5rem !important;
            font-size: 0.85rem !important;
        }
    }

    /* ========== Improve Touch Targets ========== */
    @media screen and (max-width: 768px) {
        /* Ensure minimum touch target size (44x44px) */
        button,
        [role="button"],
        a,
        input[type="checkbox"],
        input[type="radio"] {
            min-height: 44px;
            min-width: 44px;
        }
    }

    /* ========== Viewport Meta Tag Helper ========== */
    /* Ensure proper viewport scaling */
    @viewport {
        width: device-width;
        initial-scale: 1.0;
        maximum-scale: 5.0;
        user-scalable: yes;
    }

    /* ========== Custom Utility Classes ========== */
    .mobile-hide {
        display: block;
    }

    @media screen and (max-width: 768px) {
        .mobile-hide {
            display: none !important;
        }
    }

    .mobile-only {
        display: none;
    }

    @media screen and (max-width: 768px) {
        .mobile-only {
            display: block !important;
        }
    }

    /* ========== Improve Readability ========== */
    @media screen and (max-width: 768px) {
        /* Better line height for mobile reading */
        p, li, div {
            line-height: 1.6;
        }
    }
</style>
"""


def inject_mobile_styles():
    """Inject mobile-responsive CSS into the Streamlit app."""
    import streamlit as st

    st.markdown(MOBILE_STYLES, unsafe_allow_html=True)
