import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from utils.humanizer import Humanizer
from utils.ai_detector import AIDetector
from utils.text_analyzer import TextAnalyzer

# Page configuration
st.set_page_config(
    page_title="HumanizerPro - AI Text Humanizer & Detector",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize classes
@st.cache_resource
def get_humanizer():
    return Humanizer()

@st.cache_resource
def get_detector():
    return AIDetector()

@st.cache_resource
def get_analyzer():
    return TextAnalyzer()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 HumanizerPro</h1>
        <p>Advanced AI Text Humanizer & Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🏠 Home", "🔍 AI Detector", "✨ Text Humanizer", "📊 Text Analysis", "ℹ️ About"]
    )
    
    humanizer = get_humanizer()
    detector = get_detector()
    analyzer = get_analyzer()
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "🔍 AI Detector":
        show_ai_detector(detector, analyzer)
    elif app_mode == "✨ Text Humanizer":
        show_text_humanizer(humanizer, analyzer)
    elif app_mode == "📊 Text Analysis":
        show_text_analysis(analyzer)
    elif app_mode == "ℹ️ About":
        show_about()

def show_home():
    st.markdown("""
    <div class="feature-card">
        <h2>Welcome to HumanizerPro! 🎉</h2>
        <p>Your all-in-one solution for detecting AI-generated content and humanizing text to make it undetectable.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 AI Detection</h3>
            <p>Advanced algorithms to detect AI-generated content with detailed analysis and confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✨ Text Humanizer</h3>
            <p>Transform AI-generated text into human-like content with multiple intensity levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Deep Analysis</h3>
            <p>Comprehensive text analysis with readability scores, grammar checks, and improvement suggestions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("Start AI Detection", use_container_width=True):
            st.session_state.navigation = "AI Detector"
            st.rerun()
    
    with quick_col2:
        if st.button("Humanize Text", use_container_width=True):
            st.session_state.navigation = "Text Humanizer"
            st.rerun()
    
    with quick_col3:
        if st.button("Analyze Text", use_container_width=True):
            st.session_state.navigation = "Text Analysis"
            st.rerun()

def show_ai_detector(detector, analyzer):
    st.header("🔍 AI Content Detector")
    
    text_input = st.text_area(
        "Enter text to analyze for AI content:",
        height=200,
        placeholder="Paste your text here to check if it was AI-generated..."
    )
    
    if st.button("Analyze for AI Content", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing text for AI patterns..."):
                time.sleep(1)  # Simulate processing
                result = detector.advanced_detection(text_input)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card ai-score">
                        <h3>AI Score</h3>
                        <h2>{result['ai_score']}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card human-score">
                        <h3>Human Score</h3>
                        <h2>{result['human_score']}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['human_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Human-like Confidence"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightcoral"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed analysis
                st.subheader("📊 Detailed Analysis")
                
                if result['indicators_found']:
                    st.warning(f"**AI Indicators Found:** {', '.join(result['indicators_found'])}")
                else:
                    st.success("No strong AI indicators detected!")
                
                # Text analysis
                analysis = analyzer.analyze_text(text_input)
                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
                
                with analysis_col1:
                    st.metric("Words", analysis['word_count'])
                    st.metric("Sentences", analysis['sentence_count'])
                
                with analysis_col2:
                    st.metric("Readability Score", f"{analysis['flesch_reading_ease']:.1f}")
                    st.metric("Grammar Errors", analysis['grammar_errors'])
                
                with analysis_col3:
                    st.metric("Lexical Diversity", f"{analysis['lexical_diversity']:.2%}")
                    st.metric("Avg Sentence Length", f"{analysis['avg_sentence_length']:.1f}")
        
        else:
            st.error("Please enter some text to analyze.")

def show_text_humanizer(humanizer, analyzer):
    st.header("✨ AI Text Humanizer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        original_text = st.text_area(
            "Original Text:",
            height=250,
            placeholder="Paste AI-generated text here to humanize it..."
        )
    
    with col2:
        intensity = st.selectbox(
            "Humanization Intensity:",
            ["low", "medium", "high"],
            index=1,
            help="Low: Minor changes, Medium: Balanced, High: Extensive rewriting"
        )
        
        st.markdown("""
        **Intensity Levels:**
        - **Low:** Basic phrase replacement
        - **Medium:** Sentence restructuring + contractions
        - **High:** Advanced rewriting + conversational elements
        """)
    
    if st.button("Humanize Text", use_container_width=True):
        if original_text.strip():
            with st.spinner("Humanizing text... This may take a few seconds."):
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                humanized_text = humanizer.humanize_text(original_text, intensity)
                report = humanizer.get_humanization_report(original_text, humanized_text)
                
                # Display results
                st.subheader("🎉 Humanized Text")
                st.text_area("Humanized Output:", humanized_text, height=250)
                
                # Improvement report
                st.subheader("📈 Improvement Report")
                
                imp_col1, imp_col2, imp_col3, imp_col4 = st.columns(4)
                
                with imp_col1:
                    change = report['improvements']['readability_change']
                    trend = "↗️" if change > 0 else "↘️"
                    st.metric("Readability Change", f"{change:+.1f}", delta=trend)
                
                with imp_col2:
                    change = report['improvements']['lexical_diversity_change']
                    trend = "↗️" if change > 0 else "↘️"
                    st.metric("Vocabulary Diversity", f"{change:+.3f}", delta=trend)
                
                with imp_col3:
                    change = report['improvements']['sentence_variety']
                    st.metric("Sentence Variety", f"{change:.1f}")
                
                with imp_col4:
                    change = report['improvements']['word_count_change']  # Changed from grammar_improvement
                    trend = "↗️" if change > 0 else "↘️"
                    st.metric("Word Count Change", f"{change:+d}", delta=trend)
                
                # Comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Readability comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Original',
                        x=['Readability'],
                        y=[report['original_analysis']['flesch_reading_ease']],
                        marker_color='indianred'
                    ))
                    fig.add_trace(go.Bar(
                        name='Humanized',
                        x=['Readability'],
                        y=[report['humanized_analysis']['flesch_reading_ease']],
                        marker_color='lightsalmon'
                    ))
                    fig.update_layout(title="Readability Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics comparison
                    metrics_df = pd.DataFrame({
                        'Metric': ['Lexical Diversity', 'Sentence Length'],
                        'Original': [
                            report['original_analysis']['lexical_diversity'],
                            report['original_analysis']['avg_sentence_length']
                        ],
                        'Humanized': [
                            report['humanized_analysis']['lexical_diversity'],
                            report['humanized_analysis']['avg_sentence_length']
                        ]
                    })
                    
                    fig = px.line(metrics_df, x='Metric', y=['Original', 'Humanized'], 
                                title="Text Metrics Comparison", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Please enter some text to humanize.")

def show_text_analysis(analyzer):
    st.header("📊 Advanced Text Analysis")
    
    text_input = st.text_area(
        "Enter text for comprehensive analysis:",
        height=200,
        placeholder="Paste any text here for detailed analysis..."
    )
    
    if st.button("Analyze Text", use_container_width=True):
        if text_input.strip():
            with st.spinner("Performing comprehensive text analysis..."):
                time.sleep(1)
                analysis = analyzer.analyze_text(text_input)
                suggestions = analyzer.get_grammar_suggestions(text_input)
                
                # Display metrics
                st.subheader("📈 Text Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Characters", analysis['char_count'])
                    st.metric("Words", analysis['word_count'])
                
                with col2:
                    st.metric("Sentences", analysis['sentence_count'])
                    st.metric("Paragraphs", analysis['paragraph_count'])
                
                with col3:
                    st.metric("Unique Words", analysis['unique_words'])
                    st.metric("Lexical Diversity", f"{analysis['lexical_diversity']:.2%}")
                
                with col4:
                    st.metric("Avg Sentence Length", f"{analysis['avg_sentence_length']:.1f}")
                    st.metric("Avg Word Length", f"{analysis['avg_word_length']:.1f}")
                
                # Readability scores
                st.subheader("🎯 Readability Scores")
                
                read_col1, read_col2 = st.columns(2)
                
                with read_col1:
                    ease_score = analysis['flesch_reading_ease']
                    if ease_score >= 90:
                        level = "Very Easy"
                        color = "green"
                    elif ease_score >= 80:
                        level = "Easy"
                        color = "lightgreen"
                    elif ease_score >= 70:
                        level = "Fairly Easy"
                        color = "yellow"
                    elif ease_score >= 60:
                        level = "Standard"
                        color = "orange"
                    elif ease_score >= 50:
                        level = "Fairly Difficult"
                        color = "red"
                    else:
                        level = "Very Difficult"
                        color = "darkred"
                    
                    st.metric("Flesch Reading Ease", f"{ease_score:.1f} ({level})")
                
                with read_col2:
                    grade_level = analysis['flesch_kincaid_grade']
                    st.metric("Flesch-Kincaid Grade", f"Grade {grade_level:.1f}")
                
                # Grammar suggestions
                if suggestions:
                    st.subheader("🔧 Grammar & Style Suggestions")
                    
                    for i, suggestion in enumerate(suggestions, 1):
                        with st.expander(f"Suggestion {i}: {suggestion['message']}"):
                            st.write(f"**Context:** {suggestion['context']}")
                            if suggestion['replacements']:
                                st.write(f"**Suggestions:** {', '.join(suggestion['replacements'])}")
                else:
                    st.success("🎉 No grammar or style suggestions found! Your text looks good.")
        
        else:
            st.error("Please enter some text to analyze.")

def show_about():
    st.header("ℹ️ About HumanizerPro")
    
    st.markdown("""
    <div class="feature-card">
        <h3>What is HumanizerPro?</h3>
        <p>HumanizerPro is an advanced AI-powered tool designed to detect AI-generated content and transform it into human-like text. 
        It uses sophisticated algorithms and natural language processing techniques to analyze and improve text quality.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Key Features
    
    **🔍 AI Detection**
    - Pattern recognition for AI-generated content
    - Confidence scoring system
    - Detailed analysis reports
    - Multiple detection algorithms
    
    **✨ Text Humanization**
    - Multiple intensity levels
    - Sentence structure variation
    - Vocabulary enhancement
    - Grammar optimization
    - Conversational elements
    
    **📊 Advanced Analytics**
    - Readability scoring
    - Lexical diversity analysis
    - Grammar and style checking
    - Improvement tracking
    - Comparative analysis
    
    ### 🛠️ Technology Stack
    
    - **Streamlit** - Web application framework
    - **NLTK** - Natural language processing
    - **TextStat** - Readability metrics
    - **LanguageTool** - Grammar checking
    - **Plotly** - Data visualization
    - **Scikit-learn** - Machine learning algorithms
    
    ### 📝 How It Works
    
    1. **Text Analysis**: Comprehensive analysis of input text
    2. **Pattern Detection**: Identification of AI-generated patterns
    3. **Transformation**: Application of humanization techniques
    4. **Optimization**: Grammar and style improvements
    5. **Reporting**: Detailed improvement metrics and analysis
    
    ### 🔒 Privacy & Security
    
    - All processing happens locally in your browser
    - No data is stored on our servers
    - Complete privacy protection
    - Free and open-source
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and open-source technologies</p>
        <p>HumanizerPro v1.0 - Making AI content undetectable</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()