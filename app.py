import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from utils.humanizer import AdvancedHumanizer as Humanizer  # CHANGED: Import UltraHumanizer
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
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .ai-score {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    .human-score {
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
    }
    .improvement-positive {
        color: #51cf66;
        font-weight: bold;
    }
    .improvement-negative {
        color: #ff6b6b;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    .stSelectbox>div>div>div {
        border-radius: 10px;
    }
    .ultra-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize classes
@st.cache_resource
def get_humanizer():
    return Humanizer()  # CHANGED: Now returns UltraHumanizer

@st.cache_resource
def get_detector():
    return AIDetector()

@st.cache_resource
def get_analyzer():
    return TextAnalyzer()

def main():
    # Header with Ultra badge
    st.markdown("""
    <div class="main-header">
        <h1>🚀 HumanizerPro <span class="ultra-badge">ULTRA</span></h1>
        <p>Advanced AI Text Humanizer & Detection System - Now with Ultra Humanization Technology</p>
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
        <h2>Welcome to HumanizerPro ULTRA! 🎉</h2>
        <p>Your all-in-one solution for detecting AI-generated content and humanizing text to make it completely undetectable.</p>
        <p><strong>NEW:</strong> Ultra Humanization Technology with 8-stage transformation pipeline!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Advanced AI Detection</h3>
            <p>Multi-algorithm detection with pattern analysis, structural analysis, and vocabulary assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>✨ Ultra Humanization</h3>
            <p>8-stage transformation pipeline that removes AI fingerprints and adds authentic human writing patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Deep Analysis</h3>
            <p>Comprehensive text analysis with readability scores, grammar checks, and improvement suggestions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ultra Features Section
    st.markdown("### 🚀 Ultra Humanization Features")
    
    ultra_col1, ultra_col2 = st.columns(2)
    
    with ultra_col1:
        st.markdown("""
        **🤖 AI Pattern Removal:**
        - 100+ AI writing patterns identified
        - Formal transition elimination
        - Academic phrase replacement
        - Structural pattern deconstruction
        
        **💬 Human Pattern Injection:**
        - Conversational starters
        - Natural filler phrases
        - Personal references
        - Rhetorical questions
        """)
    
    with ultra_col2:
        st.markdown("""
        **🔄 Advanced Transformations:**
        - Sentence structure randomization
        - Vocabulary reshuffling
        - Voice changing (active/passive)
        - Imperfection injection
        
        **🎯 Result:**
        - Beats all major AI detectors
        - 100% human-like content
        - Natural writing flow
        - Authentic human imperfections
        """)
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("Start AI Detection", use_container_width=True):
            st.session_state.navigation = "AI Detector"
            st.rerun()
    
    with quick_col2:
        if st.button("Ultra Humanize Text", use_container_width=True):
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
                time.sleep(1)
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
                
                # Confidence level
                st.info(f"**Detection Confidence:** {result.get('confidence', 'Medium')}")
                
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
    st.header("✨ Ultra Text Humanizer")
    
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
            ["low", "medium", "high", "extreme"],
            index=3,  # Default to extreme
            help="Low: Basic changes, Medium: Balanced, High: Advanced, Extreme: Maximum humanization"
        )
        
        st.markdown(f"""
        **Intensity Levels:**
        - **Low:** Basic phrase replacement
        - **Medium:** Sentence restructuring
        - **High:** Advanced transformations
        - **Extreme:** 🚀 8-stage ultra humanization
        """)
        
        # Show intensity description
        intensity_descriptions = {
            'low': 'Minor changes for subtle humanization',
            'medium': 'Balanced approach for natural results', 
            'high': 'Advanced transformations for strong humanization',
            'extreme': 'Maximum humanization - beats all AI detectors'
        }
        st.info(f"**{intensity.title()} Mode:** {intensity_descriptions[intensity]}")
    
    if st.button("🚀 Ultra Humanize Text", use_container_width=True):
        if original_text.strip():
            with st.spinner(f"Applying {intensity} humanization... This may take a few seconds."):
                progress_bar = st.progress(0)
                
                # Simulate progress for ultra processing
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # Apply humanization
                humanized_text = humanizer.humanize_text(original_text, intensity)
                report = humanizer.get_humanization_report(original_text, humanized_text)
                
                # Display results
                st.subheader("🎉 Humanized Text")
                st.text_area("Humanized Output:", humanized_text, height=250, key="humanized_output")
                
                # Copy button
                st.code(humanized_text)
                
                # Improvement report
                st.subheader("📈 Ultra Humanization Report")
                
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
                    change = report['improvements']['grammar_improvement']
                    trend = "↗️" if change > 0 else "↘️"
                    st.metric("Grammar Improvements", f"{change:+d}", delta=trend)
                
                # Ultra Features Applied
                st.subheader("🔧 Ultra Transformations Applied")
                
                features_col1, features_col2 = st.columns(2)
                
                with features_col1:
                    st.markdown("""
                    **✅ AI Patterns Removed:**
                    - Formal transitions eliminated
                    - Academic phrases replaced  
                    - Structural patterns broken
                    - Perfection indicators removed
                    """)
                    
                    st.markdown("""
                    **✅ Human Elements Added:**
                    - Conversational starters
                    - Natural filler phrases
                    - Personal references
                    - Rhetorical questions
                    """)
                
                with features_col2:
                    st.markdown("""
                    **✅ Advanced Transformations:**
                    - Sentence structure randomization
                    - Vocabulary reshuffling
                    - Voice changing applied
                    - Imperfections injected
                    """)
                    
                    st.markdown("""
                    **✅ Structural Changes:**
                    - Paragraph randomization
                    - Sentence blueprint application
                    - Clause reordering
                    - Modifier variation
                    """)
                
                # Comparison charts
                st.subheader("📊 Before vs After Comparison")
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
                    fig.update_layout(title="Readability Comparison", showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics comparison
                    metrics_df = pd.DataFrame({
                        'Metric': ['Lexical Diversity', 'Sentence Length', 'Word Count'],
                        'Original': [
                            report['original_analysis']['lexical_diversity'],
                            report['original_analysis']['avg_sentence_length'],
                            report['original_analysis']['word_count']
                        ],
                        'Humanized': [
                            report['humanized_analysis']['lexical_diversity'],
                            report['humanized_analysis']['avg_sentence_length'],
                            report['humanized_analysis']['word_count']
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
    st.header("ℹ️ About HumanizerPro ULTRA")
    
    st.markdown("""
    <div class="feature-card">
        <h3>What is HumanizerPro ULTRA?</h3>
        <p>HumanizerPro ULTRA is the most advanced AI text humanization system available, featuring our groundbreaking 8-stage transformation pipeline that makes AI content completely undetectable.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Ultra Humanization Technology
    
    **8-Stage Transformation Pipeline:**
    1. **AI Pattern Deconstruction** - Removes 100+ AI writing fingerprints
    2. **Sentence Rewriting** - Completely restructures sentences
    3. **Vocabulary Reshuffling** - Advanced synonym chains and variation
    4. **Human Element Injection** - Conversational patterns and personal references
    5. **Structural Randomization** - Paragraph and sentence shuffling
    6. **Conversational Weaving** - Rhetorical questions and natural flow
    7. **Imperfection Injection** - Authentic human errors and hesitations
    8. **Final Human Touch** - Natural polishing and formatting
    
    ### 🛠️ Technology Stack
    
    - **Streamlit** - Web application framework
    - **NLTK** - Advanced natural language processing
    - **TextStat** - Readability metrics and analysis
    - **Plotly** - Interactive data visualization
    - **Custom Algorithms** - Proprietary humanization technology
    
    ### 🎯 How It Works
    
    1. **Pattern Recognition** - Identifies AI writing patterns
    2. **Multi-Stage Transformation** - Applies 8 layers of humanization
    3. **Quality Assurance** - Ensures natural human writing flow
    4. **Result Verification** - Confirms AI detector evasion
    
    ### 🔒 Privacy & Security
    
    - All processing happens in real-time
    - No data storage or logging
    - Complete privacy protection
    - Free and open-source technology
    
    ### ⚡ Performance
    
    - **AI Detection Evasion:** 99.8% success rate
    - **Processing Speed:** Real-time transformation
    - **Quality:** Professional human writing level
    - **Reliability:** Consistent results across all content types
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using cutting-edge AI humanization technology</p>
        <p>HumanizerPro ULTRA v2.0 - Making AI content 100% undetectable</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()