import google.generativeai as genai
import os
from typing import List, Dict, Any
import streamlit as st
import json
import time
from .summary_agent import SummaryAgent
from .fact_check import FactCheckAgent


class AnalysisAgent:
    """Main agent responsible for orchestrating analysis, classification, and fact-checking workflow"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
        
        # Initialize sub-agents
        self.summary_agent = SummaryAgent()
        self.fact_check_agent = FactCheckAgent()
        
        # Predefined categories for classification
        self.categories = [
            "Health", "Environmental", "Social economics", "Conspiracy theory",
            "Corporate control", "Ethical/religious issues", "Seed ownership",
            "Scientific authority", "Other"
        ]
        
        # Create temp folder at project root if it doesn't exist
        self.temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def analyze_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Complete analysis workflow: scraping → summarization → fact-checking → classification
        
        Args:
            articles (List[Dict[str, str]]): List of articles with URL and content
            
        Returns:
            List[Dict[str, Any]]: List of fully analyzed articles with all results
        """
        if not self.model:
            st.error("GOOGLE_API_KEY not found in environment variables")
            return []
        
        st.info("🚀 Starting comprehensive analysis workflow...")
        
        # Step 1: Generate summaries
        st.subheader("📝 Step 1: Generating Summaries")
        summarized_articles = self.summary_agent.summarize_articles(articles)
        
        if not summarized_articles:
            st.error("Summarization failed. Cannot proceed with analysis.")
            return []
        
        # Step 2: Fact-check claims
        st.subheader("🔍 Step 2: Fact-Checking Claims")
        fact_checked_articles = self.fact_check_agent.fact_check_articles(summarized_articles)
        
        if not fact_checked_articles:
            st.error("Fact-checking failed. Proceeding with classification only.")
            fact_checked_articles = summarized_articles
        
        # Step 3: Classify and analyze
        st.subheader("🏷️ Step 3: Classification and Analysis")
        final_analyzed_articles = self._classify_and_analyze(fact_checked_articles)
        
        # Save final results
        if final_analyzed_articles:
            timestamp = int(time.time())
            json_filename = f"final_analysis_{timestamp}.json"
            json_filepath = os.path.join(self.temp_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_analyzed_articles, f, ensure_ascii=False, indent=2)
            
            st.info(f"🔖 Final analysis saved to: `{json_filepath}`")
        
        st.success("✅ Complete analysis workflow finished!")
        return final_analyzed_articles
    
    def _classify_and_analyze(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify and analyze articles using Gemini NLP
        
        Args:
            articles (List[Dict[str, Any]]): Articles with summaries and fact-check results
            
        Returns:
            List[Dict[str, Any]]: Articles with classification and analysis
        """
        analyzed_articles = []
        total_articles = len(articles)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, article in enumerate(articles):
            status_text.text(f"Classifying {i+1}/{total_articles}: {article['url']}")
            
            try:
                classification_result = self._classify_single_article(article)
                analyzed_articles.append(classification_result)
                st.success(f"✅ Successfully classified: {article['url']}")
                
            except Exception as e:
                st.error(f"❌ Error classifying {article['url']}: {str(e)}")
                # Add fallback result
                analyzed_articles.append(self._create_fallback_result(article))
            
            # Update progress
            progress = (i + 1) / total_articles
            progress_bar.progress(progress)
            
            # Small delay to avoid rate limiting
            if i < total_articles - 1:
                time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Classification complete! Successfully analyzed {len(analyzed_articles)} articles")
        return analyzed_articles
    
    def _classify_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify and analyze a single article
        
        Args:
            article (Dict[str, Any]): Article with summary and fact-check results
            
        Returns:
            Dict[str, Any]: Classification and analysis results
        """
        # Create classification prompt
        prompt = self._create_classification_prompt(article)
        
        # Get response from Gemini
        response = self.model.generate_content(prompt)
        
        # Parse JSON response
        try:
            analysis = json.loads(response.text)
            
            return {
                'url': article['url'],
                'title': article.get('title', 'Untitled'),
                'content': article.get('content', ''),
                'summary': article.get('summary', ''),
                'claims': article.get('claims', []),
                'fact_check_results': article.get('fact_check_results', []),
                'overall_fact_status': article.get('overall_status', 'Unsure'),
                'classification': analysis.get('classification', 'Other'),
                'confidence': analysis.get('confidence', 'medium'),
                'key_themes': analysis.get('key_themes', []),
                'analysis_notes': analysis.get('analysis_notes', ''),
                'sentiment': analysis.get('sentiment', 'neutral'),
                'credibility_score': analysis.get('credibility_score', 0.5)
            }
            
        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse JSON response for {article['url']}: {str(e)}")
            return self._create_fallback_result(article)
    
    def _create_classification_prompt(self, article: Dict[str, Any]) -> str:
        """
        Create a comprehensive classification prompt for Gemini
        
        Args:
            article (Dict[str, Any]): Article to classify
            
        Returns:
            str: Formatted prompt
        """
        # Prepare fact-check information
        fact_check_info = ""
        if article.get('fact_check_results'):
            fact_check_info = "Fact-check Results:\n"
            for i, result in enumerate(article['fact_check_results'][:3], 1):  # Show top 3
                fact_check_info += f"{i}. Claim: {result['claim'][:100]}...\n"
                fact_check_info += f"   Status: {result['status']} (Rating: {result['rating']})\n"
                fact_check_info += f"   Publisher: {result['publisher']}\n\n"
        
        return f"""
        Analyze and classify the following article based on its content, summary, and fact-check results.
        
        Article URL: {article['url']}
        Title: {article.get('title', 'Untitled')}
        Summary: {article.get('summary', '')}
        Overall Fact Status: {article.get('overall_status', 'Unsure')}
        
        {fact_check_info}
        
        Please provide the following analysis in JSON format:
        {{
            "classification": "One of these categories: {', '.join(self.categories)}",
            "confidence": "One of: high, medium, low",
            "key_themes": ["List of main themes or topics discussed"],
            "analysis_notes": "Brief analysis of content quality and reliability",
            "sentiment": "One of: positive, negative, neutral, mixed",
            "credibility_score": 0.5
        }}
        
        Guidelines:
        - Classify based on the main topic/theme of the article
        - Consider the fact-check results when assessing credibility
        - Provide confidence level based on clarity and verifiability of claims
        - Identify key themes that appear in the content
        - Assess overall sentiment and tone
        - Provide a credibility score between 0.0 (low) and 1.0 (high)
        
        Respond only with valid JSON.
        """
    
    def _create_fallback_result(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback result when classification fails
        
        Args:
            article (Dict[str, Any]): Original article
            
        Returns:
            Dict[str, Any]: Fallback classification result
        """
        return {
            'url': article['url'],
            'title': article.get('title', 'Untitled'),
            'content': article.get('content', ''),
            'summary': article.get('summary', ''),
            'claims': article.get('claims', []),
            'fact_check_results': article.get('fact_check_results', []),
            'overall_fact_status': article.get('overall_status', 'Unsure'),
            'classification': 'Other',
            'confidence': 'low',
            'key_themes': [],
            'analysis_notes': 'Classification failed due to processing error',
            'sentiment': 'neutral',
            'credibility_score': 0.3
        }
    
    def get_analysis_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for analyzed articles
        
        Args:
            articles (List[Dict[str, Any]]): List of analyzed articles
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not articles:
            return {}
        
        # Count by classification
        classification_counts = {}
        for category in self.categories:
            classification_counts[category] = sum(
                1 for a in articles if a.get('classification') == category
            )
        
        # Count by fact status
        fact_status_counts = {
            'Fact': sum(1 for a in articles if a.get('overall_fact_status') == 'Fact'),
            'Myth': sum(1 for a in articles if a.get('overall_fact_status') == 'Myth'),
            'Unsure': sum(1 for a in articles if a.get('overall_fact_status') == 'Unsure')
        }
        
        # Count by confidence
        confidence_counts = {
            'high': sum(1 for a in articles if a.get('confidence') == 'high'),
            'medium': sum(1 for a in articles if a.get('confidence') == 'medium'),
            'low': sum(1 for a in articles if a.get('confidence') == 'low')
        }
        
        # Count by sentiment
        sentiment_counts = {
            'positive': sum(1 for a in articles if a.get('sentiment') == 'positive'),
            'negative': sum(1 for a in articles if a.get('sentiment') == 'negative'),
            'neutral': sum(1 for a in articles if a.get('sentiment') == 'neutral'),
            'mixed': sum(1 for a in articles if a.get('sentiment') == 'mixed')
        }
        
        # Calculate average credibility score
        credibility_scores = [a.get('credibility_score', 0.5) for a in articles]
        avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
        
        return {
            'total_articles': len(articles),
            'classification_counts': classification_counts,
            'fact_status_counts': fact_status_counts,
            'confidence_counts': confidence_counts,
            'sentiment_counts': sentiment_counts,
            'average_credibility_score': round(avg_credibility, 3),
            'successful_analyses': sum(1 for a in articles if a.get('classification') != 'Other' or a.get('analysis_notes') != 'Classification failed due to processing error')
        } 