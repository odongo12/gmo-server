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
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
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
        Classify and analyze a single article based on its summary and fact-check results
        
        Args:
            article (Dict[str, Any]): Article with summary and fact-check results
            
        Returns:
            Dict[str, Any]: Classification and analysis results
        """
        # Create classification prompt
        prompt = self._create_classification_prompt(article)
        
        # Try up to 3 times to get a valid response
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get response from Gemini
                st.info(f"Requesting classification from Gemini (attempt {attempt + 1}/{max_retries}) for: {article['url']}")
                response = self.model.generate_content(prompt)
                st.success(f"Successfully received response from Gemini for: {article['url']}")
                break
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    st.error(f"Failed to get response from Gemini for {article['url']} after {max_retries} attempts: {str(e)}")
                    return self._create_fallback_result(article)
                else:
                    st.warning(f"Attempt {attempt + 1} failed for {article['url']}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
        
        # Clean and parse JSON response
        try:
            # Check if response is empty
            if not response.text or response.text.strip() == "":
                st.warning(f"Empty response from Gemini for {article['url']}")
                return self._create_fallback_result(article)
            
            # Clean the response text to extract JSON
            cleaned_response = self._extract_json_from_response(response.text)
            
            # Check if cleaned response is empty
            if not cleaned_response or cleaned_response.strip() == "":
                st.warning(f"Could not extract JSON content from response for {article['url']}")
                st.info(f"Raw response: {response.text[:200]}...")
                return self._create_fallback_result(article)
            
            # Validate JSON structure before parsing
            if not self._validate_json_structure(cleaned_response):
                st.warning(f"Invalid JSON structure in response for {article['url']}")
                st.info(f"Cleaned response: {cleaned_response[:200]}...")
                return self._create_fallback_result(article)
            
            analysis = json.loads(cleaned_response)
            
            # Validate that required fields are present
            if not self._validate_analysis_fields(analysis):
                st.warning(f"Missing required fields in analysis for {article['url']}")
                analysis = self._fix_missing_analysis_fields(analysis)
            
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
            st.info(f"Raw response: {response.text[:200]}...")
            st.info(f"Cleaned response: {cleaned_response[:200]}...")
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
        Analyze and classify the following article based on its SUMMARY and fact-check results.
        DO NOT analyze the full content - focus only on the summary provided.
        
        Article URL: {article['url']}
        Title: {article.get('title', 'Untitled')}
        Summary: {article.get('summary', '')}
        Overall Fact Status: {article.get('overall_status', 'Unsure')}
        
        {fact_check_info}
        
        Please provide the following analysis in EXACT JSON format:
        {{
            "classification": "One of these categories: {', '.join(self.categories)}",
            "confidence": "One of: high, medium, low",
            "key_themes": ["List of main themes or topics discussed"],
            "analysis_notes": "Brief analysis of content quality and reliability",
            "sentiment": "One of: positive, negative, neutral, mixed",
            "credibility_score": 0.5
        }}
        
        Guidelines:
        - Classify based on the main topic/theme of the SUMMARY only
        - Consider the fact-check results when assessing credibility
        - Provide confidence level based on clarity and verifiability of claims
        - Identify key themes that appear in the summary
        - Assess overall sentiment and tone from the summary
        - Provide a credibility score between 0.0 (low) and 1.0 (high)
        
        CRITICAL: 
        - Respond ONLY with valid JSON
        - Do not include any markdown formatting, explanations, or additional text
        - Ensure all field values are properly quoted and formatted
        - Use double quotes for strings, not single quotes
        - Ensure arrays and objects are properly closed
        """
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON content from Gemini response text
        
        Args:
            response_text (str): Raw response from Gemini
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove markdown code blocks if present
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end != -1:
                return response_text[start:end].strip()
        
        # Remove markdown code blocks without language specifier
        if '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            if end != -1:
                return response_text[start:end].strip()
        
        # Try to find JSON-like content between curly braces
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return response_text[start:end+1].strip()
        
        # If no JSON structure found, return the original text
        return response_text.strip()
    
    def _validate_json_structure(self, json_str: str) -> bool:
        """
        Validate if a string contains valid JSON structure
        
        Args:
            json_str (str): String to validate
            
        Returns:
            bool: True if valid JSON structure, False otherwise
        """
        try:
            # Try to parse as JSON
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    def _validate_analysis_fields(self, analysis: Dict[str, Any]) -> bool:
        """
        Validate that all required fields are present in the analysis
        
        Args:
            analysis (Dict[str, Any]): Analysis dictionary to validate
            
        Returns:
            bool: True if all required fields are present, False otherwise
        """
        required_fields = ['classification', 'confidence', 'key_themes', 'analysis_notes', 'sentiment', 'credibility_score']
        return all(field in analysis for field in required_fields)
    
    def _fix_missing_analysis_fields(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix missing fields in analysis with default values
        
        Args:
            analysis (Dict[str, Any]): Analysis dictionary with missing fields
            
        Returns:
            Dict[str, Any]: Fixed analysis dictionary
        """
        defaults = {
            'classification': 'Other',
            'confidence': 'medium',
            'key_themes': [],
            'analysis_notes': 'Analysis completed with default values for missing fields',
            'sentiment': 'neutral',
            'credibility_score': 0.5
        }
        
        # Fill in missing fields with defaults
        for field, default_value in defaults.items():
            if field not in analysis or analysis[field] is None:
                analysis[field] = default_value
        
        return analysis
    
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
