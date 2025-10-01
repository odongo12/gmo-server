import google.generativeai as genai
import os
from typing import List, Dict, Any
import streamlit as st
import json
import time


class SummaryAgent:
    """Agent responsible for generating concise summaries of scraped content using Google Gemini"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        else:
            self.model = None
        
        # Create temp folder at project root if it doesn't exist
        self.temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def summarize_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Generate summaries for a list of scraped articles
        
        Args:
            articles (List[Dict[str, str]]): List of articles with URL, content, and title
            
        Returns:
            List[Dict[str, Any]]: List of articles with summaries
        """
        if not self.model:
            st.error("GOOGLE_API_KEY not found in environment variables")
            return []
        
        summarized_articles = []
        total_articles = len(articles)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, article in enumerate(articles):
            status_text.text(f"Summarizing {i+1}/{total_articles}: {article['url']}")
            
            try:
                summary_result = self._summarize_single_article(article)
                summarized_articles.append(summary_result)
                st.success(f"✅ Successfully summarized: {article['url']}")
                
            except Exception as e:
                st.error(f"❌ Error summarizing {article['url']}: {str(e)}")
                # Add fallback result
                summarized_articles.append(self._create_fallback_result(article))
            
            # Update progress
            progress = (i + 1) / total_articles
            progress_bar.progress(progress)
            
            # Small delay to avoid rate limiting
            if i < total_articles - 1:
                time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Summarization complete! Successfully summarized {len(summarized_articles)} articles")
        
        # Save results to JSON file in temp folder
        if summarized_articles:
            timestamp = int(time.time())
            json_filename = f"summarized_articles_{timestamp}.json"
            json_filepath = os.path.join(self.temp_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(summarized_articles, f, ensure_ascii=False, indent=2)
            
            st.info(f"🔖 Summarized data saved to: `{json_filepath}`")
        
        return summarized_articles
    
    def _summarize_single_article(self, article: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a summary for a single article
        
        Args:
            article (Dict[str, str]): Article with URL, content, and title
            
        Returns:
            Dict[str, Any]: Article with summary
        """
        # Create summary prompt
        prompt = self._create_summary_prompt(article)
        
        # Get response from Gemini
        response = self.model.generate_content(prompt)
        
        # Extract summary from response
        summary = response.text.strip()
        
        return {
            'url': article['url'],
            'title': article.get('title', 'Untitled'),
            'content': article['content'],
            'summary': summary
        }
    
    def _create_summary_prompt(self, article: Dict[str, str]) -> str:
        """
        Create a summary prompt for Gemini
        
        Args:
            article (Dict[str, str]): Article to summarize
            
        Returns:
            str: Formatted prompt
        """
        return f"""
        Please provide a concise summary of the following article content.
        
        Title: {article.get('title', 'Untitled')}
        URL: {article['url']}
        Content: {article['content'][:3000]}
        
        Guidelines for the summary:
        - Keep it concise (2-3 sentences maximum)
        - Focus on the main points and key information
        - Be objective and factual
        - Avoid repetition and unnecessary details
        - Maintain the core message of the article
        
        Provide only the summary text without any additional formatting or labels.
        """
    
    def _create_fallback_result(self, article: Dict[str, str]) -> Dict[str, Any]:
        """
        Create a fallback result when summarization fails
        
        Args:
            article (Dict[str, str]): Original article
            
        Returns:
            Dict[str, Any]: Fallback summary result
        """
        return {
            'url': article['url'],
            'title': article.get('title', 'Untitled'),
            'content': article['content'],
            'summary': 'Summarization failed - unable to process content'
        }
