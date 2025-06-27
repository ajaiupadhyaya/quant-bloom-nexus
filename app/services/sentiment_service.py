from transformers import pipeline
from typing import List, Dict
import torch

class SentimentService:
    """
    A service for performing sentiment analysis on financial news headlines
    using a pre-trained model from the Hugging Face Transformers library.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initializes the SentimentService with a specified pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model to use.
                              Defaults to "ProsusAI/finbert" for financial sentiment.
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        
        print(f"Loading sentiment model '{self.model_name}' on device: {'cuda' if self.device == 0 else 'cpu'}")
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device # Use GPU if available
            )
            print("Sentiment model loaded successfully.")
        except Exception as e:
            print(f"Error loading sentiment model {self.model_name}: {e}")
            print("Falling back to CPU if GPU was attempted or model not found.")
            self.device = -1 # Fallback to CPU
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device
            )

    def analyze_headlines(self, headlines: List[str]) -> List[Dict[str, str]]:
        """
        Analyzes the sentiment of a list of news headlines.

        Args:
            headlines (List[str]): A list of news headlines to analyze.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary
                                  contains the original 'headline' and its 'sentiment'
                                  ('positive', 'negative', or 'neutral').
        """
        if not headlines:
            return []

        # Perform sentiment analysis in batches for performance
        # The pipeline automatically handles batching, but explicit batching can be done for very large lists
        results = self.sentiment_pipeline(headlines)

        analyzed_results = []
        for i, headline in enumerate(headlines):
            sentiment_label = results[i]['label'].lower() # e.g., 'positive', 'negative', 'neutral'
            analyzed_results.append({
                "headline": headline,
                "sentiment": sentiment_label
            })
        return analyzed_results

# Example Usage:
if __name__ == "__main__":
    sentiment_analyzer = SentimentService()

    sample_headlines = [
        "Apple stock soars after record earnings report.",
        "Company X faces massive lawsuit over data breach.",
        "Market remains flat amidst economic uncertainty.",
        "New product launch expected to boost sales.",
        "Inflation concerns weigh on consumer spending."
    ]

    print("\nAnalyzing sample headlines:")
    sentiments = sentiment_analyzer.analyze_headlines(sample_headlines)
    for item in sentiments:
        print(f"Headline: {item['headline']}\nSentiment: {item['sentiment']}\n")

    # Test with an empty list
    print("\nAnalyzing empty list:")
    empty_sentiments = sentiment_analyzer.analyze_headlines([])
    print(empty_sentiments)
