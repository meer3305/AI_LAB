"""
Simple Natural Language Processing Tools
Author: AI Lab
Description: Basic NLP techniques including text preprocessing and sentiment analysis
"""

import re
import string
from collections import Counter, defaultdict
import math


class TextPreprocessor:
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text):
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)
    
    def tokenize(self, text):
        """Split text into tokens"""
        return text.split()
    
    def remove_stop_words(self, tokens):
        """Remove common stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def preprocess(self, text, remove_numbers=True, remove_stop_words=True):
        """Complete preprocessing pipeline"""
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        
        if remove_numbers:
            text = self.remove_numbers(text)
        
        tokens = self.tokenize(text)
        
        if remove_stop_words:
            tokens = self.remove_stop_words(tokens)
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens


class SimpleSentimentAnalyzer:
    def __init__(self):
        # Simple sentiment lexicon
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'perfect', 'beautiful', 'love', 'like',
            'happy', 'pleased', 'satisfied', 'delighted', 'thrilled', 'excited',
            'positive', 'optimistic', 'cheerful', 'glad', 'joyful', 'magnificent'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'sad', 'angry', 'frustrated', 'disappointed', 'annoyed', 'upset',
            'negative', 'pessimistic', 'depressed', 'worried', 'concerned',
            'poor', 'inferior', 'inadequate', 'unsatisfactory', 'disgusting'
        }
        
        self.preprocessor = TextPreprocessor()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        tokens = self.preprocessor.preprocess(text, remove_stop_words=False)
        
        positive_score = sum(1 for token in tokens if token in self.positive_words)
        negative_score = sum(1 for token in tokens if token in self.negative_words)
        
        # Calculate sentiment
        total_sentiment_words = positive_score + negative_score
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            confidence = 0.5
        else:
            sentiment_score = (positive_score - negative_score) / len(tokens)
            confidence = total_sentiment_words / len(tokens)
            
            if sentiment_score > 0:
                sentiment = "positive"
            elif sentiment_score < 0:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'positive_words_found': [w for w in tokens if w in self.positive_words],
            'negative_words_found': [w for w in tokens if w in self.negative_words]
        }


class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(int)
        self.vocabulary = set()
        self.preprocessor = TextPreprocessor()
    
    def extract_ngrams(self, tokens):
        """Extract n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, texts):
        """Train the n-gram model on a collection of texts"""
        for text in texts:
            tokens = self.preprocessor.preprocess(text)
            self.vocabulary.update(tokens)
            
            # Add start and end tokens
            padded_tokens = ['<START>'] * (self.n - 1) + tokens + ['<END>']
            
            ngrams = self.extract_ngrams(padded_tokens)
            for ngram in ngrams:
                self.ngrams[ngram] += 1
    
    def get_probability(self, ngram):
        """Get probability of an n-gram"""
        if len(ngram) != self.n:
            return 0.0
        
        # Simple maximum likelihood estimation
        count = self.ngrams[ngram]
        total_ngrams = sum(self.ngrams.values())
        
        if total_ngrams == 0:
            return 0.0
        
        return count / total_ngrams
    
    def generate_text(self, length=10, start_tokens=None):
        """Generate text using the n-gram model"""
        if not self.ngrams:
            return "Model not trained yet."
        
        if start_tokens is None:
            start_tokens = ['<START>'] * (self.n - 1)
        
        generated = start_tokens[:]
        
        for _ in range(length):
            # Get context (last n-1 tokens)
            context = tuple(generated[-(self.n-1):])
            
            # Find all n-grams that start with this context
            candidates = []
            for ngram, count in self.ngrams.items():
                if ngram[:-1] == context:
                    candidates.extend([ngram[-1]] * count)
            
            if not candidates:
                break
            
            # Randomly select next token (weighted by frequency)
            import random
            next_token = random.choice(candidates)
            
            if next_token == '<END>':
                break
            
            generated.append(next_token)
        
        return ' '.join(generated[(self.n-1):])


def demonstrate_text_preprocessing():
    """Demonstrate text preprocessing"""
    print("=== Text Preprocessing Demonstration ===")
    
    sample_text = "Hello World! This is a SAMPLE text with Numbers 123 and Punctuation!!!"
    
    preprocessor = TextPreprocessor()
    
    print(f"Original text: {sample_text}")
    print(f"Lowercase: {preprocessor.to_lowercase(sample_text)}")
    print(f"No punctuation: {preprocessor.remove_punctuation(sample_text.lower())}")
    print(f"No numbers: {preprocessor.remove_numbers(preprocessor.remove_punctuation(sample_text.lower()))}")
    
    tokens = preprocessor.preprocess(sample_text)
    print(f"Final tokens: {tokens}")


def demonstrate_sentiment_analysis():
    """Demonstrate sentiment analysis"""
    print("\n=== Sentiment Analysis Demonstration ===")
    
    analyzer = SimpleSentimentAnalyzer()
    
    test_texts = [
        "I love this product! It's absolutely amazing and wonderful!",
        "This is terrible. I hate it so much. Worst experience ever.",
        "The weather is okay today. Nothing special.",
        "Great job! Excellent work. I'm very happy with the results.",
        "Bad service, poor quality, very disappointed and frustrated."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print(f"Positive words: {result['positive_words_found']}")
        print(f"Negative words: {result['negative_words_found']}")


def demonstrate_ngram_model():
    """Demonstrate n-gram language model"""
    print("\n=== N-gram Language Model Demonstration ===")
    
    # Sample training texts
    training_texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "A cat and a dog played together",
        "The mat was comfortable for the cat",
        "Dogs and cats are pets",
        "The park is a good place for dogs",
        "Cats like to sit on mats",
        "Animals play in the park"
    ]
    
    print("Training texts:")
    for i, text in enumerate(training_texts, 1):
        print(f"{i}. {text}")
    
    # Train bigram model
    bigram_model = NGramModel(n=2)
    bigram_model.train(training_texts)
    
    print(f"\nTrained bigram model with {len(bigram_model.ngrams)} unique bigrams")
    
    # Show some bigram probabilities
    print("\nSample bigram probabilities:")
    sample_bigrams = [('the', 'cat'), ('cat', 'sat'), ('on', 'the'), ('the', 'mat')]
    
    for bigram in sample_bigrams:
        prob = bigram_model.get_probability(bigram)
        print(f"P({bigram[0]} {bigram[1]}) = {prob:.3f}")
    
    # Generate text
    print(f"\nGenerated text:")
    for i in range(3):
        generated = bigram_model.generate_text(length=8)
        print(f"{i+1}. {generated}")


def analyze_text_statistics():
    """Analyze text statistics"""
    print("\n=== Text Statistics Analysis ===")
    
    sample_text = """
    Artificial intelligence is transforming the world. Machine learning algorithms
    can process vast amounts of data and learn patterns. Deep learning neural networks
    are particularly effective for image recognition and natural language processing.
    The future of AI holds great promise for solving complex problems.
    """
    
    preprocessor = TextPreprocessor()
    tokens = preprocessor.preprocess(sample_text)
    
    # Word frequency analysis
    word_freq = Counter(tokens)
    
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique tokens: {len(word_freq)}")
    print(f"Average word length: {sum(len(word) for word in tokens) / len(tokens):.2f}")
    
    print(f"\nTop 10 most frequent words:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")
    
    # Calculate lexical diversity
    lexical_diversity = len(word_freq) / len(tokens)
    print(f"\nLexical diversity: {lexical_diversity:.3f}")


if __name__ == "__main__":
    demonstrate_text_preprocessing()
    demonstrate_sentiment_analysis()
    demonstrate_ngram_model()
    analyze_text_statistics()