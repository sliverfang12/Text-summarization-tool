from transformers import pipeline

# Load the summarizer pipeline from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(text, max_length=150, min_length=50):
    """
    Summarize the input article using the BART model.
    
    Parameters:
    - text (str): The article to summarize.
    - max_length (int): Maximum length of the summary.
    - min_length (int): Minimum length of the summary.
    
    Returns:
    - str: The summarized text.
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example article text (replace with your lengthy article text)
article_text = """
Your lengthy article text goes here. Replace this placeholder with the actual content of the article. 
The tool will summarize this long article into a short, concise summary using advanced Natural Language Processing techniques. 
Make sure to provide meaningful and informative text to get accurate and relevant summaries.
"""

# Call the function to summarize the article
summary = summarize_article(article_text)

# Output the original article and the summary
print("Original Article:\n")
print(article_text)
print("\nSummary:\n")
print(summary)
