"""Small starter corpus based on the user's explicit domain choices, not personal facts."""
from __future__ import annotations

DOMAIN_WORDS = {
    "ai-coding": ["Gemini", "Kimi", "ChatGPT", "DeepSeek", "CPA", "API", "GitHub", "Python", "Lipflow"],
    "finance": ["earnings", "valuation", "cash flow", "private equity", "Jane Street", "Optiver"],
    "education": ["Imperial College London", "LSE", "UCAS", "mathematics", "tutorial", "lecture"],
}
DOMAIN_PHRASES = {
    "ai-coding": [
        "Can you explain this Python function in simple terms?",
        "Please compare Gemini and Kimi for this coding task.",
        "Do not change the API without checking the tests.",
    ],
    "finance": [
        "Please summarize the earnings report and highlight the risks.",
        "Can you compare their valuation and free cash flow?",
        "I would like to learn more about private equity.",
    ],
    "education": [
        "Could we move the tutorial to tomorrow afternoon?",
        "Please explain the main idea from today's lecture.",
        "I need to check the application deadline first.",
    ],
}
GENERAL_PHRASES = [
    "Thanks for the update, I will check it tomorrow.",
    "Could you send me the document when you have time?",
    "I am not available until four thirty this afternoon.",
]


def corpus(domains):
    words, phrases = [], list(GENERAL_PHRASES)
    for domain in dict.fromkeys(domains):
        words.extend(DOMAIN_WORDS.get(domain, []))
        phrases.extend(DOMAIN_PHRASES.get(domain, []))
    return list(dict.fromkeys(words)), list(dict.fromkeys(phrases))


def apply_choices(domains):
    from .customize import import_plain_text
    words, phrases = corpus(domains)
    return import_plain_text(words=words, phrases=phrases)
