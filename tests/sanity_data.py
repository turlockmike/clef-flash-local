"""12-ticket sanity set shared by the tests and benchmarks: (message, intent, is_spam, urgency_level)."""
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": {
        "billing": "Questions about charges, invoices, refunds or payments",
        "technical": "Something is broken, erroring or not working",
        "account": "Login, password, profile or account settings",
        "sales": "Wants to buy, upgrade, or asks about pricing/plans"}},
    "spam": {"type": "noul", "instructions": "Is this message spam or a scam?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Not urgent", "Somewhat urgent", "Very urgent / emergency"]},
}
DATA = [
    ("I was charged twice for my subscription this month, please refund one.", "billing", False, 1),
    ("Our whole production site is down and returning 502 errors, customers can't check out!", "technical", False, 2),
    ("How do I reset my password? The email never arrives.", "account", False, 1),
    ("What does the enterprise plan cost for 200 seats?", "sales", False, 0),
    ("CONGRATULATIONS you won a $1000 gift card, click here to claim now!!!", "sales", True, 0),
    ("Can I get a copy of last year's invoices for my accountant? No rush.", "billing", False, 0),
    ("The API returns a 401 even with a valid key since this morning.", "technical", False, 1),
    ("I'd like to change the email address on my profile.", "account", False, 0),
    ("We're thinking about upgrading to Pro, does it include SSO?", "sales", False, 0),
    ("Your account has been suspended. Verify your bank login at http://secure-verify.xyz", "account", True, 1),
    ("Payment failed and our service will be cut off tonight, need help ASAP", "billing", False, 2),
    ("The mobile app crashes every time I open settings.", "technical", False, 1),
]


def record(msg):
    return {"state": {"message": msg}, "questions": QUESTIONS}
