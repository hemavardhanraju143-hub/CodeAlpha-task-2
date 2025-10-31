𝗡𝗮𝗺𝗲: Hemavardhan raju
𝗦𝘁𝘂𝗱𝗲𝗻𝘁 𝗜𝗗: CA/SE1/18250
𝗗𝗼𝗺𝗮𝗶𝗻: Artificial Intelligence
𝗗𝘂𝗿𝗮𝘁𝗶𝗼𝗻: 10th October 2025 to 10th November 2025
# CodeAlpha-task-2
# --- Import Libraries ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Download NLTK Data ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Sample FAQ Dataset ---
faqs = [
    {"question": "What is your return policy?", "answer": "You can return items within 30 days."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship worldwide."},
    {"question": "How can I track my order?", "answer": "Use the tracking link sent to your email."},
    {"question": "What payment methods do you accept?", "answer": "We accept credit cards, PayPal, and Apple Pay."},
    {"question": "How can I contact customer support?", "answer": "You can email us at support@example.com or call 123-456-7890."}
]

# --- Preprocessing Function ---
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Prepare FAQ Vectors ---
questions = [preprocess(faq["question"]) for faq in faqs]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# --- Function to Get Best Matching Answer ---
def get_answer(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarity.argmax()
    return faqs[best_match_idx]["answer"]

# --- Chatbot Loop ---
print("FAQ Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    answer = get_answer(user_input)
    print("Chatbot:", answer)
<img width="762" height="477" alt="image" src="https://github.com/user-attachments/assets/785a7f9a-4af9-4ebf-adfe-1bc3e8c40a65" />
