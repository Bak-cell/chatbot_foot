import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Téléchargements des ressources NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Lecture du fichier et traitement de base du texte
with open('football_info.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenisation des phrases
sentences = sent_tokenize(data)

# Initialisation des stopwords et du lemmatiseur en dehors des fonctions pour éviter les répétitions
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

# Fonction de prétraitement pour chaque phrase
def preprocess(sentence):
    # Tokenisation en mots, suppression des stopwords et ponctuations, et lemmatisation
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(sentence)
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return words

# Prétraitement de chaque phrase dans le texte
corpus = [preprocess(sentence) for sentence in sentences]

# Fonction pour trouver la phrase la plus pertinente pour une requête donnée
def get_most_relevant_sentence(query):
    # Prétraitement de la requête
    query = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = ""
    
    for sentence in corpus:
        # Calcul de la similarité entre la requête et chaque phrase
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    
    return most_relevant_sentence

# Fonction du chatbot
def chatbot(question):
    return get_most_relevant_sentence(question)

# Interface utilisateur avec Streamlit
def main():
    st.title("Chatbot Football")
    st.write("Bonjour ! Posez-moi une question sur le football.")
    
    question = st.text_input("Vous :")
    if st.button("Envoyer"):
        response = chatbot(question)
        st.write("Chatbot : " + response)

if __name__ == "__main__":
    main()
