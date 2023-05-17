from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a ChatBot instance
chatbot = ChatBot('ThaiBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot using the Thai corpus
trainer.train('chatterbot.corpus.thai')

# Get a response from the chatbot
response = chatbot.get_response('สวัสดี')

print('Bot:', response)
