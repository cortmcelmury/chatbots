import numpy as np
import pandas as pd

# Import datasets
# ATIS dataset
atis_intents = pd.read_csv(
    'https://raw.githubusercontent.com/cortmcelmury/chatbots/master/data/atis_intents.csv, header=None)
atis_intents_test = pd.read_csv(
    'https://raw.githubusercontent.com/cortmcelmury/chatbots/master/data/atis_intents_test.csv', header=None)
atis_intents_train = pd.read_csv(
    'https://raw.githubusercontent.com/cortmcelmury/chatbots/master/data/atis_intents_train.csv', header=None)

atis_intents.head()

# Hotels data
# db file with only one table - 'hotels'
# Donwload to your machine and point the connect string to the location
import sqlite3
conn = sqlite3.connect(
    'https://github.com/cortmcelmury/chatbots/blob/master/data/hotels.db')
c = conn.cursor()

c.execute("SELECT * FROM hotels")
hotels = c.fetchall()
hotels

c.execute("PRAGMA table_info(hotels)")
c.fetchall()

# Ran the below to rename column 'area' to column 'location' which is used in the examples
#c.execute("ALTER TABLE hotels RENAME COLUMN area TO location")


################################################################################
########################## USING WORD VECTORS ##################################
################################################################################
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("cat")

# .similarity() computes cosine similarity, which is the angle between two vectors
# ranges from -1 (opposite) to 1 (exactly the same) with 0 being perpendicular
doc.similarity(nlp("can"))
doc.similarity(nlp("dog"))

################################################################################
### word vectors with spaCy ###
# Use the ATIS dataset, which contains thousands of sentences from real people
# interacting with a flight booking system.
# The user utterances are available in the list sentences, and the corresponding intents in labels.
# Your job is to create a 2D array X with as many rows as there are sentences in
# the dataset, where each row is a vector describing that sentence.

sentences = list(atis_intents[1])
labels = list(atis_intents[0])

sentences_train = list(atis_intents_train[1])
labels_train = list(atis_intents_train[0])

sentences_test = list(atis_intents_test[1])
labels_test = list(atis_intents_test[0])

# Load the spacy model: nlp
# spacy model 'en_core_web_sm' does not contain actual word vectors –
# but to make them more useful, they take advantage of the shared context-sensitive
# token vectors used by the tagger, parser and NER.
# This means that you can still use the similarity() methods, even without word vectors.
# spacy model 'en_core_web_md' has the word vectors - takes longer to load and perform operations with

nlp = spacy.load('en_core_web_md')

# Calculate the length of sentences
n_sentences = len(sentences)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

# Iterate over the sentences
for idx, sentence in enumerate(sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector


################################################################################
# Supervised learning
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
# classifier predicts the 'intent label' given a sentence

# Create set X_train that transforms sentences to vectors
X_train_shape = (len(sentences_train), nlp.vocab.vectors_length)
X_train = np.zeros(X_train_shape)
X_train_shape
sentences_train[:2]

X_test_shape = (len(sentences_test), nlp.vocab.vectors_length)
X_test = np.zeros(X_test_shape)
X_test_shape

# Gets a vector for each sentence, which is an average of each word vector in the sentence
for i, sentence in enumerate(sentences_train):
    X_train[i, :] = nlp(sentence).vector

for i, sentence, in enumerate(sentences_test):
    X_test[i, :] = nlp(sentence).vector


labels_train
########################### Transform labels ###################################
le = LabelEncoder()
y_train = le.fit_transform(labels_train)
y_test = le.transform(labels_test)
y_train

list(le.inverse_transform(y_train))

# Intent classification with sklearn
# An array X containing vectors describing each of the sentences in the ATIS dataset
# has been created for you, along with a 1D array y containing the labels.
# The labels are integers corresponding to the intents in the dataset.
# For example, label 0 corresponds to the intent atis_flight.

# Import SVC
from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC(C=1)

# Fit the classifier using the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        n_correct += 1

print("Predicted {0} correctly out of {1} test examples".format(
    n_correct, len(y_test)))

################################################################################
############################ ENTITY EXTRACTION #################################
# Keywords don't work for entities you haven't seen before
# Use contextual clues

# Using spaCy's entity recognizer
# In this exercise, you'll use spaCy's built-in entity recognizer to extract
# names, dates, and organizations from search queries.

# Your job is to define a function called extract_entities(), which takes in a
# single argument message and returns a dictionary with the included
# entity types as keys, and the extracted entities as values.
# The included entity types are contained in a list called include_entities.

import spacy
nlp = spacy.load('en_core_web_md')

# Define included_entities
include_entities = ['DATE', 'ORG', 'PERSON']

# Define extract_entities()


def extract_entities(message):
    # Create a dict to hold the entities
    ents = dict.fromkeys(include_entities)
    # Create a spacy document
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents


print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))


# Assigning roles using spaCy's parser
# Use spaCy's powerful syntax parser to assign roles to the entities in your users' messages.
# To do this, you'll define two functions, find_parent_item() and assign_colors().
# In doing so, you'll use a parse tree to assign roles, similar to how Alan did in the video.

# Recall that you can access the ancestors of a word using its .ancestors attribute.

# Create the document
doc = nlp("let's see that jacket in red and some blue jeans")

# Iterate over parents in parse tree until an item entity is found


def find_parent_item(word):
    # Iterate over the word's ancestors
    for parent in word.ancestors:
        # Check for an "item" entity
        if entity_type(parent) == "item":
            return parent.text
    return None

# For all color entities, find their parent item


def assign_colors(doc):
    # Iterate over the document
    for word in doc:
        # Check for "color" entities
        if entity_type(word) == "color":
            # Find the parent
            item = find_parent_item(word)
            print("item: {0} has color : {1}".format(item, word))


# Assign the colors
assign_colors(doc)

################################################################################
################## ROBUST LANGUAGE UNDERSTANDING WITH rasa NLU #################
################################################################################
# Rasa data format - JSON
# NLU = Natural Language Understanding
import rasa_nlu
help(rasa_nlu)

# Rasa NLU
# In this exercise, you'll use Rasa NLU to create an interpreter, which parses
# incoming user messages and returns a set of entities.
# Your job is to train an interpreter using the MITIE entity recognition model in Rasa NLU.

# Import necessary modules
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer

# Create args dictionary
args = {"pipeline": "spacy_sklearn"}

# Create a configuration and trainer
config = RasaNLUModelConfig(configuration_values=args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("./training_data.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Test the interpreter
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))


# Data-efficient entity recognition
# Most systems for extracting entities from text are built to extract 'Universal'
# things like names, dates, and places. But you probably don't have enough
# training data for your bot to make these systems perform well!

# In this exercise, you'll activate the MITIE entity recognizer inside Rasa to
# extract restaurants-related entities using a very small amount of training data.

# Import necessary modules
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer

pipeline = [
    "nlp_spacy",
    "tokenizer_spacy",
    "ner_crf"
]

# Create a config that uses this pipeline
config = RasaNLUModelConfig(configuration_values={'pipeline': pipeline})

# Create a trainer that uses this config
trainer = Trainer(config)

# Create an interpreter by training the model
interpreter = trainer.train(
    'C:\\Users\cmcelmury\Documents\Python\Datasets\\rasa_data\\testData.json')

# Parse some messages
print(interpreter.parse("show me Chinese food in the centre of town"))
print(interpreter.parse("I want an Indian restaurant in the west"))
print(interpreter.parse("are there any good pizza places in the center?"))


################################################################################
################### VIRTUAL ASSISTANTS & ACCESSING DATA ########################
################################################################################

# SQL statements in Python
# It's time to begin writing SQL queries! In this exercise, your job is to run
# a query against the hotels database to find all the expensive hotels in the south.

# You should be careful about SQL injection. Here, you'll pass parameters
# the safe way: As an extra tuple argument to the .execute() method.
# This ensures malicious code can't be injected into your query.

# Import sqlite3
import sqlite3

# Open connection to DB
conn = sqlite3.connect(
    'https://github.com/cortmcelmury/chatbots/blob/master/data/hotels.db')
# Create a cursor
c = conn.cursor()

# Define area and price
area, price = "south", "hi"
t = (area, price)

# Execute the query
c.execute('SELECT * FROM hotels WHERE area=? AND price=?', t)

# Print the results
print(c.fetchall())


# Creating queries from parameters
# Now you're going to implement a more powerful function for querying the
# hotels database. The goal is for that function to take arguments that
# can later be specified by other parts of your code.

# More specifically, your job is to define a find_hotels() function which takes
# a single argument - a dictionary of column names and values -
# and returns a list of matching hotels from the database.

# Define find_hotels()
def find_hotels(params):
    # Create the base query
    query = 'SELECT * FROM hotels'
    # Add filter clauses for each of the parameters
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params]
        query += " WHERE " + " AND ".join(filters)
    # Create the tuple of values
    t = tuple(params.values())

    # Open connection to DB
    conn = sqlite3.connect(
        'https://github.com/cortmcelmury/chatbots/blob/master/data/hotels.db')
    # Create a cursor
    c = conn.cursor()
    # Execute the query
    c.execute(query, t)
    # Return the results
    return c.fetchall()


# Test it out
# Create the dictionary of column names and values
params = {
    "area": 'south',
    'price': 'lo'
}

# Find the hotels that match the parameters
print(find_hotels(params))


####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# Creating SQL from natural language
# Now you'll write a respond() function that can handle messages like
# "I want an expensive hotel in the south of town" and respond appropriately
# according to the number of matching results in a database.
# This is an important functionality for any database-backed chatbot.

# BELOW IS HOW TO RECREATE THE INTERPRETER OBJECT
# Get the test data from online - can create your own with a json file or online
# Download and save in proper directory

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config

training_data = load_data(
    'C:\\Users\cmcelmury\Documents\Python\Datasets\\rasa_data\\testData.json')
trainer = Trainer(config.load(
    'C:\\Users\cmcelmury\Documents\Python\Datasets\\rasa_data\\nlu_config.yml'))
trainer.train(training_data)
# Returns the directory the model is stored in
model_directory = trainer.persist('./Models/rasa_nlu/')

from rasa_nlu.model import Interpreter
interpreter = Interpreter.load(model_directory)

####################################################################################################
####################################################################################################
####################################################################################################


responses_3 = ["I'm sorry :( I couldn't find anything like that",
               '{} is a great hotel!',
               '{} or {} would work!',
               '{} is one option, but I know others too :)']


# Define respond()
def respond(message):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    # Initialize an empty params dictionary
    params = {}
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find hotels that match the dictionary
    results = find_hotels(params)
    # Get the names of the hotels and index of the response
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Select the nth element of the responses array
    return responses_3[n].format(*names)


# Test the respond() function
print(respond("I want an expensive hotel in the south of town"))


# Refining your search
# Now you'll write a bot that allows users to add filters incrementally,
# just in case they don't specify all of their preferences in one message.

# To do this, initialize an empty dictionary params outside of your respond()
# function (as opposed to inside the function, like in the previous exercise).
# Your respond() function will take in this dictionary as an argument.


# Define a respond function, taking the message and existing params as input
def respond(message, params):
    # Extract the entities
    entities = interpreter.parse(message)['entities']
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find the hotels
    results = find_hotels(params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Return the appropriate response
    return responses_3[n].format(*names), params


# Initialize params dictionary
params = {}

# Pass the messages to the bot
for message in ["I want an expensive hotel", "in the north of town"]:
    print("USER: {}".format(message))
    response, params = respond(message, params)
    print("BOT: {}".format(response))

# Note here - error returns "no such column location" - I think I could rename
# location to area to match the table's setup and it would run ok


# Basic negation
# Quite often, you'll find your users telling you what they don't want -
# and that's important to understand! In general, negation is a difficult problem
# in NLP. Here, we'll take a very simple approach that works for many cases.

# A list of tests called tests is defined below. Each test is a tuple consisting of:

# A string containing a message with entities.
# A dictionary containing the entities as keys and a Boolean saying whether they are negated as the key.

# Your job is to define a function called negated_ents() which looks for negated entities in a message.

tests = [("no I don't want to be in the south", {'south': False}),
         ('no it should be in the south', {'south': True}),
         ('no in the south not the north', {'north': False, 'south': True}),
         ('not north', {'north': False})]

# Define negated_ents()


def negated_ents(phrase):
    # Extract the entities using keyword matching
    ents = [e for e in ["south", "north"] if e in phrase]
    # Find the index of the final character of each entity
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    # Initialise a list to store sentence chunks
    chunks = []
    # Take slices of the sentence up to and including each entitiy
    start = 0
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    # Iterate over the chunks and look for entities
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                # If the entity contains a negation, assign the key to be False
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result


# Check that the entities are correctly assigned as True or False
for test in tests:
    print(negated_ents(test[0]) == test[1])

# Filtering with excluded slots
# Now you're going to put together some of the ideas from previous exercises
# in order to allow users to tell your bot about what they do and do not want,
# split across multiple messages.

# The negated_ents() function has already been defined for you. Additionally,
# a slightly tweaked version of the find_hotels() function, which accepts a
# neg_params dictionary in addition to a params dictionary, has been defined.


def negated_ents(phrase, ent_vals):
    ents = [e for e in ent_vals if e in phrase]
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    start = 0
    chunks = []
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result


def find_hotels(params, neg_params):
    query = 'SELECT * FROM hotels'
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params] + \
            ["{}!=?".format(k) for k in neg_params]
        query += " WHERE " + " and ".join(filters)
    t = tuple(params.values())

    # open connection to DB
    conn = sqlite3.connect(
        'https://github.com/cortmcelmury/chatbots/blob/master/data/hotels.db')
    # create a cursor
    c = conn.cursor()
    c.execute(query, t)
    return c.fetchall()


# Define the respond function
def respond(message, params, neg_params):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    ent_vals = [e["value"] for e in entities]
    # Look for negated entities
    negated = negated_ents(message, ent_vals)
    for ent in entities:
        if ent["value"] in negated and negated[ent["value"]]:
            neg_params[ent["entity"]] = str(ent["value"])
        else:
            params[ent["entity"]] = str(ent["value"])
    # Find the hotels
    results = find_hotels(params, neg_params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Return the correct response
    return responses_3[n].format(*names), params, neg_params


# Initialize params and neg_params
params = {}
neg_params = {}

# Pass the messages to the bot
for message in ["I want a cheap hotel", "but not in the north of town"]:
    print("USER: {}".format(message))
    response, params, neg_params = respond(message, params, neg_params)
    print("BOT: {}".format(response))
