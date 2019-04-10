################################################################################
################################ CHATBOT #######################################
################################################################################

# ECHOBOT - Simple bot that just replies with the same message it receives

bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define a function that responds to a user's message: respond


def respond(message):
    # Concatenate the user's message to the end of a standard bot respone
    bot_message = "I can hear you! You said: " + message
    # Return the result
    return bot_message


# Test function
print(respond("hello!"))


# Define a function that sends a message to the bot: send_message
def send_message(message):
    # Print user_template including the user_message
    print(user_template.format(message))
    # Get the bot's response to the message
    response = respond(message)
    # Print the bot template including the bot's response.
    print(bot_template.format(response))


# Send a message to the bot
send_message("hello")


# Chitchat - using a dictionary to answer questions
# This means the bot will only respond correctly if the message matches EXACTLY,
# which is a big limitation. In later exercises you will create much more robust solutions.

# Define variables
name = "Greg"
weather = "cloudy"

# Define a dictionary with the predefined responses
responses = {
    "what's your name?": "my name is {0}".format(name),
    "what's today's weather?": "the weather is {0}".format(weather),
    "default": "default message"
}

# Return the matching response if there is one, default otherwise


def respond(message):
    # Check if the message is in the responses
    if message in responses:
        # Return the matching message
        bot_message = responses[message]
    else:
        # Return the "default" message
        bot_message = responses["default"]
    return bot_message


# It can get a little boring hearing the same old answers over and over.
# In this exercise, you'll add some variation.
# If you ask your bot how it's feeling, the likelihood that it
# responds with "oh I'm great!" or "I'm very sad today" should be equal.

# Here, you'll use the random module - specifically random.choice(ls) -
# which randomly selects an element from a list ls.

# Import the random module
import random

name = "Greg"
weather = "cloudy"

# Define a dictionary containing a list of responses for each message
responses = {
    "what's your name?": [
        "my name is {0}".format(name),
        "they call me {0}".format(name),
        "I go by {0}".format(name)
    ],
    "what's today's weather?": [
        "the weather is {0}".format(weather),
        "it's {0} today".format(weather)
    ],
    "default": ["default message"]
}

# Use random.choice() to choose a matching response


def respond(message):
    if message in responses:
        bot_message = random.choice(responses[message])
    else:
        bot_message = random.choice(responses["default"])
    return bot_message


################################################################################
################################ ELIZA #########################################
################################################################################
# ELIZA 1: Asking Questions
# Responding to statements and questions in a more engaged way
import random

responses_1 = {'question': ["I don't know :(", 'you tell me!'],
               'statement': ['tell me more!',
                             'why do you think that?',
                             'how long have you felt this way?',
                             'I find that extremely interesting',
                             'can you back that up?',
                             'oh wow!',
                             ':)']}


def respond(message):
    # Check for a question mark
    if message.endswith('?'):
        # Return a random question
        return random.choice(responses_1["question"])
    # Return a random statement
    return random.choice(responses_1["statement"])


# Send messages ending in a question mark
send_message("what's today's weather?")
send_message("what's today's weather?")

# Send messages which don't end with a question mark
send_message("I love building chatbots")
send_message("I love building chatbots")


# ELIZA II: Extracting key phrases
# The really clever thing about ELIZA is the way the program appears to understand
# what you told it by occasionally including phrases uttered by the user in its responses.

# In this exercise, you will match messages against some common patterns and extract
# phrases using re.search()
import re


rules = {'I want (.*)': ['What would it mean if you got {0}',
                         'Why do you want {0}',
                         "What's stopping you from getting {0}"],
         'do you remember (.*)': ['Did you think I would forget {0}',
                                  "Why haven't you been able to forget {0}",
                                  'What about {0}',
                                  'Yes .. and?'],
         'do you think (.*)': ['if {0}? Absolutely.', 'No chance'],
         'if (.*)': ["Do you really think it's likely that {0}",
                     'Do you wish that {0}',
                     'What do you think about {0}',
                     'Really--if {0}']}

# Define match_rule()


def match_rule(rules, message):
    response, phrase = "default", None

    # Iterate over the rules dictionary
    for pattern, responses in rules.items():
        # Create a match object
        match = re.search(pattern, message)
        if match is not None:
            # Choose a random response
            response = random.choice(responses)
            if '{0}' in response:
                phrase = match.group(1)
    # Return the response and phrase
    return response.format(phrase)


# Test match_rule
print(match_rule(rules, "do you remember your last birthday"))


# ELIZA III: Pronouns
# To make responses grammatically coherent, you'll want to transform the extracted
# phrases from first to second person and vice versa. Works in most cases.

# Define a function called replace_pronouns() which uses re.sub() to map
# "me" and "my" to "you" and "your" (and vice versa) in a string.

# Define replace_pronouns()
def replace_pronouns(message):

    message = message.lower()
    if 'me' in message:
        # Replace 'me' with 'you'
        return re.sub('me', 'you', message)
    if 'my' in message:
        # Replace 'my' with 'your'
        return re.sub('my', 'your', message)
    if 'your' in message:
        # Replace 'your' with 'my'
        return re.sub('your', 'my', message)
    if 'you' in message:
        # Replace 'you' with 'me'
        return re.sub('you', 'me', message)

    return message


print(replace_pronouns("my last birthday"))
print(replace_pronouns("when you went to Florida"))
print(replace_pronouns("I had my own castle"))


# ELIZA IV: Putting it all together
# Now you're going to put everything from the previous exercises together!
# The match_rule(), send_message(), and replace_pronouns() functions have already
# been defined, and the rules dictionary is available in your workspace.

# Your job here is to write a function called respond() with a single argument
# message which creates an appropriate response to be handled by send_message().

# Define respond()
def respond(message):
    # Call match_rule
    response, phrase = match_rule(rules, message)
    if '{0}' in response:
        # Replace the pronouns in the phrase
        phrase = replace_pronouns(phrase)
        # Include the phrase in the response
        response = response.format(phrase)
    return response


# Send the messages
send_message("do you remember your last birthday")
send_message("do you think humans should be worried about AI")
send_message("I want a robot friend")
send_message("what if you could be anything you wanted")


################################################################################
################### UNDERSTANDING INTENTS & ENTITIES ###########################
################################################################################

# Intent classification with regex I
# Look for presence of keywords using 'keywords' dictionary
# 'responses' dictionary indicates how bot should respond to each intent in keywords dict
# Also uses the send_message() function

keywords = {'goodbye': ['bye', 'farewell'],
            'greet': ['hello', 'hi', 'hey'],
            'thankyou': ['thank', 'thx']}

responses_2 = {'default': 'default message',
               'goodbye': 'goodbye for now',
               'greet': 'Hello you! :)',
               'thankyou': 'you are very welcome'}

# Define a dictionary of patterns
patterns = {}

# Iterate over the keywords dictionary
for intent, keys in keywords.items():
    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile('|'.join(keys))

# Print the patterns
print(patterns)


# Intent classification with regex II
# Define function to find intent of a message
# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None
    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message
        if pattern.search(message):
            matched_intent = intent
    return matched_intent

# Define a respond function


def respond(message):
    # Call the match_intent function
    intent = match_intent(message)
    # Fall back to the default response
    key = "default"
    if intent in responses:
        key = intent
    return responses[key]


# Send messages
send_message("hello!")
send_message("bye byeee")
send_message("thanks very much!")


# Entity extraction with regex
# Finding a name in a sentence - look for keywords 'name' or 'call(ed)', find
# capitalized words w/ regex and assume those are names

# Define find_name()
def find_name(message):
    name = None
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile('name|call')
    # Create a pattern for finding capitalized words
    name_pattern = re.compile('[A-Z]{1}[a-z]*')
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
        if len(name_words) > 0:
            # Return the name if the keywords are present
            name = ' '.join(name_words)
    return name

# Define respond()


def respond(message):
    # Find the name
    name = find_name(message)
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)


# Send messages
send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("People call me Cassandra")

# Assumptions are made - ex: bot thinks "People" is an entity
