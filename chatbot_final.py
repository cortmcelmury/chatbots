############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################

# Putting it all together I
# It's time to put everything you've learned in the course together by combining
# the coffee ordering bot with the ELIZA rules from chapter 1.

# To begin, you'll define a function called chitchat_response(), which calls the
# predefined function match_rule() from back in chapter 1. This returns a response
# if the message matched an ELIZA template, and otherwise, None.

# The ELIZA rules are contained in a dictionary called eliza_rules.

import re
import random
import string

INIT = 0
AUTHED = 1
CHOOSE_COFFEE = 2
ORDERED = 3

eliza_rules = {'I want (.*)': ['What would it mean if you got {0}',
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

policy_rules = {(0, 'number'): (1, 'perfect, welcome back!', None),
                (0, 'order'): (0,
                               "you'll have to log in first, what's your phone number?",
                               1),
                (1, 'order'): (2, 'would you like Columbian or Kenyan?', None),
                (2, 'specify_coffee'): (3, 'perfect, the beans are on their way!', None)}


def interpret(message):
    msg = message.lower()
    if 'order' in msg:
        return 'order'
    if 'kenyan' in msg or 'columbian' in msg:
        return 'specify_coffee'
    if any([d in msg for d in string.digits]):
        return 'number'
    return 'none'


def match_rule(rules, message):
    for pattern, responses in rules.items():
        match = re.search(pattern, message)
        if match is not None:
            response = random.choice(responses)
            var = match.group(1) if '{0}' in response else None
            return response, var
    return "default", None


def replace_pronouns(message):
    message = message.lower()
    if 'me' in message:
        return re.sub('me', 'you', message)
    if 'i' in message:
        return re.sub('i', 'you', message)
    elif 'my' in message:
        return re.sub('my', 'your', message)
    elif 'your' in message:
        return re.sub('your', 'my', message)
    elif 'you' in message:
        return re.sub('you', 'me', message)
    return message


# Define chitchat_response()


def chitchat_response(message):
    # Call match_rule()
    response, phrase = match_rule(eliza_rules, message)
    # Return none if response is "default"
    if response == "default":
        return None
    if '{0}' in response:
        # Replace the pronouns of phrase
        phrase = replace_pronouns(phrase)
        # Calculate the response
        response = response.format(phrase)
    return response

# Define send_message()


def send_message(state, pending, message):
    print("USER : {}".format(message))
    response = chitchat_response(message)
    if response is not None:
        print("BOT : {}".format(response))
        return state, None

    # Calculate the new_state, response, and pending_state
    new_state, response, pending_state = policy_rules[(
        state, interpret(message))]
    print("BOT : {}".format(response))
    if pending is not None:
        new_state, response, pending_state = policy_rules[pending]
        print("BOT : {}".format(response))
    if pending_state is not None:
        pending = (pending_state, interpret(message))
    return new_state, pending

# Define send_messages()


def send_messages(messages):
    state = INIT
    pending = None
    for msg in messages:
        state, pending = send_message(state, pending, msg)


# Send the messages
send_messages([
    "I'd like to order some coffee",
    "555-12345",
    "do you remember when I ordered 1000 kilos by accident?",
    "kenyan"
])
