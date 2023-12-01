import json
import datetime
import time
import os
import dateutil.parser
import logging
import boto3
from boto3.dynamodb.conditions import Key
from langchain.llms.bedrock import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage
from chat import Chat
from fsi_agent import FSIAgent
from pypdf import PdfReader, PdfWriter

# Create reference to DynamoDB tables
loan_application_table_name = os.environ['USER_PENDING_ACCOUNTS_TABLE']
user_accounts_table_name = os.environ['USER_EXISTING_ACCOUNTS_TABLE']
s3_artifact_bucket = os.environ['S3_ARTIFACT_BUCKET_NAME']

# Instantiate boto3 clients and resources
boto3_session = boto3.Session(region_name=os.environ['AWS_REGION'])
dynamodb = boto3.resource('dynamodb',region_name=os.environ['AWS_REGION'])
s3_client = boto3.client('s3',region_name=os.environ['AWS_REGION'],config=boto3.session.Config(signature_version='s3v4',))
s3_object = boto3.resource('s3')
bedrock_client = boto3_session.client(service_name="bedrock-runtime")

# --- Lex v2 request/response helpers (https://docs.aws.amazon.com/lexv2/latest/dg/lambda-response-format.html) ---

def elicit_slot(session_attributes, active_contexts, intent, slot_to_elicit, message):
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'ElicitSlot',
                'slotToElicit': slot_to_elicit
            },
            'intent': intent,
        },
        'messages': [{
            "contentType": "PlainText",
            "content": message,
        }]
    }

    return response

def confirm_intent(active_contexts, session_attributes, intent, message):
    response = {
        'sessionState': {
            'activeContexts': [active_contexts],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'ConfirmIntent'
            },
            'intent': intent
        }
    }

    return response

def close(session_attributes, active_contexts, fulfillment_state, intent, message):
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close',
            },
            'intent': intent,
        },
        'messages': [{'contentType': 'PlainText', 'content': message}]
    }

    return response

def elicit_intent(intent_request, session_attributes, message):
    response = {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [
            {
                'contentType': 'PlainText', 
                'content': message
            },
            {
                'contentType': 'ImageResponseCard',
                'imageResponseCard': {
                    "buttons": [
                        {
                            "text": "Loan Application",
                            "value": "Loan Application"
                        },
                        {
                            "text": "Loan Calculator",
                            "value": "Loan Calculator"
                        },
                        {
                            "text": "Ask GenAI",
                            "value": "What kind of questions can the Assistant answer?"
                        }
                    ],
                    "title": "How can I help you?"
                }
            }     
        ]
    }

    return response

def delegate(session_attributes, active_contexts, intent, message):
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Delegate',
            },
            'intent': intent,
        },
        'messages': [{'contentType': 'PlainText', 'content': message}]
    }

    return response

def initial_message(intent_name):
    response = {
            'sessionState': {
                'dialogAction': {
                    'type': 'ElicitSlot',
                    'slotToElicit': 'UserName' if intent_name=='MakePayment' else 'PickUpCity'
                },
                'intent': {
                    'confirmationState': 'None',
                    'name': intent_name,
                    'state': 'InProgress'
                }
            }
    }
    
    return response

def build_response_card(title, subtitle, options):
    """
    Build a responseCard with a title, subtitle, and an optional set of options which should be displayed as buttons.
    """
    buttons = None
    if options is not None:
        buttons = []
        for i in range(min(5, len(options))):
            buttons.append(options[i])

    return {
        'contentType': 'ImageResponseCard',
        'imageResponseCard': {
            'title': title,
            'subTitle': subtitle,
            'buttons': buttons
        }
    }

def build_slot(intent_request, slot_to_build, slot_value):
    intent_request['sessionState']['intent']['slots'][slot_to_build] = {
        'shape': 'Scalar', 'value': 
        {
            'originalValue': slot_value, 'resolvedValues': [slot_value], 
            'interpretedValue': slot_value
        }
    }

def build_validation_result(isvalid, violated_slot, message_content):
    print("Build Validation")
    return {
        'isValid': isvalid,
        'violatedSlot': violated_slot,
        'message': message_content
    }
    
# --- Utility helper functions ---

def isvalid_date(date):
    try:
        dateutil.parser.parse(date, fuzzy=True)
        print("TRUE DATE")
        return True
    except ValueError as e:
        print("DATE PARSER ERROR = " + str(e))
        return False

def isvalid_yes_or_no(value):
    if value == 'Yes' or value == 'yes' or value == 'No' or value == 'no':
        return True
    return False

def isvalid_credit_score(credit_score):
    if int(credit_score) < 851 and int(credit_score) > 300:
        return True
    return False

def isvalid_zero_or_greater(value):
    if int(value) >= 0:
        return True
    return False

def safe_int(n):
    if n is not None:
        return int(n)
    return n

def create_presigned_url(bucket_name, object_name, expiration=600):
    # Generate a presigned URL for the S3 object
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(e)
        logging.error(e)
        return "Error"

    # The response contains the presigned URL
    return response

def try_ex(value):
    """
    Safely access Slots dictionary values.
    """
    if value is not None:
        if value['value']['resolvedValues']:
            return value['value']['interpretedValue']
        elif value['value']['originalValue']:
            return value['value']['originalValue']
        else:
            return None
    else:
        return None

# --- Intent fulfillment functions --- 

def isvalid_pin(userName, pin):
    """
    Validates the user-provided PIN using a DynamoDB table lookup.
    """
    plans_table = dynamodb.Table(user_accounts_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': 'userName = :c',
            'ExpressionAttributeValues': {
                ':c': userName
            }
        }

        # Execute the query and get the result
        response = plans_table.query(**params)

        # iterate over the items returned in the response
        if len(response['Items']) > 0:
            pin_to_compare = int(response['Items'][0]['pin'])
            # check if the password in the item matches the specified password
            if pin_to_compare == int(pin):
                return True

        return False

    except Exception as e:
        print(e)
        return e

def isvalid_username(userName):
    """
    Validates the user-provided username exists in the 'user_accounts_table_name' DynamoDB table.
    """
    plans_table = dynamodb.Table(user_accounts_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': 'userName = :c',
            'ExpressionAttributeValues': {
                ':c': userName
            }
        }

        # Execute the query and get the result
        response = plans_table.query(**params)

        # Check if any items were returned
        if response['Count'] != 0:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return e

def validate_pin(intent_request, slots):
    """
    Performs slot validation for username and PIN. Invoked as part of 'verify_identity' intent fulfillment.
    """
    username = try_ex(slots['UserName'])
    pin = try_ex(slots['Pin'])

    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
        session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
        session_attributes['UserName'] = username
        intent_request['sessionState']['sessionAttributes']['UserName'] = username

    else:
        return build_validation_result(
            False,
            'UserName',
            'Our records indicate there are no accounts belonging to that username. Please try again.'
        )

    if pin is not None:
        if  not isvalid_pin(username, pin):
            return build_validation_result(
                False,
                'Pin',
                'You have entered an incorrect PIN. Please try again.'.format(pin)
            )
    else:
        message = "Thank you for choosing Octank Financial, {}. Please confirm your 4-digit PIN before we proceed.".format(username)
        return build_validation_result(
            False,
            'Pin',
            message
        )

    return {'isValid': True}

def verify_identity(intent_request):
    """
    Performs dialog management and fulfillment for username verification.
    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting.
    2) Use of sessionAttributes {UserName} to pass information that can be used to guide conversation.
    """
    slots = intent_request['sessionState']['intent']['slots']
    pin = try_ex(slots['Pin'])
    username=try_ex(slots['UserName'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    # Validate any slots which have been specified.  If any are invalid, re-elicit for their value
    validation_result = validate_pin(intent_request, intent_request['sessionState']['intent']['slots'])
    session_attributes['UserName'] = username

    if not validation_result['isValid']:
        slots = intent_request['sessionState']['intent']['slots']
        slots[validation_result['violatedSlot']] = None

        return elicit_slot(
            session_attributes,
            active_contexts,
            intent_request['sessionState']['intent'],
            validation_result['violatedSlot'],
            validation_result['message']
        )
    else:
        if confirmation_status == 'None':
            # Query DDB for user information before offering intents
            plans_table = dynamodb.Table(user_accounts_table_name)

            try:
                # Query the table using the partition key
                response = plans_table.query(
                    KeyConditionExpression=Key('userName').eq(username)
                )

                # TODO: Customize account readout based on account type
                message = ""
                items = response['Items']
                for item in items:
                    if item['planName'] == 'mortgage' or item['planName'] == 'Mortgage':
                        message = "Your mortgage account summary includes a ${:,} loan at {}% interest with ${:,} of unpaid principal. Your next payment of ${:,} is scheduled for {}.".format(item['loanAmount'], item['loanInterest'], item['unpaidPrincipal'], item['amountDue'], item['dueDate'])
                    elif item['planName'] == 'Checking' or item['planName'] == 'checking':
                        message = "I see you have a Savings account with Octank Financial. Your account balance is ${:,} and your next payment \
                            amount of ${:,} is scheduled for {}.".format(item['unpaidPrincipal'], item['paymentAmount'], item['dueDate'])
                    elif item['planName'] == 'Loan' or item['planName'] == 'loan':
                            message = "I see you have a Loan account with Octank Financial. Your account balance is ${:,} and your next payment \
                            amount of ${:,} is scheduled for {}.".format(item['unpaidPrincipal'], item['paymentAmount'], item['dueDate'])
                return elicit_intent(intent_request, session_attributes, 
                    'Thank you for confirming your username and PIN, {}. {}'.format(username, message)
                    )

            except Exception as e:
                print(e)
                return e

def invoke_fm(prompt):
    """
    Invokes Foundational Model endpoint hosted on Amazon Bedrock and parses the response.
    """
    chat = Chat(prompt)
    llm = Bedrock(client=bedrock_client, model_id="anthropic.claude-v2", region_name=os.environ['AWS_REGION']) # "anthropic.claude-v2 "
    llm.model_kwargs = {'max_tokens_to_sample': 350}
    lex_agent = FSIAgent(llm, chat.memory)
    formatted_prompt = "\n\nHuman: " + prompt + " \n\nAssistant:"

    try:
        message = lex_agent.run(input=formatted_prompt)
    except ValueError as e:
        message = str(e)
        if not message.startswith("Could not parse LLM output:"):
            raise e
        message = message.removeprefix("Could not parse LLM output: `").removesuffix("`")

    return message

def genai_intent(intent_request):
    """
    Performs dialog management and fulfillment for user utterances that do not match defined intents (i.e., FallbackIntent).
    Sends user utterance to Foundational Model endpoint via 'invoke_fm' function.
    """
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    
    if intent_request['invocationSource'] == 'DialogCodeHook':
        prompt = intent_request['inputTranscript']
        output = invoke_fm(prompt)
        return elicit_intent(intent_request, session_attributes, output)

# --- Intents ---

def dispatch(intent_request):
    """
    Routes the incoming request based on intent.
    """
    slots = intent_request['sessionState']['intent']['slots']
    username = slots['UserName'] if 'UserName' in slots else None
    intent_name = intent_request['sessionState']['intent']['name']

    if intent_name == 'VerifyIdentity':
        return verify_identity(intent_request)
    elif intent_name == 'LoanApplication':
        return loan_application(intent_request)
    elif intent_name == 'LoanCalculator':
        return loan_calculator(intent_request)
    else:
        return genai_intent(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')
        
# --- Main handler ---

def handler(event, context):
    """
    Invoked when the user provides an utterance that maps to a Lex bot intent.
    The JSON body of the user request is provided in the event slot.
    """
    os.environ['TZ'] = 'America/New_York'
    time.tzset()

    return dispatch(event)
