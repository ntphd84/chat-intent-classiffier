import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
import os
import glob
import json
from collections import Counter
import hashlib  # Added for unique hash generation

# Load models
nlp = spacy.load('en_core_web_sm')
sia = SentimentIntensityAnalyzer()

# Path to existing master file
master_file_path = 'results/master_chat_class.xlsx'

# Path to store intent patterns
intent_patterns_file = 'results/chat_intent_patterns.json'

# Default intent patterns dictionary
default_intent_patterns = {
    'Wayfinding query': [r'\b(where|location|which|floor)\b'],
    'Concert/event schedule query': [r'\b(when|time|concert|show|open)\b'],
    'Cancellation/Refund request': [r'\b(refund|cancel|cancellation)\b'],
    'Booking request': [r'\b(booking|room reservation|book)\b'],
    'Free room requests': [r'\b(free room|complimentary|comp room|birthday month|)\b'],
    'Account update/access issues': [r'\b(account|access|login|activate|card|account update|update info)\b'],
    'Membership entitlements & other issues': [r'\b(membership|vip|gold|genting rewards card|e-voucher|subsidy)\b'],
    'Password/OTP issues': [r'\b(password|otp|one time pin|forgot password|password reset)\b'],
    'Theme Park query': [r'\b(SkyWorlds|theme park|snow world|outdoor theme park|skytropolis|indoor themepark)\b'],
    'Transportation/Cable Car query': [r'\b(cable car|skyway|transpo|limo|taxi|bus|terminal)\b'],
    'hotel amenities/Check-in query': [r'\b(first world|hotel|genting skyworlds hotel|resorts hotel|maxim|crockfords|checkin)\b'],
    'Others/unclassified/Incomplete query': []
}

# Load existing intent patterns if available, otherwise use defaults
if os.path.exists(intent_patterns_file):
    with open(intent_patterns_file, 'r') as f:
        intent_patterns = json.load(f)
    print(f"Loaded {len(intent_patterns)} intent patterns from existing file")
else:
    intent_patterns = default_intent_patterns
    print(f"Using {len(intent_patterns)} default intent patterns")

# Function to generate a unique hash for a conversation
def generate_row_hash(conversation_id, date_part, chat):
    """
    Generate a unique MD5 hash for a conversation based on its ID, date, and chat text.
    """
    combined_string = f"{conversation_id}|{date_part}|{chat}"
    return hashlib.md5(combined_string.encode('utf-8')).hexdigest()

# Function to detect potential new intents
def detect_new_intents(message, existing_intents):
    # Extract potential intent-indicating phrases
    doc = nlp(message.lower())
    
    # Look for question patterns (starting with what, how, can, etc.)
    question_words = ['what', 'how', 'can', 'could', 'would', 'will', 'is', 'are', 'do', 'does']
    potential_intents = []
    
    # Check for question patterns
    if any(message.lower().startswith(word) for word in question_words):
        # Get verb and object to form intent name
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        objects = [chunk.text for chunk in doc.noun_chunks]
        
        if verbs and objects:
            potential_intent = f"{verbs[0]} {objects[0]}"
            potential_intents.append((potential_intent, [verbs[0]]))
    
    # Look for common customer service intents using entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    for ent_text, ent_label in entities:
        if ent_label in ['ORG', 'PRODUCT']:
            potential_intent = f"{ent_text} inquiry"
            potential_intents.append((potential_intent, [ent_text.lower()]))
    
    # Check for action verbs followed by direct objects
    action_verbs = ['book', 'reserve', 'buy', 'booking', 'need', 'want', 'request', 'check', 'change', 'update', 'pay', 'help']
    for verb in action_verbs:
        if verb in [token.lemma_ for token in doc]:
            verb_token = next(token for token in doc if token.lemma_ == verb)
            children = list(verb_token.children)
            objects = [child.text for child in children if child.dep_ in ['dobj', 'pobj']]
            if objects:
                potential_intent = f"{verb} {objects[0]}"
                potential_intents.append((potential_intent, [verb, objects[0]]))
    
    return potential_intents

# Function to update intent patterns with new patterns
def update_intent_patterns(new_patterns):
    global intent_patterns
    updated = False
    
    for intent_name, patterns in new_patterns.items():
        if intent_name not in intent_patterns:
            intent_patterns[intent_name] = patterns
            print(f"  Added new intent: {intent_name} with patterns: {patterns}")
            updated = True
        else:
            # Add any new patterns that don't exist
            for pattern in patterns:
                if pattern not in intent_patterns[intent_name]:
                    intent_patterns[intent_name].append(pattern)
                    print(f"  Added new pattern '{pattern}' to intent: {intent_name}")
                    updated = True
    
    if updated:
        # Save updated patterns
        with open(intent_patterns_file, 'w') as f:
            json.dump(intent_patterns, f, indent=2)
        print("  Intent patterns updated and saved")
    
    return updated

# Load existing master file if it exists
if os.path.exists(master_file_path):
    existing_master_df = pd.read_excel(master_file_path)
    print(f"Loaded existing master file with {len(existing_master_df)} records")
else:
    existing_master_df = pd.DataFrame()
    print("No existing master file found. Creating a new one.")

# Get all Excel files in the directory
files_path = 'datasets/2024_chatLog/rawchats'
all_files = glob.glob(os.path.join(files_path, '*.xlsx'))

print(f"Found {len(all_files)} Excel files to process")

# Create a list to store data for the master file
all_new_data = []

# Process each file
for file_path in all_files:
    # Get the filename
    filename = os.path.basename(file_path)
    print(f"\nProcessing file: {filename}")
    
    # Extract date from filename (text after '-' and remove '.xlsx')
    if '-' in filename:
        date_part = filename.split('-', 1)[1].replace('.xlsx', '')
    else:
        date_part = filename.replace('.xlsx', '')
    
    # Read the Excel file
    try:
        df = pd.read_excel(file_path)
        print(f"  File contains {len(df)} conversations")
    except Exception as e:
        print(f"  Error reading file {filename}: {e}")
        continue
    
    # Analyze each conversation
    file_data = []
    for i, row in df.iterrows():
        # Get the conversation ID and chat text
        conversation_id = row['ConversationId']
        chat = row['Chats']
        
        # Generate unique hash for the current conversation
        current_hash = generate_row_hash(conversation_id, date_part, chat)
        
        # Check if this conversation was already processed based on the unique hash.
        if not existing_master_df.empty:
            if 'unique_hash' in existing_master_df.columns:
                if current_hash in existing_master_df['unique_hash'].values:
                    print(f"  Skipping conversation {conversation_id} - already processed (hash match)")
                    continue
            else:
                # Fallback check if 'unique_hash' column doesn't exist:
                if ((existing_master_df['date'] == date_part) & (existing_master_df['ConversationId'] == conversation_id)).any():
                    print(f"  Skipping conversation {conversation_id} - already in master file")
                    continue
        
        # Split chat into lines
        lines = chat.split('\n')
        
        # Extract user messages properly
        user_messages = []
        j = 0
        while j < len(lines):
            if lines[j].startswith('user('):
                # The user message is in the next line
                if j+1 < len(lines):
                    user_messages.append(lines[j+1])
                j += 3  # Skip timestamp, message, and empty line
            else:
                j += 1
        
        # Extract bot messages properly
        bot_messages = []
        j = 0
        while j < len(lines):
            if lines[j].startswith('bot('):
                # The bot message can span multiple lines until an empty line
                bot_msg = []
                j += 1  # Move to the first line of the message
                while j < len(lines) and lines[j] != '':
                    bot_msg.append(lines[j])
                    j += 1
                if bot_msg:
                    bot_messages.append('\n'.join(bot_msg))
                if j < len(lines) and lines[j] == '':
                    j += 1  # Skip the empty line
            else:
                j += 1
        
        # Join all user messages for analysis
        all_user_text = ' '.join(user_messages)
        
        # Learn new intents from user messages
        new_intent_patterns = {}
        for msg in user_messages:
            potential_intents = detect_new_intents(msg, intent_patterns)
            for intent_name, patterns in potential_intents:
                if intent_name not in new_intent_patterns:
                    new_intent_patterns[intent_name] = patterns
                else:
                    new_intent_patterns[intent_name].extend(patterns)
        
        # Update intent patterns if we found any new ones
        if new_intent_patterns:
            update_intent_patterns(new_intent_patterns)
        
        # Sentiment Analysis
        sentiment = sia.polarity_scores(all_user_text)
        sentiment_score = sentiment['compound']
        sentiment_category = 'Positive' if sentiment_score > 0.05 else 'Negative' if sentiment_score < -0.05 else 'Neutral'
        
        # Advanced intent classification (using updated patterns)
        intents = []
        
        # Go through each intent and check if any of its patterns match
        for intent_name, patterns in intent_patterns.items():
            if not patterns:  # Skip empty patterns (like General information request)
                continue
                
            for pattern in patterns:
                try:
                    safe_pattern = re.escape(pattern)
                    if any(re.search(safe_pattern, msg.lower()) for msg in user_messages):
                        # Insert your logic here if a pattern matches
                        intents.append(intent_name)
                        break  # Stop checking further patterns for this intent
                except re.error as e:
                    print(f"Invalid regex pattern '{pattern}': {e}")
                    intents.append(intent_name)
                    break  # No need to check other patterns for this intent
        
        # If no specific intent was found, mark as general information
        if not intents:
            intents.append('General information request')
        
        # Join all intents with a semicolon for the master file
        intent_str = '; '.join(intents)
        
        # Create a row for the master file, including the unique hash
        master_row = row.to_dict()
        master_row['date'] = date_part
        master_row['intent'] = intent_str
        master_row['sentiment_score'] = sentiment_score
        master_row['sentiment_cat'] = sentiment_category
        master_row['unique_hash'] = current_hash  # Add the unique hash to the row
        
        file_data.append(master_row)
    
    print(f"  Added {len(file_data)} new conversations from {filename}")
    all_new_data.extend(file_data)

# Create DataFrame from the new data
new_data_df = pd.DataFrame(all_new_data)

# Combine with existing master data
if existing_master_df.empty:
    updated_master_df = new_data_df
else:
    updated_master_df = pd.concat([existing_master_df, new_data_df], ignore_index=True)

# Save to Excel
updated_master_df.to_excel(master_file_path, index=False)

print(f"\nMaster file updated successfully: {master_file_path}")
print(f"Total records in master file: {len(updated_master_df)}")
print(f"\nFinal intent patterns:")
for intent, patterns in intent_patterns.items():
    print(f"  {intent}: {patterns}")

print(f"\nPreview of the updated master file:")
print(updated_master_df[['ConversationId', 'date', 'intent', 'sentiment_score', 'sentiment_cat']].head())
