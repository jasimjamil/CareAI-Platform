import os
import json
import uuid
import sqlite3
import datetime
import requests
import asyncio
from typing import List, Dict, Optional, Any
import hashlib
import secrets
import time
from datetime import datetime, timedelta

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Streamlit import
import streamlit as st
import pandas as pd

# Telegram Bot integration - FIXED IMPORT
import telegram
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Google API for AI integration
import google.generativeai as genai

# Database connection
import sqlite3
from contextlib import contextmanager

# Make Flask import optional
FLASK_AVAILABLE = False
try:
    from flask import Flask, request
    import twilio.twiml.messaging_response
    FLASK_AVAILABLE = True
    print("Flask is available, WhatsApp webhook functionality enabled")
except ImportError:
    print("Flask not installed. WhatsApp webhook functionality disabled.")
    print("To enable WhatsApp integration, install Flask with: pip install flask twilio")

# Set up database
DB_NAME = os.path.join(os.path.dirname(__file__), "agentic_cs.db")

# Initialize databases
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Agents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        prompt_template TEXT NOT NULL,
        knowledge_base TEXT NOT NULL,
        auto_followup TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Moderation rules table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS moderation_rules (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        rule_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (agent_id) REFERENCES agents (id)
    )
    ''')
    
    # Chat history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        channel TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        was_moderated BOOLEAN DEFAULT FALSE,
        was_escalated BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (agent_id) REFERENCES agents (id)
    )
    ''')
    
    # Fallback configurations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fallback_config (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        condition_text TEXT NOT NULL,
        external_service_url TEXT NOT NULL,
        service_api_key TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (agent_id) REFERENCES agents (id)
    )
    ''')
    
    # Add users table for authentication
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        is_admin BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # Add sessions table for managing login sessions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create default agents if none exist
    cursor.execute("SELECT COUNT(*) FROM agents")
    agent_count = cursor.fetchone()[0]
    
    if agent_count == 0:
        # Create default pharmacy agent
        pharmacy_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO agents (id, name, type, prompt_template, knowledge_base, auto_followup) VALUES (?, ?, ?, ?, ?, ?)",
            (
                pharmacy_id, 
                "Pharmacy Agent", 
                "pharmacy", 
                "You are a pharmacy assistant. Help with medication questions.",
                "Common medications: Paracetamol, Ibuprofen, Aspirin\nSide effects: Headache, nausea, dizziness", 
                "Is there anything else you'd like to know about medications?"
            )
        )
        
        # Create default infusion agent
        infusion_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO agents (id, name, type, prompt_template, knowledge_base, auto_followup) VALUES (?, ?, ?, ?, ?, ?)",
            (
                infusion_id, 
                "Infusion Agent", 
                "infusion", 
                "You are an infusion therapy assistant. Help with infusion-related questions.",
                "Common infusions: IV fluids, antibiotics, chemotherapy\nSide effects: Pain at site, infection risk", 
                "Is there anything else you'd like to know about infusion therapy?"
            )
        )
        
        # Add some basic moderation rules
        rule_id1 = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO moderation_rules (id, agent_id, rule_text) VALUES (?, ?, ?)",
            (rule_id1, pharmacy_id, "Do not provide specific dosage recommendations")
        )
        
        rule_id2 = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO moderation_rules (id, agent_id, rule_text) VALUES (?, ?, ?)",
            (rule_id2, infusion_id, "Do not provide medical diagnoses")
        )
        
        # Add fallback configuration
        fallback_id1 = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO fallback_config (id, agent_id, condition_text, external_service_url, service_api_key) VALUES (?, ?, ?, ?, ?)",
            (fallback_id1, pharmacy_id, "speak to a human", "https://example.com/fallback", "api-key-123")
        )
        
        fallback_id2 = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO fallback_config (id, agent_id, condition_text, external_service_url, service_api_key) VALUES (?, ?, ?, ?, ?)",
            (fallback_id2, infusion_id, "emergency", "https://example.com/urgent", "api-key-456")
        )
    
    # Add a default admin user if none exists
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    
    if user_count == 0:
        # Create admin user
        admin_id = str(uuid.uuid4())
        admin_username = "admin"
        admin_email = "admin@example.com"
        admin_password = "admin123"  # This should be changed after first login
        
        # Generate salt and hash password
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((admin_password + salt).encode()).hexdigest()
        
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash, salt, is_admin) VALUES (?, ?, ?, ?, ?, ?)",
            (admin_id, admin_username, admin_email, password_hash, salt, True)
        )
        print(f"Created default admin user: {admin_username} / {admin_password}")
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Initialize FastAPI app
app = FastAPI(title="Agentic CS Application")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests and responses
class Agent(BaseModel):
    id: Optional[str] = None
    name: str
    type: str
    prompt_template: str
    knowledge_base: str
    auto_followup: Optional[str] = None

class ModerationRule(BaseModel):
    id: Optional[str] = None
    agent_id: str
    rule_text: str

class FallbackConfig(BaseModel):
    id: Optional[str] = None
    agent_id: str
    condition_text: str
    external_service_url: str
    service_api_key: Optional[str] = None

class ChatMessage(BaseModel):
    user_id: str
    agent_id: str
    message: str
    channel: str

class ChatResponse(BaseModel):
    id: str
    response: str
    was_moderated: bool
    was_escalated: bool

# Google API helper for AI - FIXED PROPERLY
class GoogleAI:
    def __init__(self):
        # Load Gemini API key from environment variable or use default
        self.api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAnqTGGdlT1F4EY5mzWomNxNpkJuHLiFEs")
        self.api_working = False
        
        try:
            # Import and check for the library
            try:
                import google.generativeai as genai
                self.genai = genai
                print("Google Generative AI library imported successfully")
                
                # Configure with API key
                if hasattr(genai, 'configure'):
                    genai.configure(api_key=self.api_key)
                
                # Check which version we're working with
                if hasattr(genai, 'GenerativeModel'):
                    self.api_version = "new"
                    print("Using newer Google AI API version")
                else:
                    self.api_version = "old"
                    print("Using older Google AI API version")
                
                # Don't test the API here - it might fail
                self.api_working = True
                
            except ImportError:
                print("Google Generative AI library not installed")
                self.api_working = False
                
        except Exception as e:
            print(f"Error initializing Google AI: {e}")
            self.api_working = False
    
    def generate_response(self, agent_type, prompt, knowledge_base):
        """Generate a response using the Google Generative AI API"""
        try:
            # Create full prompt
            full_prompt = f"""
            You are a helpful {agent_type} assistant with the following knowledge base:
            
            {knowledge_base}
            
            Please respond to this user query based on your knowledge:
            {prompt}
            
            Provide a detailed and helpful response.
            """
            
            # Skip API call if not working
            if not self.api_working:
                return self._create_smart_fallback(agent_type, prompt, knowledge_base)
            
            try:
                # Try newer API version first
                if hasattr(self, 'api_version') and self.api_version == "new":
                    try:
                        model = self.genai.GenerativeModel('gemini-pro')
                        response = model.generate_content(full_prompt)
                        
                        if hasattr(response, 'text'):
                            return response.text
                        elif hasattr(response, 'parts'):
                            return response.parts[0].text
                        else:
                            return str(response)
                    except Exception as e:
                        print(f"Error with new API version: {e}")
                        # Fall through to old version
                
                # Try older API version
                response = self.genai.generate_text(prompt=full_prompt)
                
                if isinstance(response, dict) and 'candidates' in response:
                    return response['candidates'][0]['output']
                elif isinstance(response, list) and len(response) > 0:
                    return response[0]
                else:
                    return str(response)
                    
            except Exception as e:
                print(f"All API attempts failed: {e}")
                return self._create_smart_fallback(agent_type, prompt, knowledge_base)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._create_smart_fallback(agent_type, prompt, knowledge_base)
    
    def _create_smart_fallback(self, agent_type, prompt, knowledge_base):
        """Create a meaningful response when API fails"""
        # Look for keywords in the prompt
        keywords = prompt.lower().split()
        keywords = [k for k in keywords if len(k) > 3]  # Only consider words longer than 3 letters
        
        # Common medication names
        medications = ["paracetamol", "ibuprofen", "aspirin", "tylenol", "advil"]
        found_meds = [med for med in medications if med in prompt.lower()]
        
        if found_meds:
            medication = found_meds[0]
            if agent_type == "pharmacy":
                return f"Based on my pharmaceutical knowledge, {medication} is commonly used for pain relief and fever reduction. It's important to follow the recommended dosage and consult with a healthcare professional if you have specific questions about its use for your condition."
            else:
                return f"As an infusion specialist, I can tell you that {medication} is sometimes administered intravenously in clinical settings. However, most patients receive it orally. For specific medical advice, please consult with your healthcare provider."
        
        if "side effect" in prompt.lower():
            return "All medications can have potential side effects. It's important to read the medication guide and consult with your healthcare provider about specific side effects that may affect you based on your medical history."
        
        if "dose" in prompt.lower() or "dosage" in prompt.lower():
            return "I cannot provide specific dosage information as this should be determined by your healthcare provider based on your specific condition, age, weight, and other factors. Please follow the prescription or instructions provided by your doctor or pharmacist."
        
        # Default fallback response
        return f"As a {agent_type} assistant, I'm here to help with your questions. While I'm experiencing some technical difficulties accessing my knowledge base, I can tell you that {knowledge_base[:100]}... Please feel free to ask another question or try again later."

    def moderate_response(self, response, moderation_rules):
        """Check if response violates any moderation rules"""
        was_moderated = False
        moderated_response = response
        
        for rule in moderation_rules:
            rule_text = rule.lower()
            if any(phrase in response.lower() for phrase in rule_text.split(';')):
                was_moderated = True
                # Replace problematic content
                moderated_response = "I apologize, but I cannot provide the requested information due to our content policies. Please consult with a healthcare professional for personalized advice."
                break
                
        return was_moderated, moderated_response

# Fallback mechanism
class FallbackService:
    def check_conditions(self, message, agent_response, conditions):
        # Check if any fallback conditions are met
        for condition in conditions:
            if condition.lower() in message.lower() or condition.lower() in agent_response.lower():
                return True
        return False
        
    def call_external_service(self, external_url, api_key, context_data):
        # Mock implementation
        print(f"Calling external service at {external_url}")
        return {"status": "escalated", "ticket_id": str(uuid.uuid4())}

# Initialize services
google_ai = GoogleAI()
fallback_service = FallbackService()

# FastAPI Routes for Business Users
@app.post("/agents/", response_model=Agent)
async def create_agent(agent: Agent):
    with get_db() as conn:
        cursor = conn.cursor()
        agent_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO agents (id, name, type, prompt_template, knowledge_base, auto_followup) VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, agent.name, agent.type, agent.prompt_template, agent.knowledge_base, agent.auto_followup)
        )
        conn.commit()
        agent.id = agent_id
        st.rerun()
        return agent

@app.get("/agents/", response_model=List[Agent])
async def get_agents():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agents")
        agents = cursor.fetchall()
        return [Agent(
            id=agent["id"],
            name=agent["name"],
            type=agent["type"],
            prompt_template=agent["prompt_template"],
            knowledge_base=agent["knowledge_base"],
            auto_followup=agent["auto_followup"]
        ) for agent in agents]

@app.post("/moderation-rules/", response_model=ModerationRule)
async def create_moderation_rule(rule: ModerationRule):
    with get_db() as conn:
        cursor = conn.cursor()
        rule_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO moderation_rules (id, agent_id, rule_text) VALUES (?, ?, ?)",
            (rule_id, rule.agent_id, rule.rule_text)
        )
        conn.commit()
        rule.id = rule_id
        st.rerun()
        return rule

@app.post("/fallback-config/", response_model=FallbackConfig)
async def create_fallback_config(config: FallbackConfig):
    with get_db() as conn:
        cursor = conn.cursor()
        config_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO fallback_config (id, agent_id, condition_text, external_service_url, service_api_key) VALUES (?, ?, ?, ?, ?)",
            (config_id, config.agent_id, config.condition_text, config.external_service_url, config.service_api_key)
        )
        conn.commit()
        config.id = config_id
        st.rerun()
        return config

@app.get("/chat-history/{agent_id}", response_model=List[Dict])
async def get_chat_history(agent_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_history WHERE agent_id = ? ORDER BY timestamp DESC LIMIT 100", (agent_id,))
        history = cursor.fetchall()
        return [dict(h) for h in history]

# API endpoints for user chat
@app.post("/pharmacy/ask", response_model=ChatResponse)
async def pharmacy_ask(chat_message: ChatMessage):
    try:
        # Get agent data
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agents WHERE id = ?", (chat_message.agent_id,))
            agent = cursor.fetchone()
            
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get moderation rules for this agent
        moderation_rules = get_moderation_rules(chat_message.agent_id)
        
        # Get fallback conditions for this agent
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fallback_config WHERE agent_id = ?", (chat_message.agent_id,))
            fallback_configs = cursor.fetchall()
            fallback_conditions = [fc["condition_text"] for fc in fallback_configs]
        
        # Generate AI response
        ai_response = google_ai.generate_response(
            agent_type="pharmacy",
            prompt=chat_message.message,
            knowledge_base=agent["knowledge_base"]
        )
        
        # Check moderation
        was_moderated, final_response = google_ai.moderate_response(ai_response, moderation_rules)
        
        # Check fallback conditions
        was_escalated = False
        if fallback_service.check_conditions(chat_message.message, ai_response, fallback_conditions):
            was_escalated = True
            
            # Find the appropriate external service to call
            for fc in fallback_configs:
                if fc["condition_text"].lower() in chat_message.message.lower() or fc["condition_text"].lower() in ai_response.lower():
                    context_data = {
                        "user_id": chat_message.user_id,
                        "agent_id": chat_message.agent_id,
                        "message": chat_message.message,
                        "ai_response": ai_response,
                        "channel": chat_message.channel
                    }
                    
                    fallback_result = fallback_service.call_external_service(
                        fc["external_service_url"],
                        fc["service_api_key"],
                        context_data
                    )
                    
                    final_response += f"\n\n[This query has been escalated to our team. Ticket ID: {fallback_result.get('ticket_id', 'Unknown')}]"
                    break
        
        # Save to chat history
        chat_id = str(uuid.uuid4())
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (id, user_id, agent_id, message, response, channel, was_moderated, was_escalated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chat_id, chat_message.user_id, chat_message.agent_id, chat_message.message, final_response, chat_message.channel, was_moderated, was_escalated)
            )
            conn.commit()
        
        return ChatResponse(
            id=chat_id,
            response=final_response,
            was_moderated=was_moderated,
            was_escalated=was_escalated
        )
    except Exception as e:
        print(f"Error in pharmacy_ask: {e}")
        return ChatResponse(
            id=str(uuid.uuid4()),
            response=f"I apologize, but I encountered an error processing your request: {str(e)}",
            was_moderated=False,
            was_escalated=True
        )

@app.post("/infusion/ask", response_model=ChatResponse)
async def infusion_ask(chat_message: ChatMessage):
    return await process_chat_message(chat_message, "infusion")

async def process_chat_message(chat_message: ChatMessage, agent_type: str):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get agent details
            cursor.execute("SELECT * FROM agents WHERE id = ?", (chat_message.agent_id,))
            agent = cursor.fetchone()
            
            if not agent:
                return ChatResponse(
                    id=str(uuid.uuid4()),
                    response="Sorry, this agent could not be found. Please select a different agent.",
                    was_moderated=False,
                    was_escalated=False
                )
            
            # Get moderation rules - with better error handling
            try:
                cursor.execute("SELECT rule_text FROM moderation_rules WHERE agent_id = ?", (chat_message.agent_id,))
                moderation_rules = [rule["rule_text"] for rule in cursor.fetchall()]
            except Exception as e:
                print(f"Error retrieving moderation rules: {e}")
                moderation_rules = []
            
            # Get fallback conditions - with better error handling
            try:
                cursor.execute("SELECT * FROM fallback_config WHERE agent_id = ?", (chat_message.agent_id,))
                fallback_configs = cursor.fetchall()
                fallback_conditions = [fc["condition_text"] for fc in fallback_configs]
            except Exception as e:
                print(f"Error retrieving fallback configs: {e}")
                fallback_configs = []
                fallback_conditions = []
            
            # Generate AI response with better error handling
            try:
                ai_response = google_ai.generate_response(
                    agent_type=agent["type"],
                    prompt=chat_message.message,
                    knowledge_base=agent["knowledge_base"]
                )
            except Exception as e:
                print(f"Error generating AI response: {e}")
                ai_response = f"I'm sorry, I'm having trouble generating a response right now. Please try again later. (Error: {str(e)[:100]})"
            
            # Check moderation
            try:
                was_moderated, final_response = google_ai.moderate_response(ai_response, moderation_rules)
            except Exception as e:
                print(f"Error in moderation: {e}")
                was_moderated = False
                final_response = ai_response
            
            # Check fallback conditions
            was_escalated = False
            try:
                if fallback_service.check_conditions(chat_message.message, ai_response, fallback_conditions):
                    was_escalated = True
                    
                    # Find the appropriate external service to call
                    for fc in fallback_configs:
                        if fc["condition_text"].lower() in chat_message.message.lower() or fc["condition_text"].lower() in ai_response.lower():
                            context_data = {
                                "user_id": chat_message.user_id,
                                "agent_id": chat_message.agent_id,
                                "message": chat_message.message,
                                "ai_response": ai_response,
                                "channel": chat_message.channel
                            }
                            
                            fallback_result = fallback_service.call_external_service(
                                fc["external_service_url"],
                                fc["service_api_key"],
                                context_data
                            )
                            
                            final_response += f"\n\n[This query has been escalated to our team. Ticket ID: {fallback_result.get('ticket_id', 'Unknown')}]"
                            break
            except Exception as e:
                print(f"Error in fallback processing: {e}")
            
            # Save to chat history
            chat_id = str(uuid.uuid4())
            try:
                cursor.execute(
                    "INSERT INTO chat_history (id, user_id, agent_id, message, response, channel, was_moderated, was_escalated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (chat_id, chat_message.user_id, chat_message.agent_id, chat_message.message, final_response, chat_message.channel, was_moderated, was_escalated)
                )
                conn.commit()
            except Exception as e:
                print(f"Error saving to chat history: {e}")
            
            return ChatResponse(
                id=chat_id,
                response=final_response,
                was_moderated=was_moderated,
                was_escalated=was_escalated
            )
    except Exception as e:
        print(f"Critical error in process_chat_message: {e}")
        return ChatResponse(
            id=str(uuid.uuid4()),
            response=f"I'm sorry, an unexpected error occurred: {str(e)}. Please try again later.",
            was_moderated=False,
            was_escalated=False
        )

# Telegram Bot Integration
# Use environment variable with fallback to the provided token
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7646866559:AAHmQRB1PGfIduvsUvmdgWKVF4XcLmYr4cg")
TELEGRAM_ENABLED = True  # Set to False to disable Telegram if it keeps causing errors

async def start_telegram_bot():
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        async def start(update: Update, context: CallbackContext):
            await update.message.reply_text(
                "👋 Hi! I'm your Agentic CS Assistant. How can I help you today?"
            )
        
        async def handle_message(update: Update, context: CallbackContext):
            user_id = str(update.message.from_user.id)
            message_text = update.message.text
            
            # Default to first agent if agent selection isn't implemented in the UI
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM agents ORDER BY created_at LIMIT 1")
                agent = cursor.fetchone()
                
            if not agent:
                await update.message.reply_text("No agents configured in the system yet.")
                return
                
            # Determine if this is a pharmacy or infusion query (simplified)
            agent_type = "pharmacy" if "medicine" in message_text.lower() or "drug" in message_text.lower() else "infusion"
            
            # Prepare the chat message
            chat_message = ChatMessage(
                user_id=user_id,
                agent_id=agent["id"],
                message=message_text,
                channel="telegram"
            )
            
            # Process the message
            if agent_type == "pharmacy":
                response = await pharmacy_ask(chat_message)
            else:
                response = await infusion_ask(chat_message)
                
            # Send the response back to the user
            await update.message.reply_text(response.response)
        
        # Register handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Start the bot
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
    
    # FIXED HANDLING OF TELEGRAM EXCEPTIONS
    except telegram.error.TimedOut:
        print("Telegram bot timed out during initialization. Check your internet connection.")
    except Exception as e:
        print(f"Error starting Telegram bot: {e}")

# WhatsApp Integration using Twilio
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "whatsapp:+14155238886")
WHATSAPP_ENABLED = FLASK_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN

# Initialize Twilio client conditionally
twilio_client = None
if WHATSAPP_ENABLED:
    try:
        from twilio.rest import Client
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("Twilio client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Twilio client: {e}")
        WHATSAPP_ENABLED = False

# Add this function before the create_session function
def get_auth_css():
    """Return CSS styles for authentication pages"""
    return """
    <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 20px;
            color: #262730;
        }
        
        .login-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        .auth-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4361ee;
        }
        
        .metric-label {
            font-size: 14px;
            color: #555;
        }
        
        .dashboard-section {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        
        .section-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #262730;
        }
        
        .chart-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }
        
        .empty-state {
            text-align: center;
            padding: 30px;
            color: #666;
        }
        
        .empty-state-icon {
            font-size: 30px;
            margin-bottom: 10px;
        }
    </style>
    """

def create_session(user_id):
    """Create a new session for the user"""
    token = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(days=30)
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at)
        )
        conn.commit()
    
    return token

def validate_session(token):
    """Validate a session token and return user if valid"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT s.*, u.* FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP",
            (token,)
        )
        result = cursor.fetchone()
        
        if result:
            return {
                "id": result["user_id"],
                "username": result["username"],
                "email": result["email"],
                "is_admin": result["is_admin"]
            }
    
    return None

def end_session(token):
    """End a user session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()

def show_login_page():
    st.markdown(get_auth_css(), unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="login-header">Login</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password")
                return
            
            # Check credentials
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                user = cursor.fetchone()
                
                if not user or not verify_password(password, user["password_hash"], user["salt"]):
                    st.error("Invalid username or password")
                    return
                
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user["id"],)
                )
                conn.commit()
                
                # Create session
                token = create_session(user["id"])
                
                # Set user in session state
                st.session_state.user = {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "is_admin": user["is_admin"]
                }
                
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.button("Sign Up", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()
    
    st.markdown('</div>')
    
    # Show default credentials
    st.info("Default login: username=admin, password=admin123")
    st.markdown('</div>', unsafe_allow_html=True)

def show_signup_page():
    st.markdown(get_auth_css(), unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="login-header">Create Account</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account", use_container_width=True):
            # Validate inputs
            if not username or not email or not password or not confirm_password:
                st.error("Please fill all fields")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            
            # Check if username or email already exists
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    st.error("Username or email already exists")
                    return
                
                # Create new user
                user_id = str(uuid.uuid4())
                password_hash, salt = hash_password(password)
                
                try:
                    cursor.execute(
                        "INSERT INTO users (id, username, email, password_hash, salt, is_admin) VALUES (?, ?, ?, ?, ?, ?)",
                        (user_id, username, email, password_hash, salt, False)
                    )
                    conn.commit()
                    
                    # Create session
                    token = create_session(user_id)
                    
                    # Set user in session state
                    st.session_state.user = {
                        "id": user_id,
                        "username": username,
                        "email": email,
                        "is_admin": False
                    }
                    
                    st.success("Account created successfully!")
                    time.sleep(1)
                    st.session_state.show_signup = False
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error creating account: {str(e)}")
    
    with col2:
        if st.button("Back to Login", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()
    
    st.markdown('</div>')
    st.markdown('</div>', unsafe_allow_html=True)

def show_user_management():
    st.markdown(get_auth_css(), unsafe_allow_html=True)
    st.title("User Management")
    
    tabs = st.tabs(["All Users", "Create User", "User Activity"])
    
    with tabs[0]:
        # Get all users
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, is_admin, created_at, last_login FROM users ORDER BY created_at DESC")
            users = cursor.fetchall()
        
        # Display users in a table
        if users:
            user_data = []
            for user in users:
                user_data.append({
                    "ID": user["id"][:8] + "...",
                    "Username": user["username"],
                    "Email": user["email"],
                    "Role": "Admin" if user["is_admin"] else "User",
                    "Created": user["created_at"],
                    "Last Login": user["last_login"] or "Never"
                })
            
            # Convert to DataFrame for better display
            import pandas as pd
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found")
    
    with tabs[1]:
        # Create new user (admin only)
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<h3>Create New User</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username", key="new_user_username")
            new_email = st.text_input("Email", key="new_user_email")
        
        with col2:
            new_password = st.text_input("Password", type="password", key="new_user_password")
            is_admin = st.checkbox("Admin privileges", key="new_user_admin")
        
        if st.button("Create User", type="primary"):
            if not new_username or not new_email or not new_password:
                st.error("Please fill in all fields")
                return
            
            # Check if username or email already exists
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (new_username, new_email))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    st.error("Username or email already exists")
                    return
                
                # Create new user
                user_id = str(uuid.uuid4())
                password_hash, salt = hash_password(new_password)
                
                try:
                    cursor.execute(
                        "INSERT INTO users (id, username, email, password_hash, salt, is_admin) VALUES (?, ?, ?, ?, ?, ?)",
                        (user_id, new_username, new_email, password_hash, salt, is_admin)
                    )
                    conn.commit()
                    st.success(f"User {new_username} created successfully!")
                    time.sleep(1)
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error creating user: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        # User activity summary
        st.subheader("User Activity Summary")
        
        # Get activity stats
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Most active users
            cursor.execute("""
                SELECT u.username, COUNT(ch.id) as msg_count 
                FROM users u
                LEFT JOIN chat_history ch ON u.id = ch.user_id
                GROUP BY u.id
                ORDER BY msg_count DESC
                LIMIT 5
            """)
            active_users = cursor.fetchall()
            
            # Recent logins
            cursor.execute("""
                SELECT username, last_login 
                FROM users 
                WHERE last_login IS NOT NULL 
                ORDER BY last_login DESC
                LIMIT 5
            """)
            recent_logins = cursor.fetchall()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Most Active Users")
            if active_users:
                import altair as alt
                import pandas as pd
                
                # Create DataFrame
                df = pd.DataFrame([(u["username"], u["msg_count"]) for u in active_users], 
                                 columns=["Username", "Message Count"])
                
                # Create chart
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Message Count:Q'),
                    y=alt.Y('Username:N', sort='-x'),
                    color=alt.Color('Message Count:Q', scale=alt.Scale(scheme='blues'))
                ).properties(height=200)
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No user activity recorded yet")
        
        with col2:
            st.markdown("#### Recent Logins")
            if recent_logins:
                for user in recent_logins:
                    login_time = user["last_login"]
                    st.markdown(f"**{user['username']}** - {login_time}")
            else:
                st.info("No recent login data available")

# Streamlit Frontend
def get_analytics_data():
    """Get analytics data for the dashboard"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get agent counts
        cursor.execute("SELECT COUNT(*) as count FROM agents")
        result = cursor.fetchone()
        agent_count = result["count"] if result else 0
        
        # Get agent type distribution
        cursor.execute("SELECT type, COUNT(*) as count FROM agents GROUP BY type")
        agent_types = {row["type"]: row["count"] for row in cursor.fetchall()}
        
        # Get chat statistics
        cursor.execute("SELECT COUNT(*) as count FROM chat_history")
        result = cursor.fetchone()
        total_chats = result["count"] if result else 0
        
        # Get chats per agent
        cursor.execute("""
            SELECT a.name, COUNT(ch.id) as chat_count 
            FROM agents a
            LEFT JOIN chat_history ch ON a.id = ch.agent_id
            GROUP BY a.id
            ORDER BY chat_count DESC
        """)
        agent_usage = [(row["name"], row["chat_count"]) for row in cursor.fetchall()]
        
        # Get channel distribution
        cursor.execute("SELECT channel, COUNT(*) as count FROM chat_history GROUP BY channel")
        channels = {row["channel"]: row["count"] for row in cursor.fetchall()}
        
        # Get moderation statistics
        cursor.execute("SELECT COUNT(*) as count FROM chat_history WHERE was_moderated = 1")
        result = cursor.fetchone()
        moderated_count = result["count"] if result else 0
        
        # Get escalation statistics
        cursor.execute("SELECT COUNT(*) as count FROM chat_history WHERE was_escalated = 1")
        result = cursor.fetchone()
        escalated_count = result["count"] if result else 0
        
        # Get recent activity (last 7 days)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM chat_history 
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
        recent_activity = [(row["date"], row["count"]) for row in cursor.fetchall()]
        
        return {
            "agent_count": agent_count,
            "agent_types": agent_types,
            "total_chats": total_chats,
            "agent_usage": agent_usage,
            "channels": channels,
            "moderated_count": moderated_count,
            "escalated_count": escalated_count,
            "recent_activity": recent_activity
        }

import hashlib
import secrets

def hash_password(password):
    """Generate hash and salt for password"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

def verify_password(password, stored_hash, salt):
    """Verify password against stored hash"""
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash == stored_hash

# Move these functions BEFORE run_streamlit function

# Create a synchronous wrapper for async chat functions
def sync_pharmacy_ask(chat_message):
    """Synchronous wrapper for pharmacy_ask"""
    # This avoids the asyncio.run() within Streamlit's own event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(pharmacy_ask(chat_message))
        return result
    finally:
        loop.close()

def sync_infusion_ask(chat_message):
    """Synchronous wrapper for infusion_ask"""
    # This avoids the asyncio.run() within Streamlit's own event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_chat_message(chat_message, "infusion"))
        return result
    finally:
        loop.close()

# Now define run_streamlit AFTER these functions
def run_streamlit():
    st.set_page_config(page_title="Agentic CS Platform", layout="wide")
    
    # Check if user is logged in
    if "user" not in st.session_state:
        # Show login or signup page
        if st.session_state.get("show_signup", False):
            show_signup_page()
        else:
            show_login_page()
        return
    
    # Setup sidebar navigation
    st.sidebar.title("Agentic CS Platform")
    
    # Show user info in sidebar
    st.sidebar.markdown(f"### Logged in as: {st.session_state.user['username']}")
    
    # Add logout button
    if st.sidebar.button("Logout"):
        # Clear user session
        if "user" in st.session_state:
            del st.session_state.user
        st.rerun()
    
    # Add Telegram status indicator in sidebar
    telegram_status = "🟢 Active" if TELEGRAM_ENABLED else "🔴 Inactive"
    st.sidebar.markdown(f"### Telegram Bot: {telegram_status}")
    
    # Display Telegram info prominently in sidebar
    if TELEGRAM_ENABLED:
        st.sidebar.markdown("""
        **Telegram Bot Ready!**  
        Search for your bot on Telegram:  
        [@AgentyCSBot](https://t.me/AgentyCSBot)
        """)
    
    # Main navigation (all pages available without auth)
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Agent Management", "Moderation Rules", "Fallback Configuration", 
         "Chat History", "Chat with Agent", "Telegram Setup", "WhatsApp Setup", 
         "User Management"]
    )
    
    # Connect to the database
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if page == "Dashboard":
        st.title("Customer Service Analytics")
        
        # Get analytics data
        analytics = get_analytics_data()
        
        # Top-level metrics in a modern card layout
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        # Agent count card
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{analytics["agent_count"]}</div>
                <div class="metric-label">Total Agents</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Conversation count card
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{analytics["total_chats"]}</div>
                <div class="metric-label">Conversations</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Moderation count card
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{analytics["moderated_count"]}</div>
                <div class="metric-label">Moderated</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Escalation count card
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{analytics["escalated_count"]}</div>
                <div class="metric-label">Escalated</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity chart
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if analytics["recent_activity"]:
            import pandas as pd
            import plotly.express as px
            
            # Convert to dataframe
            activity_data = pd.DataFrame({
                "Date": [date for date, _ in analytics["recent_activity"]],
                "Conversations": [count for _, count in analytics["recent_activity"]]
            })
            
            # Create interactive chart
            fig = px.line(activity_data, x="Date", y="Conversations", 
                         markers=True, line_shape="spline",
                         color_discrete_sequence=["#4361ee"])
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <p>No conversation data available yet. Start chatting with your agents to see activity here.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Agent performance and distribution
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.markdown('<div class="section-header">Agent Types</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            if analytics["agent_types"]:
                # Make sure pandas and plotly are properly imported
                import pandas as pd
                import plotly.express as px
                
                try:
                    # Create pie chart data with error handling
                    agent_types_keys = list(analytics["agent_types"].keys())
                    agent_types_values = list(analytics["agent_types"].values())
                    
                    if agent_types_keys and agent_types_values:
                        agent_data = pd.DataFrame({
                            "Type": agent_types_keys,
                            "Count": agent_types_values
                        })
                        
                        # Create donut chart
                        fig = px.pie(agent_data, values="Count", names="Type", hole=0.4,
                                  color_discrete_sequence=px.colors.qualitative.Bold)
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=300,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No agent type data available.")
                except Exception as e:
                    st.error(f"Error creating agent types chart: {str(e)}")
                    st.info("Try creating more agents with different types.")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">👤</div>
                    <p>No agents created yet. Go to Agent Management to create your first agent.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with right_col:
            st.markdown('<div class="section-header">Most Active Agents</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            if analytics["agent_usage"] and any(count > 0 for _, count in analytics["agent_usage"]):
                import plotly.express as px
                
                # Create horizontal bar chart data
                usage_data = pd.DataFrame({
                    "Agent": [name for name, _ in analytics["agent_usage"]],
                    "Conversations": [count for _, count in analytics["agent_usage"]]
                })
                
                # Create horizontal bar chart
                fig = px.bar(usage_data, y="Agent", x="Conversations", orientation="h",
                           color="Conversations", color_continuous_scale="Blues")
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">💬</div>
                    <p>No conversation history yet. Try chatting with your agents.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Communication channels
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Communication Channels</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if analytics["channels"]:
            import plotly.express as px
            
            # Create channel data
            channel_data = pd.DataFrame({
                "Channel": list(analytics["channels"].keys()),
                "Count": list(analytics["channels"].values())
            })
            
            # Create bar chart
            fig = px.bar(channel_data, x="Channel", y="Count",
                       color="Channel", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📱</div>
                <p>No channel data yet. Interact with your agents through different channels to see data here.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Getting started guide
        with st.expander("Getting Started Guide"):
            st.markdown("""
            ## Welcome to Agentic CS Platform!
            
            Follow these steps to get started:
            
            1. **Create Agents** - Go to Agent Management to create AI agents for different departments
            2. **Set Up Rules** - Configure moderation rules to control agent responses
            3. **Configure Fallbacks** - Define when to escalate conversations to human agents
            4. **Start Chatting** - Use the Chat with Agent page to test your setup
            5. **Integrate Channels** - Connect Telegram and WhatsApp to expand your reach
            
            Need more help? Check the documentation or contact support.
            """)
    
    elif page == "Agent Management":
        st.header("Agent Management")
        
        # Add new agent form
        with st.expander("Add New Agent", expanded=False):
            with st.form("new_agent_form"):
                name = st.text_input("Agent Name")
                agent_type = st.selectbox("Agent Type", ["pharmacy", "infusion", "general"])
                prompt_template = st.text_area("Prompt Template", height=100)
                knowledge_base = st.text_area("Knowledge Base", height=100)
                auto_followup = st.text_area("Auto Follow-up Response (Optional)", height=50)
                
                submitted = st.form_submit_button("Create Agent")
                if submitted:
                    # Create agent in DB
                    agent_id = str(uuid.uuid4())
                    try:
                        cursor.execute(
                            "INSERT INTO agents (id, name, type, prompt_template, knowledge_base, auto_followup) VALUES (?, ?, ?, ?, ?, ?)",
                            (agent_id, name, agent_type, prompt_template, knowledge_base, auto_followup)
                        )
                        conn.commit()
                        st.success(f"Agent '{name}' created successfully!")
                        # Force refresh to show the new agent
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating agent: {e}")
        
        # List of existing agents
        st.subheader("Existing Agents")
        cursor.execute("SELECT * FROM agents")
        agents = cursor.fetchall()
        
        if not agents:
            st.warning("No agents found. Please create an agent using the form above.")
        
        for agent in agents:
            with st.expander(f"{agent['name']} ({agent['type']})"):
                st.text_area("Prompt Template", agent["prompt_template"], height=100, key=f"pt_{agent['id']}")
                st.text_area("Knowledge Base", agent["knowledge_base"], height=100, key=f"kb_{agent['id']}")
                if agent["auto_followup"]:
                    st.text_area("Auto Follow-up", agent["auto_followup"], height=50, key=f"af_{agent['id']}")
                
                if st.button("Update", key=f"update_{agent['id']}"):
                    # Get the updated values
                    updated_prompt = st.session_state[f"pt_{agent['id']}"]
                    updated_kb = st.session_state[f"kb_{agent['id']}"]
                    updated_followup = st.session_state.get(f"af_{agent['id']}", "")
                    
                    # Update in DB
                    cursor.execute(
                        "UPDATE agents SET prompt_template = ?, knowledge_base = ?, auto_followup = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (updated_prompt, updated_kb, updated_followup, agent["id"])
                    )
                    conn.commit()
                    st.success("Agent updated successfully!")
                
                if st.button("Test Agent", key=f"test_{agent['id']}"):
                    test_message = "How can you help me?"
                    response = google_ai.generate_response(
                        agent_type=agent["type"],
                        prompt=test_message,
                        knowledge_base=agent["knowledge_base"]
                    )
                    st.info(f"Test response: {response}")
    
    elif page == "Moderation Rules":
        st.header("Moderation Rules")
        
        # Get agents for dropdown
        cursor.execute("SELECT id, name FROM agents")
        agents = cursor.fetchall()
        agent_options = {a["id"]: a["name"] for a in agents}
        
        if agents:
            selected_agent_id = st.selectbox("Select Agent", options=list(agent_options.keys()), format_func=lambda x: agent_options[x])
            
            # Display existing rules
            st.subheader(f"Moderation Rules for {agent_options[selected_agent_id]}")
            cursor.execute("SELECT * FROM moderation_rules WHERE agent_id = ?", (selected_agent_id,))
            rules = cursor.fetchall()
            
            for rule in rules:
                cols = st.columns([4, 1])
                cols[0].text(rule["rule_text"])
                if cols[1].button("Delete", key=f"del_rule_{rule['id']}"):
                    cursor.execute("DELETE FROM moderation_rules WHERE id = ?", (rule["id"],))
                    conn.commit()
                    st.success("Rule deleted successfully!")
                    st.rerun()
            
            # Add new rule
            st.subheader("Add New Rule")
            new_rule = st.text_area("Rule Description", placeholder="e.g., Do not provide medical advice about...")
            if st.button("Add Rule"):
                rule_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO moderation_rules (id, agent_id, rule_text) VALUES (?, ?, ?)",
                    (rule_id, selected_agent_id, new_rule)
                )
                conn.commit()
                st.success("Rule added successfully!")
                st.rerun()
        else:
            st.warning("No agents created yet. Please create an agent first.")
    
    elif page == "Fallback Configuration":
        st.header("Fallback Configuration")
        
        # Get agents for dropdown
        cursor.execute("SELECT id, name FROM agents")
        agents = cursor.fetchall()
        agent_options = {a["id"]: a["name"] for a in agents}
        
        if agents:
            selected_agent_id = st.selectbox("Select Agent", options=list(agent_options.keys()), format_func=lambda x: agent_options[x])
            
            # Display existing fallback configs
            st.subheader(f"Fallback Configuration for {agent_options[selected_agent_id]}")
            cursor.execute("SELECT * FROM fallback_config WHERE agent_id = ?", (selected_agent_id,))
            configs = cursor.fetchall()
            
            for config in configs:
                with st.expander(f"Condition: {config['condition_text'][:30]}..."):
                    st.text(f"Full condition: {config['condition_text']}")
                    st.text(f"External service URL: {config['external_service_url']}")
                    if config['service_api_key']:
                        st.text("API Key: [Hidden]")
                    
                    if st.button("Delete", key=f"del_fb_{config['id']}"):
                        cursor.execute("DELETE FROM fallback_config WHERE id = ?", (config["id"],))
                        conn.commit()
                        st.success("Fallback configuration deleted!")
                        st.rerun()
            
            # Add new fallback config
            st.subheader("Add New Fallback Configuration")
            with st.form("new_fallback_form"):
                condition = st.text_area("Condition Text", placeholder="e.g., I need to speak to a human")
                external_url = st.text_input("External Service URL", placeholder="https://example.com/api/escalate")
                api_key = st.text_input("Service API Key (Optional)", type="password")
                
                submitted = st.form_submit_button("Add Configuration")
                if submitted:
                    config_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO fallback_config (id, agent_id, condition_text, external_service_url, service_api_key) VALUES (?, ?, ?, ?, ?)",
                        (config_id, selected_agent_id, condition, external_url, api_key)
                    )
                    conn.commit()
                    st.success("Fallback configuration added successfully!")
                    st.rerun()
        else:
            st.warning("No agents created yet. Please create an agent first.")
    
    elif page == "Chat History":
        st.header("Chat History")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            # Get agents for dropdown
            cursor.execute("SELECT id, name FROM agents")
            agents = cursor.fetchall()
            agent_options = {a["id"]: a["name"] for a in agents}
            agent_options["all"] = "All Agents"
            
            filter_agent = st.selectbox("Filter by Agent", options=["all"] + list(agent_options.keys())[:-1], format_func=lambda x: agent_options[x])
        
        with col2:
            filter_channel = st.selectbox("Filter by Channel", options=["all", "telegram", "web"])
        
        with col3:
            filter_escalated = st.checkbox("Show Escalated Only")
        
        # Build the query
        query = "SELECT ch.*, a.name as agent_name FROM chat_history ch JOIN agents a ON ch.agent_id = a.id WHERE 1=1"
        params = []
        
        if filter_agent != "all":
            query += " AND ch.agent_id = ?"
            params.append(filter_agent)
        
        if filter_channel != "all":
            query += " AND ch.channel = ?"
            params.append(filter_channel)
        
        if filter_escalated:
            query += " AND ch.was_escalated = 1"
        
        query += " ORDER BY ch.timestamp DESC LIMIT 100"
        
        # Execute the query
        cursor.execute(query, params)
        history = cursor.fetchall()
        
        # Display the results
        if not history:
            st.info("No chat history found matching the filters.")
        
        for msg in history:
            with st.expander(f"{msg['timestamp']} - {msg['agent_name']} ({msg['channel']})"):
                st.markdown("**User Message:**")
                st.text(msg["message"])
                
                st.markdown("**Agent Response:**")
                st.text(msg["response"])
                
                st.markdown("**Flags:**")
                if msg["was_moderated"]:
                    st.warning("This response was moderated")
                if msg["was_escalated"]:
                    st.error("This conversation was escalated")
    
    elif page == "Chat with Agent":
        st.header("Chat with Agent")
        
        # Get available agents
        cursor.execute("SELECT id, name, type FROM agents")
        agents = cursor.fetchall()
        
        if not agents:
            st.warning("No agents available. Please create an agent in Agent Management first.")
        else:
            # Agent selection
            selected_agent_id = st.selectbox(
                "Select Agent to Chat With", 
                options=[a["id"] for a in agents],
                format_func=lambda x: next((a["name"] + " (" + a["type"] + ")" for a in agents if a["id"] == x), x)
            )
            
            # Get agent details
            cursor.execute("SELECT * FROM agents WHERE id = ?", (selected_agent_id,))
            agent = cursor.fetchone()
            
            if agent:
                # Display agent info
                st.subheader(f"Chatting with {agent['name']}")
                st.caption(f"Type: {agent['type']}")
                
                # Initialize chat history in session state if not exists
                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = []
                
                # Display chat history
                for msg in st.session_state.chat_messages:
                    if msg["is_user"]:
                        st.markdown(f"**You:** {msg['text']}")
                    else:
                        st.markdown(f"**{agent['name']}:** {msg['text']}")
                
                # Chat input
                user_input = st.text_input("Type your message here", key="user_message")
                
                if st.button("Send") or (user_input and user_input != st.session_state.get("last_input", "")):
                    if user_input and user_input.strip():
                        # Save last input to prevent duplicate submissions
                        st.session_state.last_input = user_input
                        
                        # Add user message to chat history
                        st.session_state.chat_messages.append({"is_user": True, "text": user_input})
                        
                        # Create chat message for processing
                        chat_message = ChatMessage(
                            user_id="streamlit_user",
                            agent_id=selected_agent_id,
                            message=user_input,
                            channel="web"
                        )
                        
                        # Process message based on agent type
                        if agent["type"] == "pharmacy":
                            response = sync_pharmacy_ask(chat_message)
                        else:
                            response = sync_infusion_ask(chat_message)
                        
                        # Add agent response to chat history
                        st.session_state.chat_messages.append({"is_user": False, "text": response.response})
                        
                        # Force a rerun to update the display with the new messages
                        st.rerun()
                
                # Add a clear chat button
                if st.button("Clear Chat"):
                    st.session_state.chat_messages = []
                    st.rerun()
    
    elif page == "Telegram Setup":
        st.title("Telegram Bot Setup")
        
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.markdown("""
            ## How to Use Your Telegram Bot
            
            Your Telegram bot is ready to use! Follow these simple steps:
            
            1. **Find Your Bot**: Search for `@AgentyCSBot` on Telegram or click the button →
            2. **Start the Conversation**: Send `/start` to begin
            3. **Ask Questions**: Type any healthcare or pharmacy question
            4. **Get Responses**: Your AI agent will respond through Telegram
            
            ### Example Questions:
            - "What are the side effects of ibuprofen?"
            - "How should I store my medication?"
            - "Can you explain how insulin works?"
            - "What should I know about blood pressure medication?"
            
            ### Telegram Features:
            - 24/7 availability
            - Private and secure conversations
            - Easy to share with patients/customers
            - Works on all devices
            """)
            
            # QR code for Telegram bot
            st.markdown("### Scan to chat with your bot")
            bot_username = "AgentyCSBot"  # Change to your actual bot username
            qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://t.me/{bot_username}"
            st.image(qr_url, width=200)
        
        with col2:
            st.markdown("### Bot Status")
            st.markdown(f"**Status:** {telegram_status}")
            st.markdown(f"**Bot Token:** {'•'*16 + TELEGRAM_TOKEN[-4:]}")
            
            # Direct link button
            st.markdown("### Quick Access")
            st.markdown(f"[Open Telegram Chat](https://t.me/{bot_username})")
            
            if st.button("Test Bot Connection"):
                try:
                    import asyncio
                    from telegram import Bot
                    
                    async def test_connection():
                        bot = Bot(token=TELEGRAM_TOKEN)
                        bot_info = await bot.get_me()
                        return bot_info.username
                    
                    bot_username = asyncio.run(test_connection())
                    st.success(f"✅ Connected successfully to @{bot_username}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
            
            # Restart button
            if st.button("Restart Telegram Bot"):
                st.info("Attempting to restart the Telegram bot...")
                try:
                    import asyncio
                    asyncio.run(start_telegram_bot())
                    st.success("✅ Telegram bot restarted successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to restart: {str(e)}")
    
    elif page == "WhatsApp Setup":
        setup_whatsapp_page()
    
    elif page == "User Management":
        show_user_management()
    
    # Close the database connection
    conn.close()

# Add this function to find an available port
def find_available_port(start_port, end_port):
    """Find an available port in the given range"""
    import socket
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None  # No available ports in range

# Add WhatsApp setup to Streamlit
def setup_whatsapp_page():
    st.header("WhatsApp Integration Setup")
    
    if not FLASK_AVAILABLE:
        st.error("Flask is not installed. WhatsApp integration is disabled.")
        st.info("To enable WhatsApp integration, install the required packages with:")
        st.code("pip install flask twilio")
        return
    
    # Display current settings
    st.subheader("Current WhatsApp Settings")
    st.write(f"WhatsApp Integration: {'Enabled' if WHATSAPP_ENABLED else 'Disabled'}")
    
    # Show the webhook URL to configure in Twilio
    base_url = st.text_input("Your Application's Base URL (e.g., https://your-app.com)", "https://your-app.com")
    st.info(f"Configure your Twilio WhatsApp webhook to point to: {base_url}/whatsapp/webhook")
    
    # Twilio settings form
    st.subheader("Twilio Configuration")
    with st.form("twilio_config"):
        account_sid = st.text_input("Twilio Account SID", TWILIO_ACCOUNT_SID)
        auth_token = st.text_input("Twilio Auth Token", TWILIO_AUTH_TOKEN, type="password")
        phone_number = st.text_input("Twilio WhatsApp Phone Number", TWILIO_PHONE_NUMBER)
        
        submitted = st.form_submit_button("Save Twilio Settings")
        if submitted:
            # In a real app, you'd save these to a secure configuration
            st.success("Twilio settings saved! (Note: In this demo, settings are not permanently stored)")
    
    # Test WhatsApp
    st.subheader("Test WhatsApp Integration")
    with st.form("test_whatsapp"):
        test_number = st.text_input("Test Phone Number (with country code, e.g., +1234567890)")
        test_message = st.text_input("Test Message", "Hello from your Agentic CS Platform!")
        
        test_submitted = st.form_submit_button("Send Test Message")
        if test_submitted and test_number:
            send_whatsapp_message(test_number, test_message)
            st.success(f"Test message sent to {test_number}!")

# Update run_webhook_server function
def run_webhook_server():
    if not FLASK_AVAILABLE or not WHATSAPP_ENABLED:
        print("WhatsApp webhook server not started (Flask not available or WhatsApp disabled)")
        return
        
    # Find an available port for the webhook server
    webhook_port = find_available_port(3000, 4000)
    if webhook_port is None:
        print("No available ports found for webhook server")
        return
        
    print(f"Starting webhook server on port {webhook_port}")
    
    # Run Flask app in a separate thread
    import threading
    threading.Thread(target=lambda: webhook_app.run(host='0.0.0.0', port=webhook_port, debug=False, use_reloader=False)).start()

# Simple way to run application
if __name__ == "__main__":
    # Initialize database with default agents
    init_db()
    
    # Start the webhook server if WhatsApp is enabled
    if WHATSAPP_ENABLED:
        run_webhook_server()
    
    # Run Streamlit directly
    run_streamlit()

def get_moderation_rules(agent_id):
    """Get moderation rules for a specific agent"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rule_text FROM moderation_rules WHERE agent_id = ?", (agent_id,))
            rules = [row["rule_text"] for row in cursor.fetchall()]
        return rules
    except Exception as e:
        print(f"Error getting moderation rules: {e}")
        return []  # Return empty list if there's an error

# Check if Node.js is installed
def is_nodejs_installed():
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Set up WhatsApp Web API
def setup_whatsapp_web():
    if not is_nodejs_installed():
        print("Node.js is required for WhatsApp Web API but is not installed")
        return False
    
    # Create package.json
    package_json = {
        "name": "whatsapp-agent-integration",
        "version": "1.0.0",
        "dependencies": {
            "whatsapp-web.js": "^1.22.2",
            "qrcode-terminal": "^0.12.0",
            "express": "^4.18.2"
        }
    }
    
    with open("package.json", "w") as f:
        json.dump(package_json, f, indent=2)
    
    # Install dependencies
    try:
        subprocess.run(["npm", "install"], check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Failed to install WhatsApp Web dependencies: {e}")
        return False

# Create WhatsApp Web API server
def create_whatsapp_server():
    server_js = """
    const { Client, LocalAuth } = require('whatsapp-web.js');
    const qrcode = require('qrcode-terminal');
    const express = require('express');
    const app = express();
    app.use(express.json());
    
    const client = new Client({
        authStrategy: new LocalAuth(),
        puppeteer: { headless: true }
    });
    
    client.on('qr', (qr) => {
        // Generate and display QR code
        console.log('QR GENERATED - Scan with WhatsApp app:');
        qrcode.generate(qr, {small: true});
    });
    
    client.on('ready', () => {
        console.log('WhatsApp client is ready!');
    });
    
    // API endpoint to send messages
    app.post('/send-message', async (req, res) => {
        const { to, message } = req.body;
        try {
            // Format phone number
            const formattedNumber = to.includes('@c.us') ? to : `${to.replace('+', '')}@c.us`;
            
            // Send message
            const result = await client.sendMessage(formattedNumber, message);
            res.json({ success: true, messageId: result.id.id });
        } catch (error) {
            console.error('Error sending message:', error);
            res.status(500).json({ success: false, error: error.message });
        }
    });
    
    // API endpoint to receive messages
    client.on('message', async (msg) => {
        // Only process messages from users, not groups
        if (!msg.from.includes('g.us')) {
            console.log('Message received:', msg.body);
            
            // Forward to your agent API
            try {
                const response = await fetch('http://localhost:8501/api/process-message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        from: msg.from,
                        body: msg.body,
                        channel: 'whatsapp-web'
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    // Reply with the agent response
                    if (data.response) {
                        msg.reply(data.response);
                    }
                }
            } catch (error) {
                console.error('Error forwarding message to agent:', error);
                msg.reply('Sorry, I encountered an error processing your message.');
            }
        }
    });
    
    // Start WhatsApp client
    client.initialize();
    
    // Start Express server
    const PORT = 3500;
    app.listen(PORT, () => {
        console.log(`WhatsApp API server running on port ${PORT}`);
    });
    """
    
    with open("whatsapp_server.js", "w") as f:
        f.write(server_js)

# Add function to start WhatsApp server
def start_whatsapp_web_server():
    if not is_nodejs_installed():
        print("Cannot start WhatsApp Web server: Node.js is not installed")
        return False
    
    # Create server file if it doesn't exist
    if not os.path.exists("whatsapp_server.js"):
        create_whatsapp_server()
    
    # Start server in a subprocess
    try:
        process = subprocess.Popen(
            ["node", "whatsapp_server.js"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Monitor for QR code generation
        print("Starting WhatsApp Web server...")
        print("Waiting for QR code, please be patient...")
        
        for line in process.stdout:
            print(line.strip())
            if "QR GENERATED" in line:
                print("QR code generated! Scan with your WhatsApp app to connect.")
            elif "WhatsApp client is ready" in line:
                print("WhatsApp connected successfully!")
                break
        
        return True
    except Exception as e:
        print(f"Failed to start WhatsApp Web server: {e}")
        return False

# Add WhatsApp Web setup to Streamlit
def setup_whatsapp_web_page():
    st.header("WhatsApp Direct Integration (Unofficial)")
    
    st.warning("""
    ⚠️ **IMPORTANT WARNING**: 
    
    This integration uses unofficial WhatsApp Web API and is:
    - Not officially supported by WhatsApp/Meta
    - May violate WhatsApp's Terms of Service
    - Could potentially get your phone number banned
    - Not recommended for production use
    
    Use at your own risk and only for personal testing.
    """)
    
    if not is_nodejs_installed():
        st.error("Node.js is required but not installed. Please install Node.js before proceeding.")
        st.markdown("[Download Node.js](https://nodejs.org/)")
        return
    
    # Setup section
    st.subheader("Setup WhatsApp Web Integration")
    if st.button("Setup WhatsApp Integration"):
        with st.spinner("Installing required dependencies..."):
            success = setup_whatsapp_web()
            if success:
                st.success("Setup completed successfully!")
            else:
                st.error("Setup failed. Check the logs for details.")
    
    # Start server section
    st.subheader("Connect WhatsApp")
    st.info("Clicking the button below will start a WhatsApp Web server. You'll need to scan a QR code with your WhatsApp app to connect.")
    
    if st.button("Start WhatsApp Connection"):
        st.warning("Starting WhatsApp connection. A QR code will appear in the terminal/console window.")
        
        # Initialize in a separate thread to not block Streamlit
        import threading
        threading.Thread(target=start_whatsapp_web_server).start()
        
        st.info("Check your terminal/console window for the QR code to scan with WhatsApp.")
        st.markdown("After scanning, messages sent to your connected WhatsApp number will be processed by your agents.")
    
    # Test sending section
    st.subheader("Test Send Message")
    with st.form("send_whatsapp_message"):
        phone_number = st.text_input("Phone Number (with country code)", placeholder="e.g., +1234567890")
        message = st.text_area("Message", placeholder="Type your test message here")
        
        submitted = st.form_submit_button("Send Message")
        if submitted and phone_number and message:
            try:
                import requests
                response = requests.post(
                    "http://localhost:3500/send-message",
                    json={"to": phone_number, "message": message}
                )
                
                if response.status_code == 200:
                    st.success("Message sent successfully!")
                else:
                    st.error(f"Failed to send message. Error: {response.text}")
            except Exception as e:
                st.error(f"Error sending message: {e}")