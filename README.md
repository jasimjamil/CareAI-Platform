Agentic CS Platform
AI-Powered Customer Service Solution
Agentic CS is a comprehensive AI-driven customer service platform designed specifically for healthcare organizations. It enables the creation, management, and deployment of intelligent, adaptive conversational agents to handle pharmacy and infusion-related inquiries across multiple communication channels.

Key Features
Intelligent AI Agents: Build specialized agents tailored for pharmacy, infusion, and other healthcare services.

Multi-Channel Support: Deploy agents seamlessly on web, Telegram, and WhatsApp platforms.

Moderation Rules: Define content policies to ensure compliant and safe responses from the agents.

Fallback Mechanisms: Enable smooth escalation to human agents whenever necessary.

Advanced Analytics: Track and analyze performance metrics, user interactions, and service quality.

Authentication System: Ensure secure, role-based multi-user access to the platform.

Google AI Integration: Leverage Google’s powerful generative AI models to enhance agent capabilities.

Quick Start Guide
Prerequisites
Python 3.8 or higher

SQLite (bundled with Python)

API keys for Google AI, Telegram, and WhatsApp (optional)

Installation Steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-repository/Agentic-CS.git
Install the necessary dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application: Start the platform by running:

bash
Copy
Edit
streamlit run app.py
Access the platform at http://localhost:8501.

Platform Configuration
Initial Setup
Upon the first login, use the following credentials:

Username: admin

Password: admin123
Note: Ensure you change the default credentials immediately after your first login.

Environment Variables
For enhanced security and integration, set the following environment variables:

GOOGLE_API_KEY

TELEGRAM_TOKEN

TWILIO_ACCOUNT_SID

TWILIO_AUTH_TOKEN

Agent Configuration
Navigate to the Agent Management section.

Create new agents by specifying their knowledge bases.

Define moderation rules to ensure agents comply with content policies.

Set fallback mechanisms for smooth escalation to human agents.

Channel Integration
Telegram Integration
Create a bot using BotFather on Telegram.

Set the TELEGRAM_TOKEN environment variable.

The bot will automatically use the configured agents for handling queries.

WhatsApp Integration
Create a Twilio account and configure the WhatsApp sandbox.

Set the necessary Twilio environment variables.

Configure the webhook URL to point to your application to start processing messages.

Analytics Dashboard
The platform includes a powerful analytics dashboard to monitor key performance metrics:

Conversation volumes and trends.

Channel distribution (Telegram, WhatsApp, Web).

Agent performance statistics.

Moderation and escalation metrics.

System Architecture
The platform’s architecture consists of:

Backend: FastAPI-based server handling API requests and agent interactions.

Frontend: User-friendly interface built with Streamlit for agent management and analytics.

Database: SQLite for storing agent configurations, rules, and interaction history.

AI Integration: Leveraging Google’s Generative AI models to power the conversational agents.

Messaging Connectors: Interfaces to integrate with Telegram and WhatsApp APIs.

User Management
Admin Users: Full access to agent creation, moderation rule management, and analytics viewing.

Regular Users: Can interact with agents and view limited analytics.

Authentication: Secure, session-based login system with role management.

Security Features
Password Security: Passwords are hashed with unique salts for each user.

Session Management: Sessions are securely managed with expiration policies.

Secrets Management: Environment variable-based secrets handling ensures secure integration.

Content Moderation: The platform includes advanced content moderation to prevent harmful or non-compliant responses.

License
This project is licensed under the MIT License. Please refer to the LICENSE file for further details.

Contributing
Contributions to the project are welcome! To contribute:

Fork the repository.

Create a feature branch:

bash
Copy
Edit
git checkout -b feature/your-feature
Commit your changes:

bash
Copy
Edit
git commit -m 'Add new feature'
Push to your fork and submit a pull request.

Support
For any questions or support requests, please open an issue in the repository or contact the maintainers directly.

Developed with ❤️ for the healthcare community










