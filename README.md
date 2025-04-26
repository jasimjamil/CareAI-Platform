Agentic CS Platform
🤖 AI-Powered Customer Service Platform
A comprehensive platform for building, managing, and deploying AI customer service agents across multiple channels. Designed for healthcare organizations that need intelligent, adaptive conversational agents to handle pharmacy and infusion-related inquiries.
!Agentic CS Dashboard
✨ Features
Intelligent AI Agents: Create specialized agents for pharmacy, infusion, and other healthcare services
Multi-Channel Support: Deploy agents on web, Telegram, and WhatsApp
Moderation Rules: Configure content policies to ensure compliant responses
Fallback Mechanisms: Seamlessly escalate to human agents when needed
Advanced Analytics: Track performance metrics and user interactions
Authentication System: Secure multi-user access with role management
Google AI Integration: Powered by Google's Generative AI models
🚀 Quick Start
Prerequisites
Python 3.8 or higher
SQLite (included in Python)
API keys for Google AI, Telegram, and WhatsApp (optional)
Installation
Clone the repository:
Install dependencies:
3. Run the application:
Access the platform at http://localhost:8501
📋 Setup Guide
Initial Configuration
The system comes with default pharmacy and infusion agents. Login with:
Username: admin
Password: admin123
Important: Change these credentials after first login!
Environment Variables
For enhanced security and integration, set the following environment variables:
Configuring Agents
1. Navigate to Agent Management
Create agents with specialized knowledge bases
Add moderation rules to ensure compliant responses
Configure fallback conditions for human escalation
🔌 Channel Integration
Telegram Setup
Create a bot through BotFather
Set the TELEGRAM_TOKEN environment variable
The bot will automatically use your configured agents
WhatsApp Setup
Create a Twilio account and configure WhatsApp sandbox
Set Twilio environment variables
Configure the webhook URL to point to your application
📊 Analytics Dashboard
The platform includes a comprehensive analytics dashboard:
Conversation volumes and trends
Channel distribution
Agent performance metrics
Moderation and escalation tracking
🏗️ Architecture
The platform consists of:
FastAPI Backend: Handling API requests and agent logic
Streamlit Frontend: User-friendly interface for management
SQLite Database: Storing agents, rules, and chat history
AI Integration: Connection to Google's Generative AI
Messaging Connectors: Interfaces for Telegram and WhatsApp
👥 User Management
Admins: Can create agents, moderation rules, and view all analytics
Regular Users: Can chat with agents and view limited analytics
Authentication: Secure session-based login system
🔒 Security Features
Password hashing with unique salts
Session management with expiration
Environment variable-based secrets management
Content moderation to prevent harmful responses
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📞 Support
For questions and support, please open an issue in the GitHub repository or contact the maintainers directly.
---
Developed with ❤️ for the healthcare community
