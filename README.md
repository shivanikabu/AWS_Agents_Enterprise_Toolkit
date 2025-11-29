# 🤖 AI Agents Enterprise Toolkit

A sophisticated multi-agent system built with Streamlit that orchestrates AI agents for intelligent query processing, document retrieval (RAG), emotional intelligence, and quality assurance. The system dynamically routes queries through different agent architectures based on user persona detection.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 🎯 Intelligent Agent Orchestration
- **Planner Agent**: Analyzes query sentiment and detects customer persona
- **Orchestration Agent**: Routes queries through optimal agent trajectories
- **RAG Agent**: Retrieves relevant documents from vector database
- **Emotions Agent**: Analyzes emotional content and intensity
- **Calming Agent**: Generates empathetic responses for frustrated users
- **Best Practices Agent**: Enhances responses with technical guidance
- **Reflector Agent**: Evaluates response quality and detects issues
- **Response Agent**: Generates final responses using Claude 3.5 Sonnet
- **Feedback Agent**: Calculates convergence scores and recommendations

### 📊 Advanced Analytics Dashboard
- **Real-time Performance Monitoring**: Track token usage, latency, and costs
- **Quality Metrics**: Accuracy, hallucination detection, consistency scoring
- **Security Metrics**: Guardrail violations, jailbreak detection, safety filters
- **Trajectory Analysis**: Compare different agent architecture paths
- **Conversation History**: Review all queries and responses with persona detection

### 🔄 Multiple Agent Trajectories

**Trajectory 1 - Simple Query Path**
```
Planner → Orchestration → RAG → Reflector → Response → Feedback
```
- Optimal for: Straightforward questions
- Best completion rate (95%+)
- Lowest error propagation

**Trajectory 2 - Emotional Support Path**
```
Planner → Orchestration → Emotions → Calming → RAG → Reflector → Response → Feedback
```
- Optimal for: Angry or confused customers
- Highest persona sensitivity
- Fastest recovery time
- Best conversational coherence

**Trajectory 3 - Technical Precision Path**
```
Planner → Orchestration → RAG → Best Practices → Reflector → Response → Feedback
```
- Optimal for: Technical queries requiring detailed guidance
- Highest consistency index
- Best for step-by-step instructions

### 📚 Document Processing
- PDF upload and automatic text extraction
- Paragraph-based chunking for optimal context
- ChromaDB vector database for semantic search
- Embedding generation using Sentence Transformers
- Complete document context retrieval

### 🔒 Safety & Guardrails
- Configurable guardrail system
- Real-time quality evaluation
- Hallucination detection
- Bias and toxicity scoring
- Performance anomaly detection

## 🏗️ Architecture

### Project Structure
```
ai-agents-toolkit/
│
├── app.py                          # Main Streamlit application
├── backend.py                      # Agent orchestration backend
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration constants
│
├── models/
│   ├── __init__.py
│   └── agent_models.py             # Data models for agent responses
│
├── services/
│   ├── __init__.py
│   ├── aws_service.py              # AWS Bedrock integration
│   ├── langfuse_service.py         # Langfuse observability
│   ├── vector_db_service.py        # ChromaDB operations
│   └── document_processor.py       # PDF processing
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py               # Base agent interface
│   ├── planner_agent.py            # Query analysis & persona detection
│   ├── orchestration_agent.py      # Agent routing logic
│   ├── rag_agent.py                # Document retrieval
│   ├── emotions_agent.py           # Emotional analysis
│   ├── calming_agent.py            # Empathetic response generation
│   ├── best_practices_agent.py     # Technical enhancement
│   ├── reflector_agent.py          # Quality evaluation
│   ├── response_agent.py           # Final response generation
│   └── feedback_agent.py           # Convergence scoring
│
├── ui/
│   ├── __init__.py
│   ├── styles.py                   # CSS styling
│   ├── sidebar.py                  # Sidebar components
│   └── tabs/
│       └── __init__.py             # Tab modules (optional)
│
└── utils/
    ├── __init__.py
    └── helpers.py                  # Helper functions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- AWS account with Bedrock access
- (Optional) Langfuse account for advanced observability

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-agents-toolkit.git
cd ai-agents-toolkit
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔧 Configuration

### AWS Bedrock Setup

1. Navigate to the **Configuration** section in the sidebar
2. Enter your AWS credentials:
   - AWS Access Key
   - AWS Secret Key
   - AWS Region (default: us-east-1)
3. Click **Connect to AWS**

**Note**: Ensure you have access to `anthropic.claude-3-5-sonnet-20240620-v1:0` model in AWS Bedrock.

### Langfuse Setup (Optional)

1. Sign up for a free account at [langfuse.com](https://langfuse.com)
2. Navigate to the **Langfuse Connectivity** section
3. Enter your credentials:
   - Public Key
   - Secret Key
   - Host (default: https://cloud.langfuse.com)
4. Click **Connect to Langfuse**

### Document Upload

1. Upload PDF documents using the **Document Upload** section
2. Only PDF files are supported
3. Click **Process Documents** to extract and store text
4. Documents are chunked by paragraphs and stored in ChromaDB

### Agent Tuning

Adjust agent behavior using the **Agent Goal Assessment** sliders:
- **Risk Tolerance**: How much risk the system should accept (0.0 - 1.0)
- **Accuracy**: Importance of response accuracy (0.0 - 1.0)
- **Latency**: Importance of response speed (0.0 - 1.0)
- **Cost**: Importance of minimizing token costs (0.0 - 1.0)

### Guardrails

Define custom guardrails in the **Guardrails** text area:
```
No profanity
No PII disclosure
Factual responses only
No medical advice
```

## 📖 Usage Examples

### Example 1: Simple Query
```
Query: "What is the weather today?"
Detected Persona: simple query
Trajectory: Planner → Orchestration → RAG → Reflector → Response → Feedback
```

### Example 2: Frustrated Customer
```
Query: "I am so frustrated! How many times do I have to ask this?"
Detected Persona: angry customer
Trajectory: Planner → Orchestration → Emotions → Calming → RAG → Reflector → Response → Feedback
Response includes: Empathetic preamble + helpful solution
```

### Example 3: Technical Question
```
Query: "Can you tell me exactly how to configure the SSL certificate step by step?"
Detected Persona: precision ask
Trajectory: Planner → Orchestration → RAG → Best Practices → Reflector → Response → Feedback
Response includes: Detailed steps + warnings + best practices
```

## 📊 Dashboard Tabs

### 1. Agent Flow
- Execute queries and visualize agent execution in real-time
- See which agents are invoked and their actions
- Monitor performance indicators (🟢 Optimal, 🟡 Acceptable, 🔴 Warning)
- View detailed logs for each agent

### 2. Conversations
- Review conversation history
- See detected persona for each query
- View final responses and agent flow paths
- Filter by REAL MODE vs DEMO MODE

### 3. Agent Analytics
- **Performance Metrics**: Accuracy, latency, consistency scores
- **Token & Cost Analysis**: Track usage and costs per agent
- **Security Metrics**: Guardrail violations, jailbreak detection
- **Quality Metrics**: Hallucination scores, bias detection
- **Langfuse Integration**: Deep observability and tracing

### 4. Trajectory Analysis
- Compare performance across different agent architectures
- Metrics: Completion rate, consistency, error propagation, recovery time
- Persona sensitivity and conversational coherence scoring
- Identify optimal trajectory for different query types

## 🧪 Testing

### Manual Testing
```bash
# Test individual modules
python -c "from agents.planner_agent import PlannerAgent; print('✓ Planner Agent')"
python -c "from services.aws_service import AWSService; print('✓ AWS Service')"
python -c "from models.agent_models import PlannerResponse; print('✓ Models')"
```

### Unit Tests (Optional)
```bash
# Install pytest
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=agents --cov=services tests/
```

## 🔍 Performance Metrics Explained

### Trajectory Completion Rate
- Percentage of successful end-to-end agent chain executions
- Higher is better (>95% is excellent)

### Consistency Index
- Measures variance in responses for similar prompts
- Scale: 0-1, where 1 is perfect consistency

### Error Propagation Rate
- How errors cascade through the agent chain
- Lower is better (<5% is excellent)

### Recovery Time
- Time to recover from interruptions or escalations
- Measured in seconds, lower is better

### Persona Sensitivity
- Correlation between persona type and adaptation behaviors
- Scale: 0-1, higher indicates better emotional intelligence

### Conversational Coherence
- Topic retention and context continuity across turns
- Scale: 0-1, where 1 is perfect coherence

## 🛠️ Development

### Adding a New Agent

1. Create new agent file in `agents/`:
```python
# agents/my_new_agent.py
from agents.base_agent import BaseAgent

class MyNewAgent(BaseAgent):
    def __init__(self):
        super().__init__("My New Agent")
    
    def execute(self, *args, **kwargs):
        # Your logic here
        return {"agent": self.name, "detail": "..."}
```

2. Update `agents/__init__.py`:
```python
from .my_new_agent import MyNewAgent

__all__ = [..., 'MyNewAgent']
```

3. Integrate in `backend.py`:
```python
self.my_new_agent = MyNewAgent()
```

### Adding a New Service

1. Create service file in `services/`:
```python
# services/my_service.py
class MyService:
    def __init__(self):
        self.client = None
    
    def connect(self, credentials):
        # Connection logic
        pass
```

2. Update `services/__init__.py` and integrate in backend

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'agents'"
- Ensure you're running from the project root directory
- Verify all `__init__.py` files exist
- Try: `PYTHONPATH=. streamlit run app.py`

### "AWS Bedrock not connected"
- Verify AWS credentials are correct
- Check AWS region has Bedrock access
- Ensure Claude 3.5 Sonnet model is enabled

### "No documents found in database"
- Upload PDF documents using the sidebar
- Click "Process Documents" after upload
- Verify PDFs contain readable text (not scanned images)

### Vector database issues
- Click "Clear Vector Database" to reset
- Re-upload and process documents
- Check console logs for detailed errors

## 📝 Environment Variables (Optional)

Create a `.env` file for default configurations:
```env
AWS_DEFAULT_REGION=us-east-1
LANGFUSE_HOST=https://cloud.langfuse.com
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AWS Bedrock** - Claude 3.5 Sonnet LLM
- **Anthropic** - Claude model development
- **Streamlit** - Web application framework
- **ChromaDB** - Vector database
- **Langfuse** - LLM observability platform
- **Sentence Transformers** - Embedding models

## 📧 Contact

For questions or support, please open an issue on GitHub or contact:
- **Your Name** - [your.email@example.com](mailto:your.email@example.com)
- **Project Link**: [https://github.com/yourusername/ai-agents-toolkit](https://github.com/yourusername/ai-agents-toolkit)

## 🗺️ Roadmap

- [ ] Add support for more LLM providers (OpenAI, Cohere)
- [ ] Implement multi-document conversation memory
- [ ] Add voice input/output capabilities
- [ ] Create Docker deployment option
- [ ] Add automated testing suite
- [ ] Implement user authentication
- [ ] Add support for more document formats (DOCX, TXT, HTML)
- [ ] Create API endpoints for programmatic access
- [ ] Add multi-language support
- [ ] Implement advanced caching strategies

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ using Streamlit and AWS Bedrock**