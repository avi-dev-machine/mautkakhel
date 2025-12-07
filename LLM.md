# AI.py (LLM) - Comprehensive Documentation

## Overview
`ai.py` is an intelligent fitness analysis module that uses Google's Gemini 2.5 Flash large language model (LLM) to provide personalized exercise feedback, performance analysis, and sport recommendations. The module reads exercise metrics from `exercise_metrics.txt` and generates actionable insights using advanced natural language processing.

---

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [LLM Integration](#llm-integration)
3. [Analysis Pipeline](#analysis-pipeline)
4. [Prompt Engineering](#prompt-engineering)
5. [Important Functions](#important-functions)
6. [Input/Output Structure](#inputoutput-structure)
7. [LangChain Framework](#langchain-framework)
8. [Configuration Details](#configuration-details)
9. [Usage Examples](#usage-examples)

---

## Core Architecture

### Technology Stack

```
┌─────────────────────────────────────────────┐
│          ai.py Module Architecture          │
└─────────────────────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  Environment Configuration    │
    │  (dotenv + .env file)         │
    │  • GOOGLE_API_KEY             │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  File I/O                     │
    │  Read exercise_metrics.txt    │
    │  JSON format data             │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  LangChain Framework          │
    │  ChatGoogleGenerativeAI       │
    │  • Model: gemini-2.5-flash    │
    │  • Temperature: 0             │
    │  • Max retries: 2             │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  Prompt Construction          │
    │  • System message (role)      │
    │  • Human message (data)       │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  LLM Inference                │
    │  Google Gemini API call       │
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  Response Generation          │
    │  • Performance analysis       │
    │  • Improvement suggestions    │
    │  • Sport recommendations      │
    └───────────────────────────────┘
                    ↓
              [AI Response]
```

### Key Components

```python
# Environment Management
load_dotenv()  # Load .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Instance
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Message Structure
messages = [
    ("system", system_prompt),
    ("human", user_query_with_data)
]

# Inference
response = llm.invoke(messages)
```

---

## LLM Integration

### Google Gemini 2.5 Flash

#### Model Specifications:
```python
Model Name: gemini-2.5-flash
Developer: Google DeepMind
Type: Large Language Model (Multimodal)
Architecture: Transformer-based
Context Window: ~1M tokens (extended context)
Speed: Optimized for fast inference (Flash variant)
Capabilities:
  - Text analysis and generation
  - Reasoning and logical inference
  - Domain-specific knowledge (fitness, sports, biomechanics)
  - Personalized recommendations
```

#### Model Configuration:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    
    # Temperature: Controls randomness
    # 0 = Deterministic (same input → same output)
    # 1 = Creative (varied outputs)
    temperature=0,
    
    # Max tokens: None = use model default
    max_tokens=None,
    
    # Timeout: None = no timeout limit
    timeout=None,
    
    # Max retries: Retry on API failures
    max_retries=2
)
```

#### Why Gemini 2.5 Flash?
1. **Fast Inference**: Flash variant optimized for speed
2. **Large Context**: Can handle extensive exercise metrics
3. **Domain Knowledge**: Pre-trained on fitness/sports data
4. **Cost-Effective**: Lower cost per token vs full model
5. **Reliable**: Production-ready with retry logic

---

## Analysis Pipeline

### Step-by-Step Process

#### Step 1: Load Environment Variables
```python
from dotenv import load_dotenv
load_dotenv()  # Reads .env file

# .env file structure:
# GOOGLE_API_KEY=your_api_key_here

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

**Purpose:**
- Secure API key storage (not hardcoded)
- Environment-specific configuration
- Easy deployment across environments

#### Step 2: Read Exercise Metrics
```python
try:
    with open(metrics_file, "r") as f:
        exercise_data = f.read()
except FileNotFoundError:
    exercise_data = "No exercise metrics found. Please complete a workout first."
```

**Input Format (exercise_metrics.txt):**
```json
{
  "exercise": "squat",
  "timestamp": "2025-12-06 15:30:00",
  "reps": {
    "total": 15,
    "good_form": 12,
    "bad_form": 3
  },
  "depth_analysis": {
    "max_depth": 85,
    "avg_depth": 95,
    "depths": [90, 95, 85, 88, 92, ...]
  },
  "tempo": {
    "avg_eccentric": 2.1,
    "avg_concentric": 1.8,
    "eccentric_times": [2.0, 2.2, 2.1, ...],
    "concentric_times": [1.8, 1.9, 1.7, ...]
  },
  "form_violations": {
    "torso_lean_count": 2,
    "knee_cave_count": 1
  },
  "overall_score": 87.5
}
```

#### Step 3: Initialize LLM
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,      # Deterministic responses
    max_tokens=None,    # No token limit
    timeout=None,       # No timeout
    max_retries=2       # Retry on failure
)
```

#### Step 4: Construct Prompt
```python
messages = [
    (
        "system",
        """You are a helpful fitness assistant that analyzes exercise 
        metrics and suggests improvements and tips. At last, suggest 
        which sport the user should try based on their performance data. 
        Provide specific, actionable feedback based on the scores and 
        data provided."""
    ),
    (
        "human",
        f"""Here are my exercise metrics:

{exercise_data}

Please analyze my performance and provide specific suggestions 
for improvement."""
    )
]
```

#### Step 5: LLM Inference
```python
ai_msg = llm.invoke(messages)
response_text = ai_msg.content
```

**What Happens During Inference:**
```
1. Messages converted to API format
2. HTTP request to Google Gemini API
3. Model processes exercise metrics
4. Generates personalized analysis
5. Returns structured response
6. Retry logic on failure (max 2 retries)
```

#### Step 6: Return Response
```python
return ai_msg.content  # String containing AI analysis
```

---

## Prompt Engineering

### System Prompt Design

#### Current System Prompt:
```
You are a helpful fitness assistant that analyzes exercise metrics 
and suggests improvements and tips. At last, suggest which sport 
the user should try based on their performance data. Provide specific, 
actionable feedback based on the scores and data provided.
```

#### Breakdown:
1. **Role Definition**: "helpful fitness assistant"
   - Sets conversational tone
   - Establishes domain expertise
   
2. **Primary Task**: "analyzes exercise metrics and suggests improvements"
   - Clear objective
   - Actionable focus
   
3. **Secondary Task**: "suggest which sport the user should try"
   - Personalized recommendations
   - Based on performance patterns
   
4. **Quality Guidelines**: "specific, actionable feedback"
   - Avoid generic advice
   - Concrete suggestions

#### Prompt Engineering Principles Applied:
1. **Clarity**: Clear role and task definition
2. **Specificity**: "specific, actionable feedback"
3. **Context**: References "scores and data"
4. **Structure**: Logical flow (analyze → suggest → recommend)
5. **Constraints**: "based on their performance data"

### User Prompt Structure

```python
f"""Here are my exercise metrics:

{exercise_data}

Please analyze my performance and provide specific suggestions 
for improvement."""
```

#### Components:
1. **Data Injection**: `{exercise_data}` - JSON metrics
2. **Request**: "analyze my performance"
3. **Expectation**: "specific suggestions for improvement"

---

## Important Functions

### 1. `analyze_exercise_metrics(metrics_file="exercise_metrics.txt")`

**Purpose:** Main function that orchestrates the entire analysis pipeline.

**Parameters:**
- `metrics_file` (str, default="exercise_metrics.txt"): Path to metrics file

**Returns:**
- `str`: AI-generated analysis and recommendations

**Algorithm:**
```python
def analyze_exercise_metrics(metrics_file="exercise_metrics.txt"):
    # Step 1: Read metrics file
    try:
        with open(metrics_file, "r") as f:
            exercise_data = f.read()
    except FileNotFoundError:
        exercise_data = fallback_message
    
    # Step 2: Initialize LLM
    llm = ChatGoogleGenerativeAI(...)
    
    # Step 3: Construct messages
    messages = [
        ("system", system_prompt),
        ("human", user_prompt_with_data)
    ]
    
    # Step 4: Invoke LLM
    ai_msg = llm.invoke(messages)
    
    # Step 5: Return response
    return ai_msg.content
```

**Error Handling:**
```python
# File not found
except FileNotFoundError:
    exercise_data = "No exercise metrics found..."
    # Continues with fallback message instead of crashing

# API errors (handled by LangChain)
max_retries=2  # Automatic retry on failure
```

**Example Usage:**
```python
# Default usage
analysis = analyze_exercise_metrics()
print(analysis)

# Custom file path
analysis = analyze_exercise_metrics("custom_metrics.txt")
```

---

### 2. `load_dotenv()`

**Purpose:** Load environment variables from `.env` file.

**Library:** `python-dotenv`

**Process:**
```
1. Search for .env file in current directory
2. Parse key=value pairs
3. Load into os.environ dictionary
4. Make accessible via os.getenv()
```

**Example .env File:**
```bash
GOOGLE_API_KEY=AIzaSyC1234567890abcdefghijklmnop
ENVIRONMENT=production
DEBUG=false
```

**Usage:**
```python
load_dotenv()  # Load all variables
api_key = os.getenv("GOOGLE_API_KEY")  # Access variable
```

---

### 3. `llm.invoke(messages)`

**Purpose:** Send messages to LLM and get response.

**Framework:** LangChain

**Parameters:**
- `messages` (List[Tuple[str, str]]): List of (role, content) tuples

**Returns:**
- `AIMessage` object with `.content` attribute

**Internal Process:**
```
1. Convert messages to API format
2. Add API key to request headers
3. Send HTTP POST to Gemini API
4. Parse JSON response
5. Extract generated text
6. Return AIMessage object
```

**Message Roles:**
- `"system"`: Instructions for the AI (role definition)
- `"human"`: User input/query
- `"ai"`: AI's previous responses (for multi-turn chat)

**Example:**
```python
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "What is 2+2?")
]
response = llm.invoke(messages)
print(response.content)  # "4"
```

---

## Input/Output Structure

### Input: exercise_metrics.txt

#### Format: JSON

#### Structure for Different Exercises:

**Pushups:**
```json
{
  "exercise": "pushup",
  "timestamp": "2025-12-06 14:30:00",
  "reps": {"total": 20, "good_form": 18, "bad_form": 2},
  "angle_analysis": {
    "avg_elbow_angle": 95.2,
    "min_elbow": 85,
    "max_elbow": 145
  },
  "form_score": 90.0,
  "overall_score": 88.5
}
```

**Squats:**
```json
{
  "exercise": "squat",
  "reps": {"total": 15, "good_form": 12, "bad_form": 3},
  "depth_analysis": {
    "max_depth": 85,
    "avg_depth": 95,
    "deep_squats": 8
  },
  "tempo": {
    "avg_eccentric": 2.1,
    "avg_concentric": 1.8
  },
  "form_violations": {
    "torso_lean_count": 2,
    "knee_cave_count": 1
  },
  "sticking_points": [95, 100, 98],
  "overall_score": 87.5
}
```

**Sit-and-Reach:**
```json
{
  "exercise": "sitnreach",
  "max_reach": 245,
  "avg_reach": 230,
  "flexibility_score": 85,
  "hip_flexibility": "excellent",
  "symmetry_score": 92,
  "overall_score": 88.0
}
```

### Output: AI-Generated Analysis

#### Example Output:

```
Exercise Performance Analysis
============================

Overall Performance: 87.5/100 - EXCELLENT

Strengths:
✓ Excellent depth consistency (avg 95°)
✓ Good tempo control (2.1s eccentric, 1.8s concentric)
✓ 80% of reps with good form

Areas for Improvement:

1. Depth Quality (3 shallow reps detected)
   Suggestion: Focus on hitting parallel (90°) on every rep.
   Drill: Box squats to practice depth consistency.

2. Torso Lean (2 violations)
   Suggestion: Keep chest up and core engaged throughout movement.
   Drill: Goblet squats to reinforce upright posture.

3. Sticking Points at 95-100°
   Suggestion: This indicates weakness in mid-range.
   Drill: Pause squats at parallel (3-second hold).

Progressive Overload Plan:
- Week 1-2: Practice depth with lighter weight
- Week 3-4: Add pause squats (3 sets of 8)
- Week 5-6: Increase weight by 5-10%

Sport Recommendations Based on Performance:
🏋️ Powerlifting: Strong depth control and tempo awareness
🚴 Cycling: Good leg endurance and control
🏃 Running: Solid form and consistency

Primary Recommendation: POWERLIFTING
Your controlled tempo and depth consistency suggest strong 
potential for strength sports. Consider trying competitive 
powerlifting or strength training programs.

Keep up the excellent work!
```

#### Output Characteristics:
- **Structured**: Clear sections (strengths, improvements, recommendations)
- **Specific**: References actual metrics (95°, 2.1s, 80%)
- **Actionable**: Concrete drills and exercises
- **Personalized**: Based on individual performance data
- **Encouraging**: Positive tone with constructive feedback

---

## LangChain Framework

### What is LangChain?

**LangChain** is a framework for developing applications powered by language models.

**Key Features:**
1. **Model Abstraction**: Unified interface for different LLMs
2. **Prompt Management**: Templates and prompt engineering tools
3. **Chains**: Sequence multiple LLM calls
4. **Memory**: Conversation history management
5. **Error Handling**: Automatic retries and fallbacks

### ChatGoogleGenerativeAI Class

#### Inheritance:
```
BaseChatModel (LangChain)
    ↓
ChatGoogleGenerativeAI
    ↓
Gemini API Integration
```

#### Key Methods:
```python
llm.invoke(messages)         # Synchronous inference
llm.ainvoke(messages)        # Async inference
llm.stream(messages)         # Streaming responses
llm.batch(messages_list)     # Batch processing
```

#### Configuration Parameters:

```python
ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",      # Model selection
    temperature=0,                  # Randomness (0-1)
    max_tokens=None,               # Response length limit
    timeout=None,                   # Request timeout (seconds)
    max_retries=2,                 # Retry attempts
    top_p=1.0,                     # Nucleus sampling
    top_k=None,                    # Top-k sampling
    n=1,                           # Number of responses
    stop=None,                     # Stop sequences
    presence_penalty=0,            # Penalize repetition
    frequency_penalty=0            # Penalize frequent tokens
)
```

### Temperature Parameter Explained

#### Temperature Scale (0 to 1):
```
Temperature = 0
├─ Deterministic
├─ Always picks highest probability token
├─ Same input → Same output
├─ Best for: Analysis, factual tasks
└─ Example: "2+2" → Always "4"

Temperature = 0.5
├─ Balanced
├─ Mix of consistency and creativity
├─ Slight variations in output
├─ Best for: General conversation
└─ Example: "Describe a sunset" → Varied but coherent

Temperature = 1.0
├─ Creative
├─ Samples from probability distribution
├─ High variation in output
├─ Best for: Creative writing, brainstorming
└─ Example: "Write a poem" → Very diverse outputs
```

#### Why Temperature=0 for Fitness Analysis?
1. **Consistency**: Same metrics → Same analysis
2. **Reliability**: Predictable recommendations
3. **Accuracy**: Factual, data-driven feedback
4. **Trust**: Users expect consistent coaching advice

---

## Configuration Details

### Environment Variables

#### .env File Structure:
```bash
# Required
GOOGLE_API_KEY=AIzaSyC1234567890abcdefghijklmnopqrstuvwxyz

# Optional
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
```

#### Security Best Practices:
```bash
# .gitignore
.env
*.env
.env.*

# Never commit:
❌ GOOGLE_API_KEY=actual_key_here

# Instead, provide template:
✅ .env.example:
   GOOGLE_API_KEY=your_api_key_here
```

### API Key Management

#### Obtaining Google API Key:
```
1. Go to https://aistudio.google.com/
2. Sign in with Google account
3. Navigate to "Get API Key"
4. Create new project (or select existing)
5. Generate API key
6. Copy key to .env file
```

#### Key Format:
```
Length: 39 characters
Pattern: AIzaSy[A-Za-z0-9_-]{33}
Example: AIzaSyC1234567890abcdefghijklmnopqrstuvwxyz
```

---

## Usage Examples

### Example 1: Basic Usage
```python
from ai import analyze_exercise_metrics

# Run analysis with default file
result = analyze_exercise_metrics()
print(result)
```

### Example 2: Custom Metrics File
```python
# Analyze specific workout session
result = analyze_exercise_metrics("session_2025_12_06.txt")
print(result)
```

### Example 3: Integration with Test.py
```python
# In test.py after workout completion
import json
from ai import analyze_exercise_metrics

# Save metrics
with open('exercise_metrics.txt', 'w') as f:
    f.write(json.dumps(metrics, indent=2))

# Get AI analysis
ai_feedback = analyze_exercise_metrics()
print("\n" + "="*50)
print("AI COACH FEEDBACK")
print("="*50)
print(ai_feedback)
```

### Example 4: Multiple Exercise Analysis
```python
# Analyze different exercises
exercises = ['pushup', 'squat', 'situp']

for exercise in exercises:
    metrics_file = f"{exercise}_metrics.txt"
    analysis = analyze_exercise_metrics(metrics_file)
    
    print(f"\n=== {exercise.upper()} ANALYSIS ===")
    print(analysis)
```

### Example 5: Error Handling
```python
try:
    result = analyze_exercise_metrics("missing_file.txt")
    print(result)
except Exception as e:
    print(f"Error: {e}")
    # Fallback: provide generic advice
    print("Complete a workout to get personalized analysis!")
```

### Example 6: Streaming Response (Advanced)
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=True
)

# Stream response token by token
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Example 7: Batch Processing
```python
# Analyze multiple users' workouts
user_files = [
    "user1_metrics.txt",
    "user2_metrics.txt",
    "user3_metrics.txt"
]

analyses = []
for file in user_files:
    result = analyze_exercise_metrics(file)
    analyses.append(result)

# Generate comparison report
print("Team Performance Summary:")
for i, analysis in enumerate(analyses):
    print(f"\nUser {i+1}:")
    print(analysis[:200] + "...")  # Preview
```

---

## Advanced Features

### Custom Prompt Templates

#### Using LangChain Prompt Templates:
```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert {sport} coach analyzing {exercise} 
    performance. Focus on {focus_area}."""),
    ("human", "Metrics: {metrics}\n\nProvide analysis.")
])

messages = template.format_messages(
    sport="powerlifting",
    exercise="squat",
    focus_area="depth and form",
    metrics=exercise_data
)

response = llm.invoke(messages)
```

### Memory for Multi-Turn Conversations

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# First interaction
messages = [
    ("system", system_prompt),
    ("human", f"Metrics: {metrics}\nAnalyze my squats.")
]
response1 = llm.invoke(messages)
memory.save_context({"input": messages[-1][1]}, {"output": response1.content})

# Follow-up question
messages.append(("human", "What drills should I focus on?"))
response2 = llm.invoke(messages)
```

### Structured Output Parsing

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class FitnessAnalysis(BaseModel):
    overall_score: float = Field(description="Overall performance score")
    strengths: list[str] = Field(description="List of strengths")
    improvements: list[str] = Field(description="Areas to improve")
    sport_recommendation: str = Field(description="Recommended sport")

parser = PydanticOutputParser(pydantic_object=FitnessAnalysis)

# Add parser instructions to prompt
prompt = f"{system_prompt}\n{parser.get_format_instructions()}"
```

---

## Performance Considerations

### API Call Costs

#### Gemini 2.5 Flash Pricing (Approximate):
```
Input: $0.10 per 1M tokens
Output: $0.30 per 1M tokens

Typical analysis:
- Input: ~500 tokens (metrics + prompt)
- Output: ~800 tokens (analysis)
- Cost per analysis: ~$0.0003 (negligible)
```

### Response Time

```
Average latency:
- Network: 100-300ms
- Model inference: 500-1500ms
- Total: 600-1800ms (~1 second)

Factors affecting speed:
- Geographic distance to API
- Network conditions
- Response length
- Server load
```

### Rate Limits

```
Gemini API limits (free tier):
- 60 requests per minute
- 1,500 requests per day

For production:
- Use paid tier for higher limits
- Implement request queuing
- Add exponential backoff
```

### Optimization Tips

```python
# 1. Reduce input size
metrics_summary = json.dumps(essential_metrics_only)

# 2. Limit response length
llm = ChatGoogleGenerativeAI(max_tokens=500)

# 3. Cache common analyses
from functools import lru_cache

@lru_cache(maxsize=100)
def analyze_cached(metrics_hash):
    return analyze_exercise_metrics()

# 4. Batch processing
# Analyze multiple users in single call
batch_prompt = f"Analyze these workouts:\n{all_metrics}"
```

---

## Error Handling

### Common Errors and Solutions

#### 1. API Key Not Found
```python
Error: google.generativeai.types.client.APIKeyNotFoundError

Solution:
- Check .env file exists
- Verify GOOGLE_API_KEY is set
- Ensure load_dotenv() is called
```

#### 2. File Not Found
```python
Error: FileNotFoundError: exercise_metrics.txt

Solution:
- Run test.py to generate metrics first
- Provide custom file path
- Use try-except to provide fallback
```

#### 3. Rate Limit Exceeded
```python
Error: google.api_core.exceptions.ResourceExhausted: 429

Solution:
- Implement exponential backoff
- Reduce request frequency
- Upgrade to paid tier
```

#### 4. Network Timeout
```python
Error: requests.exceptions.Timeout

Solution:
llm = ChatGoogleGenerativeAI(
    timeout=30,      # Increase timeout
    max_retries=3    # More retry attempts
)
```

### Robust Error Handling

```python
def analyze_exercise_metrics_safe(metrics_file="exercise_metrics.txt"):
    try:
        # Read file
        with open(metrics_file, "r") as f:
            exercise_data = f.read()
    except FileNotFoundError:
        return "No metrics found. Complete a workout first."
    except IOError as e:
        return f"Error reading file: {e}"
    
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=3,
            timeout=30
        )
        
        # Invoke LLM
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        # Fallback response
        return f"""Analysis temporarily unavailable.
        
Based on your metrics:
- Total reps: {extract_reps(exercise_data)}
- Keep practicing and maintain good form!
- Consult a fitness professional for detailed feedback.
        
Error: {str(e)}"""
```

---

## Summary

`ai.py` is an intelligent fitness coaching module that:

✅ **LLM Integration**: Google Gemini 2.5 Flash for analysis
✅ **LangChain Framework**: Robust message handling and retries
✅ **Prompt Engineering**: Optimized system/user prompts
✅ **Personalized Analysis**: Data-driven performance feedback
✅ **Sport Recommendations**: Suggests suitable sports based on metrics
✅ **Secure Configuration**: Environment variable management
✅ **Error Handling**: Graceful fallbacks for missing data
✅ **Fast Inference**: Flash model for quick responses
✅ **Deterministic Output**: Temperature=0 for consistency
✅ **Production-Ready**: Retry logic and timeout handling

**Key Innovation**: Combines quantitative exercise metrics from computer vision (test.py) with qualitative coaching insights from large language models, providing a complete AI-powered fitness analysis system.

**Architecture Advantage**: 
- **Frontend**: Real-time pose detection (YOLO11)
- **Backend**: Biomechanical analysis (metrics.py)
- **Intelligence**: Natural language coaching (ai.py + Gemini)
- **Result**: Comprehensive, personalized fitness guidance

**Use Case**: Perfect for fitness apps, personal training platforms, or exercise tracking systems that want to provide intelligent, conversational coaching alongside technical analysis.
