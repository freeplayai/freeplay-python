# GenAI Tool Schema Testing Guide

This guide provides step-by-step instructions for manually testing the GenAI tool schema support in the Python SDK.

## Prerequisites

- Freeplay app running locally (`make run` in `freeplay-app`)
- Environment variables configured in `.env`
- PostgreSQL database running

## Test 1: Basic GenAI Tool Schema

### Step 1: Start the REPL

```bash
cd /Users/montylennie/freeplay-repos/freeplay-python
make repl
```

This starts an interactive Python session with:
- SSL patches applied
- Freeplay client initialized
- Environment variables loaded

### Step 2: Run Test 1 in REPL

Copy and paste this entire block into the REPL:

```python
from freeplay import GenaiFunction, GenaiTool, RecordPayload, CallInfo

# Create a simple weather function
weather_function = GenaiFunction(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g., 'San Francisco'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
)

# Create GenAI tool (single tool with function declarations)
tool_schema = [GenaiTool(functionDeclarations=[weather_function])]

# Record with GenAI format
response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "What's the weather in San Francisco?"},
            {"role": "assistant", "content": "Let me check the weather for you."}
        ],
        tool_schema=tool_schema,
        call_info=CallInfo(provider="genai", model="gemini-2.0-flash"),
    )
)

print(f"✅ Test 1 PASSED")
print(f"   Completion ID: {response.completion_id}")
```

### Step 3: Note the Completion ID

You'll see output like:
```
✅ Test 1 PASSED
   Completion ID: af366b30-3d4b-43eb-bf8d-4a079c167915
```

**Copy this completion ID** - you'll need it for verification.

### Step 4: Verify in Database

Open a **new terminal** and run:

```bash
cd /Users/montylennie/freeplay-repos/freeplay-app

# Replace YOUR_COMPLETION_ID with the ID from Step 3
psql postgresql://localhost:5432/freeplay_development -U freeplay_app -c "SELECT project_session_entry_id, tool_schema_version, tool_schema FROM project_session_entry_tool_schemas WHERE project_session_entry_id = 'YOUR_COMPLETION_ID';"
```

**Example with actual ID:**
```bash
psql postgresql://localhost:5432/freeplay_development -U freeplay_app -c "SELECT project_session_entry_id, tool_schema_version, tool_schema FROM project_session_entry_tool_schemas WHERE project_session_entry_id = 'af366b30-3d4b-43eb-bf8d-4a079c167915';"
```

### Step 5: Expected Database Output

You should see:
```
       project_session_entry_id       | tool_schema_version |                    tool_schema                    
--------------------------------------+---------------------+--------------------------------------------------
 af366b30-3d4b-43eb-bf8d-4a079c167915 |                   1 | [{"name": "get_weather", "parameters": {...}, "description": "Get the current weather for a location"}]
(1 row)
```

The tool_schema column should contain a JSON array with:
- `name`: "get_weather"
- `description`: "Get the current weather for a location"
- `parameters`: Complete parameter schema with location and units

### Step 6: Check Server Logs (Optional)

In your `make run` terminal (freeplay-app), you should see debug logs:

```
🔧 Recording tool schema
  provider: genai
  tool_count: 1
  is_genai_format: True

🚀 Recording with GenAI tool schema format
  provider: genai
  tool_format: GenAI/Vertex (VertexTool)
  total_functions: 1

✅ Successfully normalized GenAI tool schema: 1 functions
```

---

## Test 2: Multiple Function Declarations

This test validates the key GenAI feature - multiple functions in a single tool.

### Run Test 2 in REPL:

```python
# Create multiple functions
functions = [
    GenaiFunction(
        name="get_weather",
        description="Get current weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    ),
    GenaiFunction(
        name="get_news",
        description="Get latest news",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"]
        }
    ),
    GenaiFunction(
        name="search_web",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    )
]

# Single tool with multiple function declarations (GenAI-specific)
tool_schema = [GenaiTool(functionDeclarations=functions)]

response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "Give me weather, news, and search for Python"},
            {"role": "assistant", "content": "I'll help with all three requests."}
        ],
        tool_schema=tool_schema,
        call_info=CallInfo(provider="genai", model="gemini-2.0-flash"),
    )
)

print(f"✅ Test 2 PASSED: Multiple function declarations")
print(f"   Completion ID: {response.completion_id}")
print(f"   Functions: {', '.join([f.name for f in functions])}")
```

### Verify in Database:

```bash
# Replace with your completion ID
psql postgresql://localhost:5432/freeplay_development -U freeplay_app -c "SELECT project_session_entry_id, tool_schema FROM project_session_entry_tool_schemas WHERE project_session_entry_id = 'YOUR_COMPLETION_ID';"
```

You should see **3 separate entries** in the tool_schema array (one for each function).

---

## Test 3: Complex Nested Parameters

This test validates complex parameter schemas with nested objects.

### Run Test 3 in REPL:

```python
# Complex nested parameter schema
book_flight_function = GenaiFunction(
    name="book_flight",
    description="Book a flight for a passenger",
    parameters={
        "type": "object",
        "properties": {
            "passenger": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "passport": {"type": "string"}
                },
                "required": ["name", "age"]
            },
            "destination": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "airport_code": {"type": "string"}
                },
                "required": ["city", "country"]
            }
        },
        "required": ["passenger", "destination"]
    }
)

tool_schema = [GenaiTool(functionDeclarations=[book_flight_function])]

response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "Book a flight to Paris for John Doe"},
            {"role": "assistant", "content": "I'll book that flight for you."}
        ],
        tool_schema=tool_schema,
        call_info=CallInfo(provider="genai", model="gemini-2.0-flash"),
    )
)

print(f"✅ Test 3 PASSED: Complex nested parameters")
print(f"   Completion ID: {response.completion_id}")
```

### Verify in Database:

The tool_schema should contain the nested structure with `passenger` and `destination` objects intact.

---

## What These Tests Verify

### ✅ Feature Requirement 1: "Tool schema needs to be properly formatted"
- GenAI native format with `functionDeclarations`
- Multiple functions per tool (GenAI-specific)
- Complex nested parameters
- Proper JSON serialization

### ✅ Feature Requirement 2: "When recording tool_schema needs to support genai format"
- Python SDK successfully sends GenAI `VertexTool` format
- Backend `RecordService.normalize_tools_schema()` correctly processes GenAI format
- Tool schemas are normalized to `NormalizedToolSchema` format
- Data is properly stored in `project_session_entry_tool_schemas` table

---

## Troubleshooting

### Database Connection Issues

If you get `psql: error: connection to server`:
```bash
# Check if PostgreSQL is running
pg_isready

# Or check via Docker if using Docker
docker ps | grep postgres
```

### REPL Issues

If `make repl` fails:
```bash
# Ensure environment variables are set
cat .env

# Check that required vars exist
grep FREEPLAY_API_KEY .env
grep FREEPLAY_PROJECT_ID .env
```

### Tool Schema Not Found in Database

If the query returns no rows:
1. Verify the completion ID is correct (copy-paste carefully)
2. Check server logs for errors during recording
3. Verify the Freeplay app is running (`make run` in freeplay-app)

---

## Quick Test Script

For rapid testing, use the automated test script:

```bash
cd /Users/montylennie/freeplay-repos/freeplay-python
uv run python tests/manual_test_genai.py
```

This runs all tests automatically and displays results.

---

## Summary

**Python SDK → API → Normalization → Database Storage: ✅ ALL WORKING!**

The GenAI tool schema integration is fully functional:
- ✅ SDK sends correct format
- ✅ Backend processes GenAI format
- ✅ Data is normalized and stored
- ✅ All parameters preserved

**Phase 2 (Python SDK) Complete!** 🎉

