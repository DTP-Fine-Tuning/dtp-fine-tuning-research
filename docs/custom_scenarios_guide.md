# Custom Evaluation Scenarios Guide

This guide explains how to create custom test scenarios for evaluating your conversational AI model using DeepEval.

## Overview

The evaluation script (`src/eval/deepeval_my_model.py`) supports custom scenario files in JSON format. This allows you to test your model with specific conversation patterns relevant to your use case.

## Scenario File Format

### Structure

A scenario file is a JSON array containing one or more test scenarios:

```json
[
    {
        "name": "Scenario Name",
        "messages": [
            "First user message",
            "Second user message",
            ...
        ]
    },
    ...
]
```

### Fields

Each scenario object must have:

- **`name`** (string, required): A descriptive name for the test case
  - Used for identifying the scenario in evaluation results
  - Example: `"Junior Web Developer"`, `"Data Analyst"`

- **`messages`** (array, required): List of user messages in conversation order
  - Must contain at least one message
  - The model will generate responses after each message
  - Messages should represent realistic user inputs for your use case

### Example Scenario

```json
[
    {
        "name": "Software Engineer Interview",
        "messages": [
            "Berikut data saya: Posisi Software Engineer, 3 tahun pengalaman dengan Python dan Java.",
            "Saya mengerjakan proyek backend API menggunakan FastAPI dan PostgreSQL."
        ]
    },
    {
        "name": "Data Scientist Interview",
        "messages": [
            "Berikut data saya: Lulusan S1 Statistika, magang di bidang data analytics.",
            "Saya familiar dengan Python libraries seperti Pandas, NumPy, dan Scikit-learn.",
            "Dalam tugas akhir, saya membuat model prediksi menggunakan random forest."
        ]
    }
]
```

## Using Custom Scenarios

### Method 1: Via Command Line

```bash
# Run evaluation with custom scenarios
./scripts/run_evaluation.sh --scenario-file my_scenarios.json

# Run complete pipeline with custom scenarios
./scripts/run_pipeline.sh pipeline --scenario-file my_scenarios.json
```

### Method 2: Via Environment Variable

```bash
export EVAL_SCENARIO_FILE="path/to/scenarios.json"
python src/eval/deepeval_my_model.py
```

### Method 3: Default Scenarios

If no custom file is provided, the script uses built-in default scenarios:
- Junior Web Developer
- Data Analyst  
- Security Analyst

## Example Scenario File

An example scenario file is provided: `example_scenarios.json`

This file contains 5 diverse test scenarios:
1. Junior Web Developer
2. Data Analyst
3. DevOps Engineer
4. Mobile Developer
5. UI/UX Designer

To use it:

```bash
./scripts/run_evaluation.sh --scenario-file example_scenarios.json
```

## Best Practices

### 1. Realistic User Messages

Create messages that represent actual user inputs:

**Good:**
```json
"Berikut data saya: Posisi Backend Developer dengan 2 tahun pengalaman. Saya menggunakan Node.js dan MongoDB dalam pekerjaan sehari-hari."
```

**Avoid:**
```json
"Test message 123"
```

### 2. Progressive Conversation

Structure messages to build on each other:

```json
{
    "name": "Interview Flow",
    "messages": [
        "Berikut data awal saya...",
        "Saya ingin menjelaskan lebih detail tentang pengalaman saya...",
        "Tantangan yang saya hadapi adalah..."
    ]
}
```

### 3. Scenario Diversity

Include various conversation types:
- Different job roles
- Different experience levels
- Different conversation lengths (2-5 messages recommended)

### 4. Language Consistency

For Indonesian interview bot:
- Use proper Indonesian (formal/semi-formal)
- Include domain-specific terminology
- Maintain professional tone

### 5. Test Coverage

Create scenarios that test:
- Technical knowledge discussion
- Problem-solving explanations
- Project descriptions
- Challenge handling

## Controlling Scenario Count

Limit the number of scenarios to evaluate:

```bash
# Evaluate only first 2 scenarios
./scripts/run_evaluation.sh --scenario-file scenarios.json --scenarios 2

# Via environment variable
export EVAL_NUM_SCENARIOS=2
```

## Evaluation Metrics

Each scenario will be evaluated on:

**Conversational Metrics (5):**
1. Turn Relevancy - Does each response address the user's message?
2. Knowledge Retention - Does the model remember context across turns?
3. Role Adherence - Does it maintain the interviewer persona?
4. Conversation Completeness - Does it gather all needed information?
5. Topic Adherence - Does it stay focused on relevant topics?

**Safety Metrics (3):**
6. Toxicity - Is the language appropriate?
7. Bias - Is the model fair and unbiased?
8. Hallucination - Does it make up information?

## Troubleshooting

### "Scenario file not found"

- Check the file path is correct (absolute or relative to project root)
- Ensure the file has `.json` extension

### "Invalid JSON in scenario file"

- Validate your JSON using a JSON validator
- Check for missing commas, quotes, or brackets
- Ensure proper UTF-8 encoding

### "Invalid scenario format"

- Verify each scenario has `name` and `messages` fields
- Ensure `messages` is an array with at least one string
- Check for typos in field names (case-sensitive)

### Fallback Behavior

If scenario loading fails, the script automatically falls back to default scenarios and continues evaluation.

## Advanced Usage

### Testing Specific Scenarios Only

Create focused test files for specific aspects:

**edge_cases.json:**
```json
[
    {
        "name": "Very Short Response",
        "messages": ["Halo"]
    },
    {
        "name": "No Experience",
        "messages": ["Saya fresh graduate tanpa pengalaman kerja"]
    }
]
```

### A/B Testing Different Prompts

Compare model behavior with different prompt styles:

**formal_scenarios.json** vs **casual_scenarios.json**

### Batch Evaluation

Evaluate multiple scenario files:

```bash
for file in scenarios/*.json; do
    echo "Evaluating: $file"
    ./scripts/run_evaluation.sh --scenario-file "$file" --scenarios 5
done
```

## Related Files

- `src/eval/deepeval_my_model.py` - Main evaluation script
- `scripts/run_evaluation.sh` - Evaluation runner script
- `example_scenarios.json` - Example scenario file
- `scripts/run_pipeline.sh` - Complete pipeline with scenario support

## Questions or Issues?

If you encounter issues or have questions about creating custom scenarios, please check:
1. This documentation
2. The example scenario file
3. The inline documentation in `src/eval/deepeval_my_model.py`
