# Sparse Autoencoder Steering Benchmark

A benchmarking tool for evaluating Goodfire model variants with feature steering.

## Setup

### 1. Install Dependencies and Activate Virtual Environment
```bash
uv sync
source .venv/bin/activate
```

### 2. Set Your Goodfire API Key

You have several options to provide your Goodfire API key:

#### Option A: Create a .env file (Recommended)
Create a `.env` file in the root directory:
```bash
echo "GOODFIRE_API_KEY=your_actual_api_key_here" > .env
```

Or manually create a `.env` file with the following content:
```
GOODFIRE_API_KEY=your_actual_api_key_here
```

#### Option B: Set Environment Variable
**Windows PowerShell:**
```powershell
$env:GOODFIRE_API_KEY="your_actual_api_key_here"
```

**Windows Command Prompt:**
```cmd
set GOODFIRE_API_KEY=your_actual_api_key_here
```

**Linux/Mac:**
```bash
export GOODFIRE_API_KEY="your_actual_api_key_here"
```

#### Option C: Inline with Command
**Linux/Mac:**
```bash
GOODFIRE_API_KEY="your_actual_api_key_here" python bench.py --model 70b --prompt "test"
```

**Windows PowerShell:**
```powershell
$env:GOODFIRE_API_KEY="your_actual_api_key_here"; python bench.py --model 70b --prompt "test"
```

## Usage

### Basic Usage
```bash
uv run bench.py --model 70b --prompt "mathematical reasoning" --task gsm8k --limit 10
```

### Available Arguments
- `--model`: Model to use ("70b" or "8b") [default: "70b"]
- `--prompt`: Prompt to search for autointerp feature labels [default: "mathematical reasoning and problem solving"]
- `--task`: Benchmark to run ("truthfulqa_gen" or "gsm8k") [default: "gsm8k"]
- `--topk`: Number of features to use [default: 5]
- `--strength`: Feature strength [default: 0.3]
- `--limit`: Number of samples to run [default: 5]
- `--baseline`: Use baseline model comparison [default: False]
- `--temp`: Temperature [default: 0.0]
- `--raw_results`: Print raw results [default: False]
- `--name`: Custom name for the log file (without .txt extension) [default: None]

### Supported Benchmarks
Currently, only generation benchmarks are supported:
- **`gsm8k`**: Grade School Math 8K - Mathematical reasoning problems
- **`truthfulqa_gen`**: TruthfulQA Generation - Truthfulness evaluation

Other benchmark types (multiple choice, classification, etc.) are not yet supported.

### Examples

**Basic benchmark:**
```bash
uv run bench.py --model 70b --prompt "mathematical reasoning" --task gsm8k --limit 20
```

**With baseline comparison:**
```bash
uv run bench.py --model 70b --prompt "truthfulness" --task truthfulqa_gen --baseline True --limit 50
```

**Custom configuration:**
```bash
uv run bench.py --model 8b --prompt "logical thinking" --task gsm8k --topk 10 --strength 0.5 --limit 100 --name my_experiment
```

## Output

Results are automatically saved to the `results/` directory with timestamps. You can specify a custom name using the `--name` argument.

The tool will display:
- Configuration details
- Controller settings for steered (and optionally baseline) variants
- Formatted evaluation results
- Raw benchmark results for steered (and optionally baseline) variants

## Troubleshooting

If you get an error about `GOODFIRE_API_KEY is not set`, make sure you've followed one of the setup methods above and that your API key is valid. 