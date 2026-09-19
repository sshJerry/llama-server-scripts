# ornith-agentworld-tandem

Two 35B MoE models, one loop. Ornith acts — AgentWorld simulates what happens next. Just two GGUF models running side by side on a pair of GPUs.

## Setup

Both servers need to be running:

```
./run-Ornith+AgentWorld-tandem.sh
```

That starts:
- **Ornith** on `:8080` — the agent that decides what to do
- **AgentWorld** on `:8081` — the world model that predicts what the environment returns

Clone the official domain prompts:

```
git clone https://github.com/QwenLM/Qwen-AgentWorld.git
cp -r Qwen-AgentWorld/prompts ./prompts
```

## Usage

```
python3 agent_world_loop.py <domain>
```

Seven domains, pick one:

| command | simulates |
|---------|-----------|
| `terminal` | Linux shell |
| `swe` | software engineering / code editing |
| `web` | browser interactions |
| `os` | desktop automation (LibreOffice, VLC, etc.) |
| `android` | Android device UI |
| `mcp` | tool calling |
| `search` | search engine results |

Then type a task and watch the loop go.

## How it works

```
You: "Fix the off-by-one error in parser.c"
  ↓
Ornith:  git diff HEAD~1 src/parser.c
  ↓
AgentWorld:  diff --git a/src/parser.c b/src/parser.c ...
  ↓
Ornith:  sed -i 's/<= i</< i</' src/parser.c
  ↓
AgentWorld:  (file changed)
  ↓
Ornith:  DONE
```

No pip dependencies. Stdlib only.

## Example tasks

Stuff to throw at it when you want to see what the loop can do.

### terminal

```
Set up a Python project from scratch: create a venv, install flask and sqlite3, 
scaffold a REST API with three endpoints (GET /items, POST /items, DELETE /items/<id>), 
write a quick curl test to make sure it works, then benchmark it with ab -n 1000 -c 10.
```

### swe

```
This repo has a memory leak in the session handler. grep for malloc without free in 
src/session.c, trace every allocation path through the login and logout codepaths, 
propose the fix, apply it, and verify with valgrind. The leak triggers after 10+ 
login/logout cycles.
```

### web

```
Go to the admin dashboard at /admin, log in with admin/admin123, navigate to 
Settings > API Keys, revoke the key labeled "old-staging-key", generate a new one 
with read+write scope named "prod-deploy-2026", copy the key value, then go to the 
Deployments tab and update the WEBHOOK_SECRET environment variable with the new key.
```

### os

```
Open the budget spreadsheet on the Desktop named Q2_forecast.ods, fix the SUM formula 
in cell D14 that accidentally excludes row 13, add conditional formatting to highlight 
any expense line over $5,000 in red, create a pie chart from the category breakdown 
in columns A and D, export the sheet as PDF, and save it to Documents/reports/.
```

### android

```
Open Settings > Storage, check how much space is used by the Downloads folder, clear 
the cache for Chrome and YouTube, uninstall the app called "Flashlight Pro" which 
has been crashing, then open the Play Store and install "OpenTracks" from the 
F-Droid repository.
```

### mcp

```
You have access to filesystem, git, and weather tools. Clone the repo 
github.com/example/sensor-dashboard, read the config in config/deploy.yaml to find 
which city the weather widget targets, fetch the current weather for that city, 
update the dashboard's index.html to show the live temperature in the header, 
commit the change, and push to a new branch called weather-update.
```

### search

```
Research the top 5 vector databases for on-premise deployment: compare Pinecone, 
Weaviate, Qdrant, Milvus, and ChromaDB across dimensions of query latency at 1M 
vectors, RAM usage, filtering support, and Python client maturity. For each, find 
the latest benchmark numbers from their docs or blog posts, summarize the results 
in a markdown table, then rank them by best fit for a mid-size e-commerce search 
use case.
```
