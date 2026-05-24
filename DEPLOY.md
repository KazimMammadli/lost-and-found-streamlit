# Streamlit Cloud Deployment Guide

Step-by-step instructions to deploy **Smart Lost & Found** to
`https://smart-lost-and-found-avaz-kazim-gulnar.streamlit.app/` using a
brand-new GitHub repository.

---

## 0 — What's included in this deployment

| | |
|---|---|
| **Provider** | OpenAI only (no failover) |
| **LLM** | `gpt-4o` (vision) |
| **Embedder** | `text-embedding-3-small` |
| **Database** | SQLite (pre-seeded, ephemeral on restart) |
| **Python** | 3.12 |
| **Entry point** | `ui/app.py` |
| **Sample data** | 8 lost items + 7 found items already in `dev.db` |

---

## 1 — Files you must commit to the deploy repo

```
SmartLostFound-Deploy/
├── ui/
│   └── app.py
├── src/                          ← entire folder
├── ai/                           ← entire folder (openai provider only used)
├── data/
│   ├── images/                   ← MUST include — referenced by dev.db
│   └── samples/                  ← optional but recommended
├── .streamlit/
│   ├── config.toml               ← server + theme config
│   └── secrets.toml.example      ← template for Streamlit Cloud secrets
├── dev.db                        ← pre-seeded SQLite (15 items)
├── requirements.txt              ← rename from requirements-deploy.txt
├── runtime.txt                   ← pins python-3.12
├── .python-version               ← also pins 3.12 (belt-and-braces)
├── .gitignore                    ← rename from .gitignore.deploy
└── README.md                     ← (optional) demo description
```

### Files you must **NOT** commit

- `.env` (contains your real API keys)
- `.streamlit/secrets.toml` (real secrets — set in Streamlit Cloud UI)
- `__pycache__/`, `.venv/`, `.pytest_cache/` etc. (covered by .gitignore)
- `artefacts/cost.jsonl` (writable at runtime)

---

## 2 — One-time setup steps

### A. Prepare the deploy folder locally

From the project root, in PowerShell:

```powershell
# 1. Create a new sibling folder for the deploy repo
$src = "C:\Users\avaza\Desktop\AI Academy Final"
$dst = "C:\Users\avaza\Desktop\SmartLostFound-Deploy"
New-Item -ItemType Directory -Force $dst | Out-Null

# 2. Copy the required files
Copy-Item -Recurse "$src\ui"  "$dst\ui"
Copy-Item -Recurse "$src\src" "$dst\src"
Copy-Item -Recurse "$src\ai"  "$dst\ai"
Copy-Item -Recurse "$src\.streamlit" "$dst\.streamlit"

# 3. Copy data folder WITHOUT the optional samples-source script
New-Item -ItemType Directory -Force "$dst\data" | Out-Null
Copy-Item -Recurse "$src\data\images"  "$dst\data\images"
Copy-Item -Recurse "$src\data\samples" "$dst\data\samples"

# 4. Copy the pre-seeded database
Copy-Item "$src\dev.db" "$dst\dev.db"

# 5. Copy the deploy-specific config files (renamed)
Copy-Item "$src\requirements-deploy.txt" "$dst\requirements.txt"
Copy-Item "$src\.gitignore.deploy"       "$dst\.gitignore"
Copy-Item "$src\runtime.txt"             "$dst\runtime.txt"
Copy-Item "$src\.python-version"         "$dst\.python-version"
Copy-Item "$src\README.md"               "$dst\README.md"
Copy-Item "$src\DEPLOY.md"               "$dst\DEPLOY.md"

# 6. (Optional) Remove __pycache__ folders that may have been copied
Get-ChildItem $dst -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

### B. Initialise git and push to a new GitHub repo

```powershell
cd C:\Users\avaza\Desktop\SmartLostFound-Deploy
git init
git add .
git status      # double-check no .env or secrets file is staged
git commit -m "Initial deploy: pre-seeded Smart Lost & Found"
git branch -M main
# Create an EMPTY repo on github.com first (no README, no .gitignore)
# Then attach the remote — replace YOUR_USERNAME / YOUR_REPO_NAME
git remote add origin https://github.com/YOUR_USERNAME/smart-lost-and-found.git
git push -u origin main
```

---

## 3 — Streamlit Cloud configuration

1. Go to https://share.streamlit.io and click **New app**.
2. Pick your new GitHub repo, branch `main`, main file `ui/app.py`.
3. Click **Advanced settings** and set:
   - **Python version**: `3.12`
   - **Custom subdomain**: `smart-lost-and-found-avaz-kazim-gulnar`
4. Open **Secrets** and paste this exact content (replace the API key
   with your real one):

   ```toml
   LLM_PROVIDER = "openai"
   LLM_MODEL = "gpt-4o"
   OPENAI_API_KEY = "sk-REPLACE-WITH-YOUR-REAL-OPENAI-KEY"

   EMBEDDING_PROVIDER = "openai"
   EMBEDDING_MODEL = "text-embedding-3-small"

   DATABASE_URL = "sqlite+aiosqlite:///./dev.db"

   IMAGES_DIR = "data/images"
   MAX_IMAGE_BYTES = 5242880

   SEMAPHORE_LIMIT = 10
   AI_CALL_TIMEOUT_S = 30.0
   AI_CALL_TPM_LIMIT = 40000

   LOG_LEVEL = "INFO"
   ENABLE_TRACING = false
   OTLP_ENDPOINT = "http://localhost:4317"

   COST_LOG_PATH = "artefacts/cost.jsonl"
   ```

5. Click **Save**, then **Deploy**.

---

## 4 — Verification checklist

After the app finishes building:

- [ ] App loads at `https://smart-lost-and-found-avaz-kazim-gulnar.streamlit.app/`
- [ ] Sidebar shows `Provider: openai / gpt-4o`
- [ ] Sidebar shows `Embedder: openai / text-embedding-3-small`
- [ ] **Home** page shows total = 15, lost = 8, found = 7
- [ ] **Browse Items** page shows a grid of 15 thumbnails
- [ ] **Search Matches** with item ID `2` returns a found backpack
  as the top match
- [ ] **Register Lost Item** uploading a new image works end-to-end
  (the new item is visible on Browse, with a generated AI description)

If anything fails, see **Troubleshooting** below.

---

## 5 — Troubleshooting

### "Registration failed: GOOGLE_API_KEY (or LLM_API_KEY) is not set."
This means `SECONDARY_LLM_PROVIDER` is set to something other than empty
in Streamlit secrets. Delete that key entirely from your secrets, or
explicitly set:
```toml
SECONDARY_LLM_PROVIDER = ""
```
Save → Reboot app.

### "ModuleNotFoundError: openai"
You committed the wrong `requirements.txt`. Make sure the deployed file
contains `openai==1.30.5` (this is in `requirements-deploy.txt` in the
source repo).

### Browse page is empty / no items shown
The deploy repo is missing `dev.db`. Verify with:
```powershell
git ls-files | findstr dev.db
```
If empty, your `.gitignore` is excluding it. Use the `.gitignore` from
`.gitignore.deploy` (which intentionally does NOT exclude `dev.db`).

### Search returns no matches even though items exist
The `data/images/` folder is missing in the deploy repo, OR the
`embedding` column is empty in `dev.db`. Re-copy `dev.db` and
`data/images/` from the source repo.

### Slow cold start (> 60 seconds)
First boot on Streamlit Cloud installs all wheels. The slim
`requirements-deploy.txt` (no FastAPI, no Anthropic SDK, no Gemini SDK)
keeps this under ~45 seconds in practice.

### Database wipes after restart
This is expected behaviour for SQLite on Streamlit Cloud — the container
filesystem is ephemeral. The deploy ships with `dev.db` committed to git,
so every cold start restores the same 15 sample items. User uploads made
during a session do not survive container restarts. For persistent
storage upgrade to PostgreSQL (configure `DATABASE_URL`).

---

## 6 — Rotating your OpenAI key

If your key was ever shared (in chat, screenshots, screen-share):

1. Go to https://platform.openai.com/api-keys
2. Find the old key and click **Revoke**
3. Create a new key
4. Update **Streamlit Cloud → Settings → Secrets → `OPENAI_API_KEY`**
5. Update your local `.env` (do NOT commit it)
6. Click **Reboot app** in Streamlit Cloud

---

## 7 — What changed vs. the dev project?

| Change | Why |
|---|---|
| `SECONDARY_LLM_PROVIDER` default → `None` | OpenAI-only deploy, no Gemini key needed |
| `.streamlit/config.toml` added | Theme, upload limit, headless mode |
| `.streamlit/secrets.toml.example` added | Onboarding template for secrets |
| `runtime.txt` added | Pins Python 3.12 on Streamlit Cloud |
| `.python-version` added | Belt-and-braces Python pin |
| `requirements-deploy.txt` added | Slim deps (OpenAI + Streamlit only) |
| `.gitignore.deploy` added | Allows `dev.db` and `data/images/` in deploy repo |
| `ui/app.py` — Home page, thumbnails, image previews in Search | Better demo presentation |
| `ui/app.py` — path resolution via `_image_path()` | Robust on Streamlit Cloud regardless of cwd |
