#!/usr/bin/env bash
# geno-voice Le Mans lap runner.
#
#   run-lap.sh              one real lap of the geno-voice improvement loop
#   run-lap.sh --dry-run    print the resolved prompt + paths, launch nothing
#
# One invocation = one lap. Cron runs it on a schedule. The loop improves the
# geno-voice local voice pipeline (STT/TTS/VAD, streaming, CLI) one verified,
# committed improvement per lap, gated by pytest. Restarted 2026-06-16 (the
# original session-only cron 0c6630bd expired); repoints the canonical paper-club/
# geno-loops-v2 runner at geno-voice. Last work before restart: iter-136 on main.
#
# All state is files (ITERATION_LOG.md + .loops/geno-voice/laps) so progress is
# observable over SSH with no TUI. Emails one progress report per lap to operator.
set -uo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

ROOT="$HOME/code-purp/geno-voice"
LOOPDIR="$ROOT/.loops/geno-voice"
LAPDIR="$LOOPDIR/laps"
LOCK="$LOOPDIR/lap.lock"
PGIDF="$LOOPDIR/lap.pgid"
LOG_FILE="$ROOT/ITERATION_LOG.md"
WORKTREES="$HOME/code-purp/geno-voice-worktrees"
STEER="$LOOPDIR/STEER.md"
EMAIL_TO="eriveraramos@blueorigin.com"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$LAPDIR"
cd "$ROOT" || { echo "no root $ROOT"; exit 1; }

CLAUDE_BIN="${CLAUDE_BIN:-}"
[ -x "$CLAUDE_BIN" ] || CLAUDE_BIN=$(command -v claude 2>/dev/null || true)
# shellcheck disable=SC2012
[ -n "$CLAUDE_BIN" ] || CLAUDE_BIN=$(ls -t "$HOME"/.nvm/versions/node/*/bin/claude 2>/dev/null | head -1 || true)
if [ -z "$CLAUDE_BIN" ] || [ ! -x "$CLAUDE_BIN" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] WARNING: claude not found on PATH=$PATH"
  else
    echo "$(date) FATAL: claude not found (PATH=$PATH)" >> "$LAPDIR/laps.log"; exit 127
  fi
fi
if [ -n "$CLAUDE_BIN" ]; then export PATH="$(dirname "$CLAUDE_BIN"):$PATH"; fi

if [ "$DRY_RUN" = "0" ]; then
  if [ -e "$LOCK" ]; then
    pid=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$(date) [geno-voice] lap running (pid $pid); skip" >> "$LAPDIR/skips.log"; exit 0
    fi
    echo "$(date) [geno-voice] stale lock; reclaim" >> "$LAPDIR/skips.log"
  fi
  echo $$ > "$LOCK"
  trap 'rm -f "$LOCK"' EXIT
fi

NEXT=1
if [ -f "$LOG_FILE" ]; then
  last=$(grep -oE 'iter-([0-9]+)' "$LOG_FILE" | grep -oE '[0-9]+' | sort -n | tail -1)
  [ -n "$last" ] && NEXT=$((10#$last + 1))
fi
printf -v NNN '%03d' "$NEXT"

STEER_BLOCK=""
if [ -f "$STEER" ]; then
  STEER_BLOCK="OPERATOR STEERING for THIS lap (overrides default item selection):
$(cat "$STEER")
After completing the steered work, delete this STEER.md ($STEER) so it is not repeated.
"
fi

PROMPT=$(cat <<EOF
Iterate on geno-voice. This is ONE lap of the geno-voice Le Mans improvement loop,
iteration $NNN. Headless, non-interactive — do all work with tools, never ask
questions, finish the lap, then stop. Repo root: $ROOT (a real git repo YOU own).

FIRST read $LOG_FILE (ITERATION_LOG.md) for context and the running bug/improvement
list, and any GENO.md/CLAUDE.md conventions in the repo.

$STEER_BLOCK
Protocol for THIS lap only:
1. Pick ONE bug or improvement (or do OPERATOR STEERING if present) — a single
   logical increment. Good sources: ITERATION_LOG next directions, STT/TTS/VAD
   quality, streaming/barge-in, the gv CLI, tests, docs.
2. Implement it on a worktree under $WORKTREES (git worktree add a branch off main).
   Write tests (unit and/or integration — tests/integration/ exists since iter-035).
3. Verify with: python -m pytest tests/unit/   (add integration tests where apt).
   Only merge to main if tests pass.
4. ff-merge the branch to main with a clean tree and a descriptive commit. Keep
   everything LOCAL — NO git push.
5. Append an iter- section to $LOG_FILE: date, branch, commit, what changed,
   why, exact pytest result, next planned item. Self-contained (source for the email).
6. Clean up the worktree when done (git worktree remove).

If tests cannot pass: revert/stash, do NOT merge red, write the reason to
$LAPDIR/$STAMP.fail, still append an honest iter- no-op log section, then stop.
Reminder: never run tmux capture-pane/list-sessions/display-message on z2.
EOF
)

if [ "$DRY_RUN" = "1" ]; then
  echo "=== geno-voice run-lap.sh --dry-run ==="
  echo "ROOT=$ROOT  LOG=$LOG_FILE  WORKTREES=$WORKTREES  next=iter-$NNN"
  echo "CLAUDE_BIN=${CLAUDE_BIN:-<unresolved>}"
  echo "--- resolved lap prompt ---"; echo "$PROMPT"; echo "--- end ---"
  exit 0
fi

echo "$(date) [geno-voice] starting iter-$NNN" >> "$LAPDIR/laps.log"
setsid "$CLAUDE_BIN" -p "$PROMPT" --dangerously-skip-permissions   --strict-mcp-config --mcp-config '{"mcpServers":{}}'   > "$LAPDIR/$STAMP.iter-$NNN.out" 2>&1 &
CPID=$!
CPGID=$(ps -o pgid= -p "$CPID" 2>/dev/null | tr -d ' ')
echo "${CPGID:-$CPID}" > "$PGIDF"
wait "$CPID"; code=$?
rm -f "$PGIDF"
echo "$(date) [geno-voice] iter-$NNN done (exit $code)" >> "$LAPDIR/laps.log"

ITER_SECTION=$(awk -v n="iter-$NNN" 'index($0,n){f=1} f && /iter-/ && index($0,n)==0 {exit} f{print}' "$LOG_FILE" 2>/dev/null)
[ -z "$ITER_SECTION" ] && ITER_SECTION="(no log section for geno-voice iter-$NNN; exit $code. See $LAPDIR/$STAMP.iter-$NNN.out)"

EMAIL_PROMPT="Send exactly one email via the blue_email MCP and do nothing else. This is a LOOP PROGRESS report to the operator about the geno-voice improvement loop on z2. Do not ask questions.
To: $EMAIL_TO.
Subject: [geno-voice] iter-$NNN ($(date +%Y-%m-%d\ %H:%M) PT) exit=$code
Body: a clean HTML rendering of the lap report below. Do not invent status.

--- geno-voice iter-$NNN report ---
$ITER_SECTION
--- end ---"

setsid "$CLAUDE_BIN" -p "$EMAIL_PROMPT" --dangerously-skip-permissions   --strict-mcp-config --mcp-config "$HOME/.claude/.mcp-email-only.json"   > "$LAPDIR/$STAMP.iter-$NNN.email.out" 2>&1   || echo "$(date) [geno-voice] iter-$NNN email failed" >> "$LAPDIR/laps.log"

exit $code
