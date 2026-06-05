# Shared Docker Compose settings — source from run_one.sh / run_all.sh only.
# Isolates this project from other Compose stacks (e.g. etteplan-smart-agent).
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-memory-agent-benchmark}"
export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

dc() {
  docker compose -p "$COMPOSE_PROJECT_NAME" -f "$COMPOSE_FILE" "$@"
}
