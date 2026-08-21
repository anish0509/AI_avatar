COMPOSE = docker compose -p ai-avatar-local -f avatar-poc/docker-compose.prod.yml -f avatar-poc/docker-compose.local.yml

.PHONY: deploy status logs stop

deploy: avatar-poc/.env
	$(COMPOSE) config --quiet
	$(COMPOSE) up -d --build --wait --wait-timeout 120
	@echo "Avatar is available at http://localhost:8087"

avatar-poc/.env:
	cp avatar-poc/.env.example avatar-poc/.env
	@echo "Created avatar-poc/.env; add API keys to enable voice and avatar requests."

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

stop:
	$(COMPOSE) down
