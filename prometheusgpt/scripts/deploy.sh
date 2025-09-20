#!/bin/bash
# PrometheusGPT Mini - Deployment Script
# Author: MagistrTheOne, Krasnodar, 2025

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для логирования
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Проверка зависимостей
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Проверка NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        warning "NVIDIA Docker runtime not detected. GPU support may not work."
    fi
    
    success "Dependencies check passed"
}

# Создание необходимых директорий
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p data checkpoints logs monitoring/grafana/dashboards monitoring/grafana/datasources monitoring/nginx/ssl
    
    success "Directories created"
}

# Сборка Docker образов
build_images() {
    log "Building Docker images..."
    
    docker-compose -f docker-compose.production.yml build --no-cache
    
    success "Docker images built"
}

# Запуск сервисов
start_services() {
    local profile=$1
    
    log "Starting services with profile: $profile"
    
    if [ -n "$profile" ]; then
        docker-compose -f docker-compose.production.yml --profile $profile up -d
    else
        docker-compose -f docker-compose.production.yml up -d prometheusgpt-api prometheus grafana
    fi
    
    success "Services started"
}

# Остановка сервисов
stop_services() {
    log "Stopping services..."
    
    docker-compose -f docker-compose.production.yml down
    
    success "Services stopped"
}

# Проверка статуса сервисов
check_status() {
    log "Checking service status..."
    
    docker-compose -f docker-compose.production.yml ps
    
    # Проверка health check
    log "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            success "API is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    if [ $i -eq 30 ]; then
        error "API failed to start within 60 seconds"
        return 1
    fi
}

# Показать логи
show_logs() {
    local service=$1
    
    if [ -n "$service" ]; then
        docker-compose -f docker-compose.production.yml logs -f $service
    else
        docker-compose -f docker-compose.production.yml logs -f
    fi
}

# Очистка
cleanup() {
    log "Cleaning up..."
    
    docker-compose -f docker-compose.production.yml down -v
    docker system prune -f
    
    success "Cleanup completed"
}

# Показать помощь
show_help() {
    echo "PrometheusGPT Mini Deployment Script"
    echo "Author: MagistrTheOne, Krasnodar, 2025"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy [profile]    Deploy services (profiles: training, nginx, cache)"
    echo "  start [profile]     Start services"
    echo "  stop               Stop services"
    echo "  restart [profile]  Restart services"
    echo "  status             Check service status"
    echo "  logs [service]     Show logs"
    echo "  cleanup            Clean up everything"
    echo "  help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Deploy basic services"
    echo "  $0 deploy training           # Deploy with training profile"
    echo "  $0 deploy nginx cache        # Deploy with nginx and cache"
    echo "  $0 logs prometheusgpt-api    # Show API logs"
}

# Основная логика
main() {
    case "${1:-help}" in
        deploy)
            check_dependencies
            create_directories
            build_images
            start_services "$2"
            check_status
            ;;
        start)
            start_services "$2"
            check_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            start_services "$2"
            check_status
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs "$2"
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Запуск
main "$@"
