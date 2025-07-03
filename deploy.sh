#!/bin/bash

# Deployment script for Quant Bloom Nexus Trading Terminal
set -e

echo "🚀 Starting Quant Bloom Nexus Trading Terminal Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cat > .env << EOF
# Database Configuration
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_SERVER=postgres
POSTGRES_DB=quant_bloom_nexus

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# InfluxDB Configuration
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=my-super-secret-token
INFLUXDB_ORG=my-org
INFLUXDB_BUCKET=market-data

# News API (Required for sentiment analysis)
NEWS_API_KEY=your_news_api_key_here

# Environment
ENVIRONMENT=production
EOF
    echo "📝 Please edit .env file with your actual API keys before continuing."
    echo "   Especially update NEWS_API_KEY with your actual NewsAPI key."
    read -p "Press Enter to continue after updating .env file..."
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo ""
echo "✅ Deployment completed!"
echo ""
echo "🌐 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "📊 Database Access:"
echo "   PostgreSQL: localhost:5432"
echo "   Redis: localhost:6379"
echo "   InfluxDB: http://localhost:8086"
echo ""
echo "🔧 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   Update services: docker-compose pull && docker-compose up -d" 