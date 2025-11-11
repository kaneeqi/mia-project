# 🚀 Guía de Deployment - MIA Project

Esta guía cubre el deployment de MIA en diferentes plataformas.

---

## 📋 Tabla de Contenidos

- [Preparación](#preparación)
- [Deployment Local (Producción)](#deployment-local-producción)
- [Deployment en la Nube](#deployment-en-la-nube)
  - [Render](#render)
  - [Railway](#railway)
  - [Vercel + Heroku](#vercel--heroku)
  - [AWS](#aws)
- [Docker](#docker)
- [Configuración de Dominio](#configuración-de-dominio)
- [Monitoreo](#monitoreo)

---

## 🛠️ Preparación

### **Checklist Pre-Deployment**

- [ ] Todas las API keys configuradas
- [ ] Modelos de IA presentes (`MiaMotion.h5`, `MiaPredict.h5`)
- [ ] FFmpeg instalado en servidor
- [ ] Rhubarb instalado en servidor
- [ ] Tests pasando
- [ ] Build de producción funcionando localmente
- [ ] Variables de entorno documentadas

### **Build de Producción Local**

```bash
# 1. Frontend
cd frontend
npm run build
# Genera: frontend/dist/

# 2. Backend (no requiere build)
cd backend
# Solo verificar que package.json tiene scripts correctos
```

---

## 🏠 Deployment Local (Producción)

### **Opción 1: PM2 (Recomendado)**

```bash
# Instalar PM2 globalmente
npm install -g pm2

# Crear ecosystem.config.js en la raíz del proyecto
```

**ecosystem.config.js:**
```javascript
module.exports = {
  apps: [
    {
      name: 'mia-backend',
      cwd: './backend',
      script: 'index.js',
      instances: 2,
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      error_file: './logs/backend-error.log',
      out_file: './logs/backend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss'
    },
    {
      name: 'mia-service',
      cwd: './backend',
      script: 'mia_service.py',
      interpreter: 'python3',
      instances: 1,
      env: {
        FLASK_ENV: 'production',
        PORT: 5000
      },
      error_file: './logs/mia-service-error.log',
      out_file: './logs/mia-service-out.log'
    }
  ]
};
```

**Iniciar:**
```bash
# Iniciar todos los servicios
pm2 start ecosystem.config.js

# Ver logs
pm2 logs

# Monitoreo
pm2 monit

# Reiniciar
pm2 restart all

# Detener
pm2 stop all
```

### **Opción 2: Systemd (Linux)**

**backend.service:**
```ini
[Unit]
Description=MIA Backend Service
After=network.target

[Service]
Type=simple
User=mia
WorkingDirectory=/home/mia/mia-project/backend
Environment="NODE_ENV=production"
Environment="PORT=3000"
ExecStart=/usr/bin/node index.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Activar:**
```bash
sudo systemctl enable backend.service
sudo systemctl start backend.service
sudo systemctl status backend.service
```

---

## ☁️ Deployment en la Nube

### **🎨 Render (Recomendado - Free Tier)**

Render permite deployar múltiples servicios gratis.

#### **1. Crear Cuenta**
https://render.com/

#### **2. Preparar Proyecto**

**Crear `render.yaml` en la raíz:**
```yaml
services:
  # Backend Node.js
  - type: web
    name: mia-backend
    env: node
    region: oregon
    plan: free
    buildCommand: cd backend && npm install
    startCommand: cd backend && node index.js
    envVars:
      - key: NODE_ENV
        value: production
      - key: PORT
        value: 3000
      - key: MIA_SERVICE_URL
        value: https://mia-service.onrender.com
      - key: GROQ_API_KEY
        sync: false  # Configurar manualmente
      - key: ELEVEN_LABS_API_KEY
        sync: false
      - key: ELEVEN_LABS_VOICE_ID
        value: EXAVITQu4vr4xnSDxMaL
      - key: TEXT_MODE
        value: groq
    
  # MIA Service Python
  - type: web
    name: mia-service
    env: python
    region: oregon
    plan: free
    buildCommand: cd backend && pip install -r requirements.txt
    startCommand: cd backend && python mia_service.py
    envVars:
      - key: PORT
        value: 5000
      - key: FLASK_ENV
        value: production
  
  # Frontend (Static Site)
  - type: web
    name: mia-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: ./frontend/dist
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

#### **3. Deploy**

```bash
# Push a GitHub
git push origin main

# En Render Dashboard:
# 1. New → Blueprint
# 2. Conectar GitHub repo
# 3. Render detecta render.yaml
# 4. Deploy automático
```

#### **4. Configurar Variables de Entorno Sensibles**

En Render Dashboard:
1. Ir a `mia-backend` service
2. Environment
3. Agregar manualmente:
   - `GROQ_API_KEY`
   - `ELEVEN_LABS_API_KEY`

#### **5. Instalar Herramientas del Sistema**

**Para FFmpeg y Rhubarb en Render:**

Crear `render-build.sh`:
```bash
#!/bin/bash

# Instalar FFmpeg
apt-get update
apt-get install -y ffmpeg

# Descargar Rhubarb
wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip
unzip rhubarb-lip-sync-1.13.0-linux.zip
mv rhubarb /usr/local/bin/
chmod +x /usr/local/bin/rhubarb
```

Actualizar `render.yaml`:
```yaml
buildCommand: bash render-build.sh && cd backend && npm install
```

---

### **🚂 Railway**

Railway es otra opción con free tier generoso.

#### **1. Instalar CLI**
```bash
npm install -g @railway/cli
railway login
```

#### **2. Inicializar Proyecto**
```bash
railway init
railway link
```

#### **3. Crear `railway.json`:**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "node backend/index.js",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### **4. Deploy**
```bash
railway up
railway open
```

---

### **▲ Vercel (Frontend) + Heroku (Backend)**

#### **Frontend en Vercel:**

```bash
# Instalar Vercel CLI
npm i -g vercel

# Deploy desde carpeta frontend
cd frontend
vercel
# Seguir prompts

# Producción
vercel --prod
```

**Configuración en `vercel.json`:**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "vite",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

#### **Backend en Heroku:**

```bash
# Instalar Heroku CLI
brew tap heroku/brew && brew install heroku

# Login
heroku login

# Crear app
heroku create mia-backend

# Agregar buildpacks
heroku buildpacks:add --index 1 heroku/nodejs
heroku buildpacks:add --index 2 heroku/python

# Configurar vars
heroku config:set GROQ_API_KEY=tu_key
heroku config:set ELEVEN_LABS_API_KEY=tu_key

# Deploy
git push heroku main
```

**Crear `Procfile`:**
```
web: cd backend && node index.js
worker: cd backend && python mia_service.py
```

---

### **☁️ AWS (Avanzado)**

**Arquitectura Recomendada:**
```
[CloudFront] → [S3] (Frontend estático)
      ↓
[API Gateway] → [Lambda] (Backend Node.js)
      ↓
[EC2] → MIA Service (Python + modelos)
      ↓
[RDS/DynamoDB] → Base de datos (futuro)
```

**Costos Estimados:** $20-50/mes

---

## 🐳 Docker

### **Dockerfile - Backend**

**backend/Dockerfile:**
```dockerfile
FROM node:18-alpine

# Instalar FFmpeg
RUN apk add --no-cache ffmpeg

# Instalar Python para MIA Service
RUN apk add --no-cache python3 py3-pip

# Directorio de trabajo
WORKDIR /app

# Copiar package.json
COPY package*.json ./

# Instalar dependencias Node.js
RUN npm install --production

# Copiar requirements.txt
COPY requirements.txt ./

# Instalar dependencias Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Descargar Rhubarb
RUN wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip && \
    unzip rhubarb-lip-sync-1.13.0-linux.zip && \
    mv rhubarb bin/ && \
    chmod +x bin/rhubarb && \
    rm rhubarb-lip-sync-1.13.0-linux.zip

# Exponer puertos
EXPOSE 3000 5000

# Script de inicio
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
```

**docker-entrypoint.sh:**
```bash
#!/bin/sh

# Iniciar MIA Service en background
python3 mia_service.py &

# Esperar a que inicie
sleep 5

# Iniciar Backend
node index.js
```

### **Dockerfile - Frontend**

**frontend/Dockerfile:**
```dockerfile
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Servidor estático
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### **docker-compose.yml**

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "3000:3000"
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - GROQ_API_KEY=${GROQ_API_KEY}
      - ELEVEN_LABS_API_KEY=${ELEVEN_LABS_API_KEY}
      - ELEVEN_LABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL
    volumes:
      - ./backend/models:/app/models
      - ./backend/audios:/app/audios
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    environment:
      - VITE_API_URL=http://localhost:3000
```

**Ejecutar:**
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## 🌐 Configuración de Dominio

### **1. Comprar Dominio**
- Namecheap, GoDaddy, Google Domains

### **2. Configurar DNS**

```
Tipo    Nombre      Valor
A       @           [IP del servidor]
CNAME   www         mia.tudominio.com
CNAME   api         [Backend URL]
```

### **3. Configurar HTTPS (Let's Encrypt)**

```bash
# Instalar Certbot
sudo apt install certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d mia.tudominio.com -d www.mia.tudominio.com

# Renovación automática
sudo certbot renew --dry-run
```

---

## 📊 Monitoreo

### **Opción 1: PM2 Monitor**
```bash
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
```

### **Opción 2: Sentry**

```bash
npm install @sentry/node
```

**backend/index.js:**
```javascript
import * as Sentry from "@sentry/node";

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV
});

app.use(Sentry.Handlers.errorHandler());
```

### **Opción 3: Uptime Monitoring**

- **UptimeRobot** (gratis)
- **Pingdom**
- **StatusCake**

---

## 🔧 Troubleshooting Production

### **Problema: Alta latencia**
```bash
# Verificar recursos
htop
df -h

# Ver logs
pm2 logs --lines 100

# Optimizar
pm2 reload all
```

### **Problema: Errores de memoria**
```javascript
// Aumentar límite de memoria Node.js
node --max-old-space-size=4096 index.js
```

### **Problema: Rhubarb no funciona**
```bash
# Verificar instalación
which rhubarb
rhubarb --version

# Reinstalar
sudo apt install rhubarb-lip-sync
```

---

## 📈 Escalamiento

### **Horizontal Scaling**
```bash
# PM2 cluster mode
pm2 start index.js -i max  # Usa todos los CPUs
```

### **Load Balancer (Nginx)**

**nginx.conf:**
```nginx
upstream backend {
    server localhost:3000;
    server localhost:3001;
    server localhost:3002;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

---

## 📚 Recursos

- [Render Docs](https://render.com/docs)
- [Railway Docs](https://docs.railway.app/)
- [Vercel Docs](https://vercel.com/docs)
- [PM2 Docs](https://pm2.keymetrics.io/docs)
- [Docker Docs](https://docs.docker.com/)

---

**¿Problemas?** Abre un [Issue en GitHub](https://github.com/tu-usuario/mia-project/issues)
