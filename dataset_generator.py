#!/usr/bin/env python3
"""
SecretGuard-CLI Dataset Generator
Generates synthetic training data for LLM-based secret detection
"""

import json
import random
import string
import hashlib
import base64
import uuid
from typing import List, Dict, Tuple
import re
from pathlib import Path

class SecretDatasetGenerator:
    def __init__(self):
        self.languages = ['python', 'javascript', 'java', 'go', 'rust', 'cpp', 'csharp', 'php', 'ruby']
        self.secret_types = [
            'api_key', 'database_password', 'jwt_token', 'oauth_token', 
            'private_key', 'certificate', 'connection_string', 'webhook_url',
            'access_token', 'secret_key', 'encryption_key', 'github_token',
            'aws_key', 'gcp_key', 'azure_key', 'slack_token', 'discord_token'
        ]
        
        # Common variable names for secrets
        self.secret_var_names = [
            'api_key', 'API_KEY', 'apiKey', 'secret', 'SECRET', 'password', 
            'PASSWORD', 'pwd', 'token', 'TOKEN', 'auth_token', 'access_token',
            'private_key', 'SECRET_KEY', 'database_url', 'db_password',
            'jwt_secret', 'oauth_secret', 'webhook_secret', 'encryption_key'
        ]
        
        # Common benign variable names
        self.benign_var_names = [
            'config', 'settings', 'debug', 'version', 'timeout', 'max_retries',
            'host', 'port', 'path', 'filename', 'buffer_size', 'log_level',
            'environment', 'mode', 'status', 'count', 'limit', 'offset'
        ]

    def generate_realistic_secret(self, secret_type: str) -> str:
        """Generate realistic-looking secrets based on type"""
        if secret_type == 'api_key':
            return f"sk-{''.join(random.choices(string.ascii_letters + string.digits, k=48))}"
        elif secret_type == 'jwt_token':
            header = base64.b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip('=')
            payload = base64.b64encode(json.dumps({"sub": "1234567890", "name": "John Doe"}).encode()).decode().rstrip('=')
            signature = ''.join(random.choices(string.ascii_letters + string.digits + '-_', k=43))
            return f"{header}.{payload}.{signature}"
        elif secret_type == 'private_key':
            return f"-----BEGIN PRIVATE KEY-----\n{''.join(random.choices(string.ascii_letters + string.digits + '+/', k=64))}\n-----END PRIVATE KEY-----"
        elif secret_type == 'database_password':
            return ''.join(random.choices(string.ascii_letters + string.digits + '!@#$%^&*', k=random.randint(12, 24)))
        elif secret_type == 'github_token':
            return f"ghp_{''.join(random.choices(string.ascii_letters + string.digits, k=36))}"
        elif secret_type == 'aws_key':
            return f"AKIA{''.join(random.choices(string.ascii_uppercase + string.digits, k=16))}"
        elif secret_type == 'connection_string':
            return f"postgresql://user:{''.join(random.choices(string.ascii_letters + string.digits, k=16))}@localhost:5432/db"
        elif secret_type == 'webhook_url':
            return f"https://hooks.slack.com/services/{''.join(random.choices(string.ascii_uppercase + string.digits, k=32))}"
        else:
            return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

    def generate_benign_value(self) -> str:
        """Generate benign configuration values"""
        benign_values = [
            "localhost", "127.0.0.1", "production", "development", "debug",
            "info", "warn", "error", "true", "false", "3000", "8080", "443",
            "/api/v1", "/health", "/status", "utf-8", "json", "xml",
            "GET", "POST", "PUT", "DELETE", "application/json",
            str(random.randint(1, 1000)), str(random.randint(1000, 9999))
        ]
        return random.choice(benign_values)

    def generate_python_snippet(self, has_secret: bool) -> Tuple[str, str]:
        """Generate Python code snippet"""
        if has_secret:
            secret_type = random.choice(self.secret_types)
            var_name = random.choice(self.secret_var_names)
            secret_value = self.generate_realistic_secret(secret_type)
            
            templates = [
                f'{var_name} = "{secret_value}"',
                f'{var_name} = \'{secret_value}\'',
                f'config = {{\n    "{var_name}": "{secret_value}"\n}}',
                f'os.environ["{var_name}"] = "{secret_value}"',
                f'settings.{var_name} = "{secret_value}"',
                f'def get_api_key():\n    return "{secret_value}"',
                f'headers = {{\n    "Authorization": "Bearer {secret_value}"\n}}',
                f'{var_name} = base64.b64decode("{base64.b64encode(secret_value.encode()).decode()}")'.replace('b\'', '').replace('\'', '')
            ]
            return random.choice(templates), "SECRET"
        else:
            var_name = random.choice(self.benign_var_names)
            benign_value = self.generate_benign_value()
            
            templates = [
                f'{var_name} = "{benign_value}"',
                f'{var_name} = {benign_value}' if benign_value.isdigit() or benign_value in ['true', 'false'] else f'{var_name} = "{benign_value}"',
                f'config = {{\n    "{var_name}": "{benign_value}"\n}}',
                f'def get_config():\n    return "{benign_value}"',
                f'logger.info("Starting service on {benign_value}")',
                f'if {var_name} == "{benign_value}":\n    print("Configuration loaded")'
            ]
            return random.choice(templates), "BENIGN"

    def generate_javascript_snippet(self, has_secret: bool) -> Tuple[str, str]:
        """Generate JavaScript code snippet"""
        if has_secret:
            secret_type = random.choice(self.secret_types)
            var_name = random.choice(self.secret_var_names)
            secret_value = self.generate_realistic_secret(secret_type)
            
            templates = [
                f'const {var_name} = "{secret_value}";',
                f'let {var_name} = \'{secret_value}\';',
                f'const config = {{\n  {var_name}: "{secret_value}"\n}};',
                f'process.env.{var_name} = "{secret_value}";',
                f'export const {var_name} = "{secret_value}";',
                f'const headers = {{\n  Authorization: `Bearer ${secret_value}`\n}};',
                f'localStorage.setItem("{var_name}", "{secret_value}");'
            ]
            return random.choice(templates), "SECRET"
        else:
            var_name = random.choice(self.benign_var_names)
            benign_value = self.generate_benign_value()
            
            templates = [
                f'const {var_name} = "{benign_value}";',
                f'let {var_name} = {benign_value};' if benign_value.isdigit() else f'let {var_name} = "{benign_value}";',
                f'const config = {{\n  {var_name}: "{benign_value}"\n}};',
                f'console.log("Server running on {benign_value}");',
                f'if ({var_name} === "{benign_value}") {{\n  console.log("Config loaded");\n}}'
            ]
            return random.choice(templates), "BENIGN"

    def generate_java_snippet(self, has_secret: bool) -> Tuple[str, str]:
        """Generate Java code snippet"""
        if has_secret:
            secret_type = random.choice(self.secret_types)
            var_name = random.choice(self.secret_var_names).upper()
            secret_value = self.generate_realistic_secret(secret_type)
            
            templates = [
                f'private static final String {var_name} = "{secret_value}";',
                f'public static final String {var_name} = "{secret_value}";',
                f'String {var_name.lower()} = "{secret_value}";',
                f'properties.setProperty("{var_name}", "{secret_value}");',
                f'System.setProperty("{var_name}", "{secret_value}");',
                f'@Value("${{{var_name}:{secret_value}}}")\nprivate String {var_name.lower()};'
            ]
            return random.choice(templates), "SECRET"
        else:
            var_name = random.choice(self.benign_var_names).upper()
            benign_value = self.generate_benign_value()
            
            templates = [
                f'private static final String {var_name} = "{benign_value}";',
                f'public static final int {var_name} = {benign_value};' if benign_value.isdigit() else f'public static final String {var_name} = "{benign_value}";',
                f'System.out.println("Application starting on " + {benign_value});',
                f'if ({var_name}.equals("{benign_value}")) {{\n    logger.info("Configuration valid");\n}}'
            ]
            return random.choice(templates), "BENIGN"

    def generate_go_snippet(self, has_secret: bool) -> Tuple[str, str]:
        """Generate Go code snippet"""
        if has_secret:
            secret_type = random.choice(self.secret_types)
            var_name = random.choice(self.secret_var_names)
            secret_value = self.generate_realistic_secret(secret_type)
            
            templates = [
                f'const {var_name} = "{secret_value}"',
                f'var {var_name} = "{secret_value}"',
                f'{var_name} := "{secret_value}"',
                f'os.Setenv("{var_name}", "{secret_value}")',
                f'config := map[string]string{{\n    "{var_name}": "{secret_value}",\n}}',
                f'func getSecret() string {{\n    return "{secret_value}"\n}}'
            ]
            return random.choice(templates), "SECRET"
        else:
            var_name = random.choice(self.benign_var_names)
            benign_value = self.generate_benign_value()
            
            templates = [
                f'const {var_name} = "{benign_value}"',
                f'var {var_name} = "{benign_value}"',
                f'{var_name} := "{benign_value}"',
                f'fmt.Printf("Server listening on %s\\n", "{benign_value}")',
                f'if {var_name} == "{benign_value}" {{\n    log.Println("Config loaded")\n}}'
            ]
            return random.choice(templates), "BENIGN"

    def generate_snippet(self, language: str, has_secret: bool) -> Tuple[str, str]:
        """Generate code snippet for specified language"""
        generators = {
            'python': self.generate_python_snippet,
            'javascript': self.generate_javascript_snippet,
            'java': self.generate_java_snippet,
            'go': self.generate_go_snippet,
        }
        
        if language in generators:
            return generators[language](has_secret)
        else:
            # Fallback generic generator
            if has_secret:
                secret_value = self.generate_realistic_secret(random.choice(self.secret_types))
                var_name = random.choice(self.secret_var_names)
                return f'{var_name} = "{secret_value}"', "SECRET"
            else:
                benign_value = self.generate_benign_value()
                var_name = random.choice(self.benign_var_names)
                return f'{var_name} = "{benign_value}"', "BENIGN"

    def generate_complex_snippet(self, has_secret: bool) -> Tuple[str, str]:
        """Generate more complex, realistic code snippets"""
        language = random.choice(['python', 'javascript', 'java', 'go'])
        
        if has_secret:
            secret_type = random.choice(self.secret_types)
            secret_value = self.generate_realistic_secret(secret_type)
            
            if language == 'python':
                templates = [
                    f'''import os
import requests

class DatabaseConfig:
    def __init__(self):
        self.host = "localhost"
        self.port = 5432
        self.password = "{secret_value}"
        self.user = "admin"
    
    def get_connection_string(self):
        return f"postgresql://{{self.user}}:{{self.password}}@{{self.host}}:{{self.port}}/db"''',
                    
                    f'''def authenticate_user(username, password):
    api_key = "{secret_value}"
    headers = {{
        "Authorization": f"Bearer {{api_key}}",
        "Content-Type": "application/json"
    }}
    response = requests.post("/auth", headers=headers)
    return response.json()''',
                    
                    f'''import json

config_data = {{
    "database": {{
        "host": "localhost",
        "port": 5432,
        "credentials": {{
            "username": "admin",
            "password": "{secret_value}"
        }}
    }},
    "api": {{
        "base_url": "https://api.example.com",
        "timeout": 30
    }}
}}'''
                ]
            elif language == 'javascript':
                templates = [
                    f'''const express = require('express');
const app = express();

const config = {{
    port: process.env.PORT || 3000,
    jwtSecret: "{secret_value}",
    database: {{
        host: 'localhost',
        port: 5432
    }}
}};

app.listen(config.port, () => {{
    console.log(`Server running on port ${{config.port}}`);
}});''',
                    
                    f'''class ApiClient {{
    constructor() {{
        this.baseURL = 'https://api.example.com';
        this.apiKey = "{secret_value}";
        this.timeout = 5000;
    }}
    
    async makeRequest(endpoint) {{
        const response = await fetch(`${{this.baseURL}}/${{endpoint}}`, {{
            headers: {{
                'Authorization': `Bearer ${{this.apiKey}}`,
                'Content-Type': 'application/json'
            }}
        }});
        return response.json();
    }}
}}'''
                ]
            elif language == 'java':
                templates = [
                    f'''public class DatabaseConnection {{
    private static final String DB_PASSWORD = "{secret_value}";
    private static final String DB_USER = "admin";
    private static final String DB_HOST = "localhost";
    
    public Connection getConnection() throws SQLException {{
        String url = "jdbc:postgresql://" + DB_HOST + ":5432/mydb";
        return DriverManager.getConnection(url, DB_USER, DB_PASSWORD);
    }}
}}''',
                    f'''@Configuration
public class ApiConfig {{
    @Value("${{api.secret:{secret_value}}}")
    private String apiSecret;
    
    @Bean
    public RestTemplate restTemplate() {{
        RestTemplate template = new RestTemplate();
        template.getInterceptors().add(new AuthInterceptor(apiSecret));
        return template;
    }}
}}'''
                ]
            else:  # go
                templates = [
                    f'''package main

import (
    "database/sql"
    "fmt"
)

const (
    dbHost     = "localhost"
    dbPort     = 5432
    dbUser     = "admin"
    dbPassword = "{secret_value}"
    dbName     = "testdb"
)

func connectDB() (*sql.DB, error) {{
    psqlInfo := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
        dbHost, dbPort, dbUser, dbPassword, dbName)
    return sql.Open("postgres", psqlInfo)
}}''',
                    f'''package config

import "os"

type Config struct {{
    APIKey      string
    DatabaseURL string
    Port        string
}}

func Load() *Config {{
    return &Config{{
        APIKey:      "{secret_value}",
        DatabaseURL: os.Getenv("DATABASE_URL"),
        Port:        os.Getenv("PORT"),
    }}
}}'''
                ]
            
            return random.choice(templates), "SECRET"
        
        else:
            # Generate benign complex snippets
            if language == 'python':
                templates = [
                    '''import logging
import sys

class ApplicationConfig:
    def __init__(self):
        self.debug = True
        self.log_level = "INFO"
        self.max_connections = 100
        self.timeout = 30
    
    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )''',
                    
                    '''def process_data(input_file, output_file):
    buffer_size = 8192
    chunk_size = 1024
    
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            while True:
                chunk = infile.read(chunk_size)
                if not chunk:
                    break
                processed = chunk.upper()
                outfile.write(processed)'''
                ]
            elif language == 'javascript':
                templates = [
                    '''const config = {
    server: {
        port: 3000,
        host: 'localhost',
        environment: 'development'
    },
    logging: {
        level: 'info',
        format: 'combined'
    },
    features: {
        caching: true,
        compression: true
    }
};

module.exports = config;''',
                    
                    '''class DataProcessor {
    constructor(options = {}) {
        this.batchSize = options.batchSize || 100;
        this.timeout = options.timeout || 5000;
        this.retries = options.retries || 3;
    }
    
    async processItems(items) {
        const results = [];
        for (let i = 0; i < items.length; i += this.batchSize) {
            const batch = items.slice(i, i + this.batchSize);
            const processed = await this.processBatch(batch);
            results.push(...processed);
        }
        return results;
    }
}'''
                ]
            elif language == 'java':
                templates = [
                    '''public class ConfigurationManager {
    private static final String DEFAULT_HOST = "localhost";
    private static final int DEFAULT_PORT = 8080;
    private static final String DEFAULT_ENV = "development";
    
    private String environment;
    private int maxConnections;
    private boolean debugMode;
    
    public ConfigurationManager() {
        this.environment = DEFAULT_ENV;
        this.maxConnections = 100;
        this.debugMode = true;
    }
}''',
                    '''@Component
public class DataService {
    private static final int BATCH_SIZE = 100;
    private static final int TIMEOUT = 5000;
    
    @Autowired
    private DataRepository repository;
    
    public List<ProcessedData> processData(List<RawData> rawData) {
        return rawData.stream()
            .map(this::transform)
            .collect(Collectors.toList());
    }
}'''
                ]
            else:  # go
                templates = [
                    '''package config

import (
    "log"
    "os"
)

type AppConfig struct {
    Port        string
    Environment string
    LogLevel    string
    MaxWorkers  int
}

func LoadConfig() *AppConfig {
    return &AppConfig{
        Port:        getEnv("PORT", "8080"),
        Environment: getEnv("ENV", "development"),
        LogLevel:    getEnv("LOG_LEVEL", "info"),
        MaxWorkers:  100,
    }
}

func getEnv(key, defaultVal string) string {
    if val := os.Getenv(key); val != "" {
        return val
    }
    return defaultVal
}''',
                    '''package main

import (
    "fmt"
    "log"
    "net/http"
)

const (
    DefaultPort = "8080"
    DefaultHost = "localhost"
    Version     = "1.0.0"
)

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/health", healthCheck)
    mux.HandleFunc("/version", versionHandler)
    
    addr := fmt.Sprintf("%s:%s", DefaultHost, DefaultPort)
    log.Printf("Server starting on %s", addr)
    log.Fatal(http.ListenAndServe(addr, mux))
}'''
                ]
            
            return random.choice(templates), "BENIGN"

    def generate_training_sample(self) -> Dict:
        """Generate a single training sample"""
        # 60% complex snippets, 40% simple snippets
        use_complex = random.random() < 0.6
        
        # 50% secret, 50% benign
        has_secret = random.random() < 0.5
        
        if use_complex:
            code, label = self.generate_complex_snippet(has_secret)
            language = random.choice(['python', 'javascript', 'java', 'go'])
        else:
            language = random.choice(self.languages)
            code, label = self.generate_snippet(language, has_secret)
        
        return {
            "code": code,
            "language": language,
            "label": label,
            "has_secret": has_secret
        }

    def generate_dataset(self, num_samples: int, output_file: str = "secret_detection_dataset.jsonl"):
        """Generate complete dataset"""
        samples = []
        
        print(f"Generating {num_samples} samples...")
        
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples} samples")
            
            sample = self.generate_training_sample()
            samples.append(sample)
        
        # Ensure balanced dataset
        secret_count = sum(1 for s in samples if s['has_secret'])
        benign_count = len(samples) - secret_count
        
        print(f"Dataset statistics:")
        print(f"Total samples: {len(samples)}")
        print(f"Secret samples: {secret_count}")
        print(f"Benign samples: {benign_count}")
        print(f"Balance ratio: {secret_count/len(samples):.2%} secrets")
        
        # Language distribution
        lang_dist = {}
        for sample in samples:
            lang = sample['language']
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        
        print(f"Language distribution: {lang_dist}")
        
        # Save dataset
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Dataset saved to {output_file}")
        
        # Generate train/val/test splits
        random.shuffle(samples)
        total = len(samples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        # Save splits
        splits = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            split_file = f"secret_detection_{split_name}.jsonl"
            with open(split_file, 'w') as f:
                for sample in split_samples:
                    f.write(json.dumps(sample) + '\n')
            print(f"{split_name.capitalize()} split saved to {split_file} ({len(split_samples)} samples)")

def main():
    generator = SecretDatasetGenerator()
    
    # Generate datasets of different sizes
    datasets = [
        (50000, "secret_detection_50k.jsonl")     
    ]
    
    for num_samples, filename in datasets:
        print(f"\n{'='*50}")
        print(f"Generating {filename}")
        print(f"{'='*50}")
        generator.generate_dataset(num_samples, filename)

if __name__ == "__main__":
    main()