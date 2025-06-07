#!/usr/bin/env python3
"""
Generate 5,000 test cases in under 10 minutes
Uses templates and variations to create realistic test data
"""
import json
import random
import string
from pathlib import Path
from itertools import product

def generate_random_string(length, chars=string.ascii_letters + string.digits):
    """Generate random string of given length"""
    return ''.join(random.choice(chars) for _ in range(length))

def generate_hex_string(length):
    """Generate random hex string"""
    return ''.join(random.choice('0123456789abcdef') for _ in range(length))

class MassTestGenerator:
    def __init__(self):
        self.test_cases = []
        self.case_id = 1
        
    def add_case(self, code, expected, category, description=""):
        """Add a test case"""
        self.test_cases.append({
            "id": f"{category}_{self.case_id}",
            "code": code,
            "expected": expected,
            "category": category,
            "description": description
        })
        self.case_id += 1

    def generate_aws_keys(self, count=500):
        """Generate AWS key variations"""
        print(f"Generating {count} AWS key test cases...")
        
        # AWS Access Key patterns
        aws_prefixes = ["AKIA", "ASIA", "AROA"]
        variable_names = ["accessKey", "aws_key", "AWS_ACCESS_KEY_ID", "awsAccessKeyId", "access_key"]
        
        for i in range(count // 2):
            # Positive cases
            prefix = random.choice(aws_prefixes)
            suffix = generate_random_string(16, string.ascii_uppercase + string.digits)
            aws_key = f"{prefix}{suffix}"
            var_name = random.choice(variable_names)
            
            # Different code formats
            formats = [
                f'const {var_name} = "{aws_key}";',
                f'{var_name.upper()} = "{aws_key}"',
                f'export {var_name} = "{aws_key}";',
                f'const config = {{ {var_name}: "{aws_key}" }};',
                f'process.env.{var_name.upper()} = "{aws_key}";'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "aws_key", f"AWS access key variant {i}")
            
        # AWS Secret Keys
        for i in range(count // 2):
            secret_key = generate_random_string(40, string.ascii_letters + string.digits + '+/=')
            var_names = ["secretKey", "aws_secret", "AWS_SECRET_ACCESS_KEY", "awsSecretAccessKey"]
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{secret_key}";',
                f'{var_name.upper()} = "{secret_key}"',
                f'AWS_SECRET_ACCESS_KEY="{secret_key}"'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "aws_secret", f"AWS secret key variant {i}")

    def generate_github_tokens(self, count=300):
        """Generate GitHub token variations"""
        print(f"Generating {count} GitHub token test cases...")
        
        prefixes = ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
        var_names = ["token", "github_token", "GITHUB_TOKEN", "githubToken", "auth_token"]
        
        for i in range(count):
            prefix = random.choice(prefixes)
            suffix = generate_random_string(36, string.ascii_letters + string.digits)
            token = f"{prefix}{suffix}"
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{token}";',
                f'{var_name.upper()}="{token}"',
                f'headers = {{"Authorization": "token {token}"}}',
                f'const authHeader = `Bearer {token}`;',
                f'git config --global github.token {token}'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "github_token", f"GitHub token variant {i}")

    def generate_api_keys(self, count=400):
        """Generate various API key patterns"""
        print(f"Generating {count} API key test cases...")
        
        api_patterns = [
            ("sk-", 48, "openai"),
            ("pk_test_", 24, "stripe_test"),
            ("pk_live_", 24, "stripe_live"),
            ("sk_test_", 24, "stripe_test_secret"),
            ("sk_live_", 24, "stripe_live_secret"),
            ("AIza", 35, "google"),
            ("ya29.", 30, "google_oauth"),
            ("xoxb-", 50, "slack_bot"),
            ("xoxp-", 50, "slack_user")
        ]
        
        var_names = ["apiKey", "api_key", "API_KEY", "key", "secret", "token"]
        
        for i in range(count):
            pattern, length, service = random.choice(api_patterns)
            key = pattern + generate_random_string(length, string.ascii_letters + string.digits + '-_')
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{key}";',
                f'{var_name.upper()} = "{key}"',
                f'process.env.{var_name.upper()} = "{key}";',
                f'const config = {{ {service}_key: "{key}" }};'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "api_key", f"{service} API key variant {i}")

    def generate_passwords(self, count=300):
        """Generate password variations"""
        print(f"Generating {count} password test cases...")
        
        # Password patterns
        password_patterns = [
            "password123", "admin123", "secret123", "123456789", "qwerty123",
            "Password@123", "Admin#2024", "Secret!Pass", "MySecr3t!", "P@ssw0rd123"
        ]
        
        var_names = ["password", "pass", "pwd", "PASSWORD", "user_pass", "admin_pass", "db_password"]
        
        for i in range(count):
            if i < len(password_patterns):
                password = password_patterns[i]
            else:
                # Generate random passwords
                password = generate_random_string(8, string.ascii_letters) + str(random.randint(100, 999))
                if random.choice([True, False]):
                    password += random.choice(['!', '@', '#', '$', '%'])
            
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{password}";',
                f'{var_name.upper()} = "{password}"',
                f'DATABASE_PASSWORD="{password}"',
                f'const loginCredentials = {{ password: "{password}" }};'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "password", f"Password variant {i}")

    def generate_jwt_tokens(self, count=200):
        """Generate JWT token variations"""
        print(f"Generating {count} JWT token test cases...")
        
        # JWT structure: header.payload.signature
        headers = ["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"]
        
        for i in range(count):
            header = random.choice(headers)
            payload = generate_random_string(50, string.ascii_letters + string.digits + '+/=')
            signature = generate_random_string(40, string.ascii_letters + string.digits + '+/=')
            jwt = f"{header}.{payload}.{signature}"
            
            var_names = ["jwt", "token", "jwtToken", "accessToken", "authToken"]
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{jwt}";',
                f'Authorization: Bearer {jwt}',
                f'const token = "{jwt}";',
                f'localStorage.setItem("token", "{jwt}");'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "jwt_token", f"JWT token variant {i}")

    def generate_database_urls(self, count=200):
        """Generate database connection strings"""
        print(f"Generating {count} database URL test cases...")
        
        db_types = [
            ("mongodb", "mongodb://{}:{}@localhost:27017/{}"),
            ("postgresql", "postgresql://{}:{}@localhost:5432/{}"),
            ("mysql", "mysql://{}:{}@localhost:3306/{}"),
            ("redis", "redis://{}:{}@localhost:6379/0"),
            ("sqlserver", "Server=localhost;Database={};User Id={};Password={};")
        ]
        
        usernames = ["admin", "root", "user", "dbuser", "postgres", "mysql"]
        passwords = ["password", "secret", "admin123", "dbpass", "rootpass"]
        databases = ["mydb", "production", "app", "main", "database"]
        
        for i in range(count):
            db_type, template = random.choice(db_types)
            username = random.choice(usernames)
            password = random.choice(passwords)
            database = random.choice(databases)
            
            if db_type == "sqlserver":
                url = template.format(database, username, password)
            else:
                url = template.format(username, password, database)
            
            var_names = ["dbUrl", "connectionString", "DATABASE_URL", "db_connection"]
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{url}";',
                f'{var_name.upper()}="{url}"',
                f'const dbConfig = {{ url: "{url}" }};'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "Yes", "database_url", f"{db_type} connection variant {i}")

    def generate_safe_code(self, count=2000):
        """Generate safe code examples (no secrets)"""
        print(f"Generating {count} safe code test cases...")
        
        # Safe variable examples
        safe_variables = [
            "userName", "userId", "firstName", "lastName", "email", "phone",
            "address", "city", "country", "zipCode", "age", "status"
        ]
        
        safe_values = [
            "john_doe", "jane_smith", "user123", "admin", "guest",
            "localhost", "127.0.0.1", "example.com", "test@example.com"
        ]
        
        # Safe functions
        safe_functions = [
            "function calculateSum(a, b) { return a + b; }",
            "const validateEmail = (email) => email.includes('@');",
            "function getUserById(id) { return users.find(u => u.id === id); }",
            "const formatDate = (date) => date.toISOString();",
            "function isValidUser(user) { return user && user.id; }"
        ]
        
        # Safe imports/requires
        safe_imports = [
            "import React from 'react';",
            "const express = require('express');",
            "import { useState } from 'react';",
            "const fs = require('fs');",
            "import axios from 'axios';"
        ]
        
        # Safe configurations
        safe_configs = [
            "const config = { host: 'localhost', port: 3000 };",
            "const apiUrl = 'https://api.example.com';",
            "const defaultSettings = { theme: 'dark', language: 'en' };",
            "const endpoints = { users: '/api/users', posts: '/api/posts' };"
        ]
        
        all_safe_examples = safe_functions + safe_imports + safe_configs
        
        for i in range(count):
            if i < len(all_safe_examples):
                code = all_safe_examples[i]
            else:
                # Generate random safe variable assignments
                var_name = random.choice(safe_variables)
                value = random.choice(safe_values)
                code = f'const {var_name} = "{value}";'
            
            self.add_case(code, "No", "safe", f"Safe code variant {i}")

    def generate_env_variables(self, count=500):
        """Generate environment variable usage (safe)"""
        print(f"Generating {count} environment variable test cases...")
        
        env_patterns = [
            "process.env.{}",
            "os.getenv('{}')",
            "os.environ.get('{}')",
            "System.getenv(\"{}\")",
            "Environment.GetEnvironmentVariable(\"{}\")"
        ]
        
        env_vars = [
            "NODE_ENV", "PORT", "HOST", "API_KEY", "DATABASE_URL", "SECRET_KEY",
            "JWT_SECRET", "GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "REDIS_URL"
        ]
        
        for i in range(count):
            pattern = random.choice(env_patterns)
            var = random.choice(env_vars)
            code = f"const config = {pattern.format(var)};"
            
            self.add_case(code, "No", "env_variable", f"Environment variable usage {i}")

    def generate_placeholders(self, count=300):
        """Generate placeholder text (safe)"""
        print(f"Generating {count} placeholder test cases...")
        
        placeholders = [
            "YOUR_API_KEY_HERE", "REPLACE_WITH_YOUR_KEY", "INSERT_SECRET_HERE",
            "YOUR_PASSWORD_HERE", "ENTER_YOUR_TOKEN", "ADD_YOUR_KEY",
            "YOUR_DATABASE_URL", "YOUR_JWT_SECRET", "REPLACE_ME", "TODO_ADD_KEY"
        ]
        
        var_names = ["apiKey", "password", "secret", "token", "key"]
        
        for i in range(count):
            placeholder = random.choice(placeholders)
            var_name = random.choice(var_names)
            
            formats = [
                f'const {var_name} = "{placeholder}";',
                f'{var_name.upper()}="{placeholder}"',
                f'// TODO: Replace {placeholder} with actual value',
                f'const config = {{ {var_name}: "{placeholder}" }};'
            ]
            
            code = random.choice(formats)
            self.add_case(code, "No", "placeholder", f"Placeholder variant {i}")

    def generate_all(self):
        """Generate all test cases"""
        print("🚀 Generating 5,000 test cases...")
        print("This will take about 5-10 minutes...")
        
        # Generate each category
        self.generate_aws_keys(500)      # 500 AWS cases
        self.generate_github_tokens(300) # 300 GitHub cases  
        self.generate_api_keys(400)      # 400 API key cases
        self.generate_passwords(300)     # 300 password cases
        self.generate_jwt_tokens(200)    # 200 JWT cases
        self.generate_database_urls(200) # 200 DB URL cases
        self.generate_safe_code(2000)    # 2000 safe cases (40%)
        self.generate_env_variables(500) # 500 env var cases
        self.generate_placeholders(300)  # 300 placeholder cases
        
        # Shuffle the test cases for randomization
        random.shuffle(self.test_cases)
        
        print(f"✅ Generated {len(self.test_cases)} test cases!")
        return self.test_cases

    def save_to_file(self, filename="data/test_samples.json"):
        """Save test cases to JSON file"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "total_cases": len(self.test_cases),
                "generation_method": "automated_template_based",
                "categories": {}
            },
            "test_cases": self.test_cases
        }
        
        # Count categories
        for case in self.test_cases:
            cat = case['category']
            data["metadata"]["categories"][cat] = data["metadata"]["categories"].get(cat, 0) + 1
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(self.test_cases)} test cases to {filename}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        for category, count in data["metadata"]["categories"].items():
            percentage = (count / len(self.test_cases)) * 100
            print(f"   {category}: {count} cases ({percentage:.1f}%)")

def main():
    print("🎯 MASS TEST CASE GENERATOR")
    print("=" * 50)
    print("Generating 5,000 realistic test cases for SecretGuard evaluation")
    
    generator = MassTestGenerator()
    test_cases = generator.generate_all()
    generator.save_to_file()
    
    print("\n🎉 SUCCESS!")
    print("You now have 5,000 test cases ready for evaluation!")
    print("\n🚀 Next steps:")
    print("1. Run: python scripts/run_evaluation.py --test-data data/massive_test_dataset.json")
    print("2. Wait for evaluation to complete (~30-60 minutes)")
    print("3. Check results in data/evaluation_results/")

if __name__ == "__main__":
    main()