#!/usr/bin/env python3
"""
Create sample data files for testing the Yale Alumni Search Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_alumni_data():
    """Create sample OCS_YALE_PEOPLE_5K.xlsx"""
    
    # Sample data
    names = [
        "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", 
        "David Wilson", "Jessica Miller", "James Anderson", "Ashley Taylor",
        "Christopher Thomas", "Amanda Jackson", "Daniel White", "Lisa Harris",
        "Matthew Martin", "Jennifer Thompson", "Andrew Garcia", "Michelle Martinez",
        "Joshua Robinson", "Stephanie Clark", "Nicholas Rodriguez", "Elizabeth Lewis",
        "Alexander Lee", "Rebecca Walker", "Ryan Hall", "Laura Allen", 
        "Benjamin Young", "Kimberly Hernandez", "Jonathan King", "Donna Wright",
        "Samuel Lopez", "Amy Hill", "Christian Scott", "Carol Green", 
        "Brandon Adams", "Sharon Baker", "Tyler Gonzalez", "Cynthia Nelson",
        "Aaron Carter", "Angela Mitchell", "Jose Perez", "Brenda Roberts",
        "Kevin Turner", "Emma Phillips", "Sean Campbell", "Helen Parker",
        "Nathan Evans", "Deborah Edwards", "Zachary Collins", "Maria Stewart",
        "Gabriel Sanchez", "Sandra Morris", "Caleb Rogers", "Ruth Reed"
    ]
    
    titles = [
        "Software Engineer", "Senior Software Engineer", "Principal Software Engineer",
        "Data Scientist", "Senior Data Scientist", "Staff Data Scientist",
        "Product Manager", "Senior Product Manager", "Director of Product",
        "Investment Banking Analyst", "Investment Banking Associate", "Vice President",
        "Management Consultant", "Senior Consultant", "Principal Consultant",
        "Machine Learning Engineer", "AI Research Scientist", "Research Engineer",
        "Marketing Manager", "Senior Marketing Manager", "Director of Marketing",
        "Sales Representative", "Senior Sales Manager", "VP of Sales",
        "Financial Analyst", "Senior Financial Analyst", "Finance Director",
        "Operations Manager", "Senior Operations Manager", "VP of Operations"
    ]
    
    companies = [
        "Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla",
        "Goldman Sachs", "Morgan Stanley", "JPMorgan Chase", "BlackRock",
        "McKinsey & Company", "Bain & Company", "Boston Consulting Group",
        "Uber", "Airbnb", "Stripe", "Coinbase", "SpaceX", "Palantir",
        "Salesforce", "Adobe", "Intel", "NVIDIA", "Qualcomm",
        "Johnson & Johnson", "Pfizer", "Merck", "Bristol Myers Squibb",
        "Deloitte", "PwC", "EY", "KPMG", "Accenture"
    ]
    
    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
        "Boston, MA", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
        "Atlanta, GA", "Washington, DC", "Philadelphia, PA", "Miami, FL",
        "Portland, OR", "San Diego, CA", "Phoenix, AZ", "Dallas, TX"
    ]
    
    degrees = [
        "Bachelor of Science", "Bachelor of Arts", "Master of Science", 
        "Master of Business Administration", "Master of Arts", "Doctor of Philosophy",
        "Juris Doctor", "Master of Engineering", "Bachelor of Engineering"
    ]
    
    majors = [
        "Computer Science", "Economics", "Mathematics", "Physics", "Chemistry",
        "Biology", "Psychology", "Political Science", "History", "English",
        "Business Administration", "Finance", "Marketing", "Engineering",
        "Data Science", "Statistics", "Philosophy", "Sociology"
    ]
    
    skills_pool = [
        "Python", "Java", "JavaScript", "React", "SQL", "Machine Learning",
        "Data Analysis", "Project Management", "Leadership", "Communication",
        "Strategic Planning", "Financial Modeling", "Excel", "Tableau",
        "PowerBI", "R", "Scala", "Go", "C++", "AWS", "Azure", "GCP",
        "Docker", "Kubernetes", "TensorFlow", "PyTorch", "Pandas", "NumPy"
    ]
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    data = []
    for i in range(5000):  # Create 5000 sample records
        person_id = f"yale_{i+1:04d}"
        name = random.choice(names)
        current_title = random.choice(titles)
        current_company = random.choice(companies)
        location = random.choice(locations)
        graduation_year = random.randint(2010, 2023)
        degree = random.choice(degrees)
        major = random.choice(majors)
        
        # Generate 3-5 skills per person
        num_skills = random.randint(3, 5)
        skills = random.sample(skills_pool, num_skills)
        
        # Generate 1-3 experience entries
        num_experiences = random.randint(1, 3)
        experiences = []
        for j in range(num_experiences):
            exp_title = random.choice(titles)
            exp_company = random.choice(companies)
            start_year = random.randint(graduation_year, 2023)
            end_year = random.randint(start_year, 2024) if j < num_experiences - 1 else None
            
            experiences.append({
                'title': exp_title,
                'company': exp_company,
                'start_date': f"{start_year}-01-01",
                'end_date': f"{end_year}-12-31" if end_year else None
            })
        
        data.append({
            'person_id': person_id,
            'name': name,
            'current_title': current_title,
            'current_company': current_company,
            'location': location,
            'graduation_year': graduation_year,
            'degree': degree,
            'major': major,
            'education_school': 'Yale University',
            'skills': skills,
            'experience': experiences
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_excel('OCS_YALE_PEOPLE_5K.xlsx', index=False)
    print(f"Created OCS_YALE_PEOPLE_5K.xlsx with {len(df)} records")
    
    return df

def create_sample_company_data():
    """Create sample 361_GPT_COMPANIES.csv"""
    
    companies = [
        ("Google", "Technology", "Search Engine", "Mountain View, CA", "Large"),
        ("Microsoft", "Technology", "Software", "Redmond, WA", "Large"),
        ("Apple", "Technology", "Consumer Electronics", "Cupertino, CA", "Large"),
        ("Amazon", "Technology", "E-commerce", "Seattle, WA", "Large"),
        ("Meta", "Technology", "Social Media", "Menlo Park, CA", "Large"),
        ("Goldman Sachs", "Finance", "Investment Banking", "New York, NY", "Large"),
        ("Morgan Stanley", "Finance", "Investment Banking", "New York, NY", "Large"),
        ("JPMorgan Chase", "Finance", "Banking", "New York, NY", "Large"),
        ("McKinsey & Company", "Consulting", "Management Consulting", "New York, NY", "Large"),
        ("Bain & Company", "Consulting", "Management Consulting", "Boston, MA", "Large"),
        ("Tesla", "Automotive", "Electric Vehicles", "Austin, TX", "Large"),
        ("Netflix", "Technology", "Streaming", "Los Gatos, CA", "Large"),
        ("Uber", "Technology", "Transportation", "San Francisco, CA", "Large"),
        ("Airbnb", "Technology", "Hospitality", "San Francisco, CA", "Large"),
        ("Stripe", "Technology", "Payments", "San Francisco, CA", "Medium"),
        ("Coinbase", "Technology", "Cryptocurrency", "San Francisco, CA", "Medium"),
        ("SpaceX", "Aerospace", "Space Technology", "Hawthorne, CA", "Large"),
        ("Palantir", "Technology", "Data Analytics", "Denver, CO", "Medium"),
        ("Salesforce", "Technology", "CRM Software", "San Francisco, CA", "Large"),
        ("Adobe", "Technology", "Creative Software", "San Jose, CA", "Large")
    ]
    
    df = pd.DataFrame(companies, columns=[
        'company_name', 'industry', 'sector', 'headquarters', 'size'
    ])
    
    df.to_csv('361_GPT_COMPANIES.csv', index=False)
    print(f"Created 361_GPT_COMPANIES.csv with {len(df)} records")
    
    return df

if __name__ == "__main__":
    print("Creating sample data files...")
    alumni_df = create_sample_alumni_data()
    company_df = create_sample_company_data()
    print("Sample data creation completed!")
    print("\nYou can now run:")
    print("python main.py --setup")
    print("python main.py --ingest")