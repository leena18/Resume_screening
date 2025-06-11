import json
import re
from pymongo import MongoClient
from typing import List, Dict, Any, Optional
from bson.regex import Regex
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException
from pymongo.errors import PyMongoError


class MongoFilter:
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db_name = "Resumes"
        self.collection_name = "Resumes1"
        self.collection = None

    def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client.server_info()
            self.collection = self.client[self.db_name][self.collection_name]
            print("Connected to MongoDB")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

    def close(self) -> None:
        """Close MongoDB connection."""
        try:
            self.client.close()
            print("MongoDB connection closed")
        except Exception as e:
            print(f"Error closing MongoDB connection: {e}")
            raise

    def get_filtered_data(self, filters: Dict[str, Any]) -> List[Dict]:
        """Fetch data based on provided filters and save unfiltered and filtered data to JSON."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        try:
            # Fetch unfiltered data
            unfiltered_data = list(self.collection.find({}))
            for doc in unfiltered_data:
                doc['_id'] = str(doc['_id'])
            
            with open('unfiltered_data.json', 'w') as f:
                json.dump(unfiltered_data, f, indent=4)
            print("Unfiltered data saved to unfiltered_data.json")
            print(f"Unfiltered data count: {len(unfiltered_data)}")

            # Build query dynamically for filtered data
            query = {}

            # Skills filter - check if any of the requested skills exist as keys in core_technical_skills_claimed
            if filters.get("skills"):
                match_conditions = {}
                skills_conditions = []
                for skill in filters["skills"]:
                    # Create case-insensitive field existence check
                    skill_key_pattern = f"^{re.escape(skill)}$"
                    skills_conditions.append({
                        f"core_technical_skills_claimed": {
                            "$regex": skill_key_pattern,
                            "$options": "i"
                        }
                    })
                    
                # Alternative approach: Check if any skill key matches (case-insensitive)
                skills_or_conditions = []
                for skill in filters["skills"]:
                    skills_or_conditions.append({
                        "$expr": {
                            "$gt": [
                                {
                                    "$size": {
                                        "$filter": {
                                            "input": {"$objectToArray": "$core_technical_skills_claimed"},
                                            "cond": {
                                                "$regexMatch": {
                                                    "input": "$$this.k",
                                                    "regex": f"^{re.escape(skill)}$",
                                                    "options": "i"
                                                }
                                            }
                                        }
                                    }
                                },
                                0
                            ]
                        }
                    })
                
                if skills_or_conditions:
                    match_conditions["$or"] = skills_or_conditions
                    query.update(match_conditions)

            # Location filter
            if filters.get("locations"):
                location_pattern = "|".join(re.escape(loc.strip()) for loc in filters["locations"])
                query["preferred_location"] = {
                    "$regex": location_pattern,
                    "$options": "i"
                }

            # Aggregation pipeline
            pipeline = [
                # Convert total_experience to float if it's a string
                {
                    "$addFields": {
                        "total_experience_float": {
                            "$convert": {
                                "input": "$total_experience",
                                "to": "double",
                                "onError": 0.0,  # Default to 0 if conversion fails
                                "onNull": 0.0
                            }
                        }
                    }
                },
                # Apply filters
                {"$match": query},
                # Add experience filter for range x-2 to x+2 (inclusive), ensuring non-negative lower bound
                {
                    "$match": {
                        "total_experience_float": {
                            "$gte": max(0, filters.get("experience", 0) - 2),
                            "$lte": filters.get("experience", 0) + 2
                        }
                    }
                },
                # Remove temporary field
                {
                    "$unset": "total_experience_float"
                }
            ]
            print(f"Final query: {query}")

            filtered_data = list(self.collection.aggregate(pipeline))
            print(json.dumps(pipeline, indent=2))
            print(f"Filtered data count: {len(filtered_data)}")
            
            for doc in filtered_data:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            with open('filtered_data.json', 'w') as f:
                json.dump(filtered_data, f, indent=4)
            print("Filtered data saved to filtered_data.json")

            return filtered_data
        except Exception as e:
            print(f"Error fetching or saving data: {e}")
            raise

    def get_data_by_id(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single resume document by its ID."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            object_id = ObjectId(resume_id)
            resume = self.collection.find_one({"_id": object_id})
            if resume:
                resume['_id'] = str(resume['_id'])  # Convert ObjectId to string
                return resume
            return None
        except InvalidId:
            print(f"Invalid ObjectId format: {resume_id}")
            return None
        except PyMongoError as e:
            print(f"MongoDB error fetching data by ID: {e}")
            raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error fetching data by ID: {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")