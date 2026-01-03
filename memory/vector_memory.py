import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from utils.config import Config


class VectorMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            metadata={"description": "Vector store for research data"}
        )

    def store(self, text: str, metadata: Dict[str, Any]) -> str:
        """Store text with metadata in vector database"""
        doc_id = str(uuid.uuid4())

        self.collection.add(
            documents=[text],
            metadatas=[{
                **metadata,
                "timestamp": datetime.now().isoformat(),
                "id": doc_id
            }],
            ids=[doc_id]
        )

        return doc_id

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })

            return formatted_results

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def get_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve items by metadata filters"""
        try:
            results = self.collection.get(
                where=filters,
                limit=10
            )

            formatted_results = []
            for i, doc_id in enumerate(results['ids']):
                formatted_results.append({
                    "id": doc_id,
                    "text": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })

            return formatted_results

        except Exception as e:
            print(f"Metadata filter error: {e}")
            return []

    def update(self, doc_id: str, text: Optional[str] = None,
               metadata: Optional[Dict] = None) -> bool:
        """Update existing document"""
        try:
            if text:
                self.collection.update(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata] if metadata else None
                )
            elif metadata:
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[metadata]
                )

            return True

        except Exception as e:
            print(f"Update error: {e}")
            return False

    def delete(self, doc_id: str) -> bool:
        """Delete document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False

    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all stored items"""
        try:
            results = self.collection.get(limit=limit)

            formatted_results = []
            for i, doc_id in enumerate(results['ids']):
                formatted_results.append({
                    "id": doc_id,
                    "text": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })

            return formatted_results

        except Exception as e:
            print(f"Get all error: {e}")
            return []

    def clear(self) -> bool:
        """Clear all stored data"""
        try:
            self.client.delete_collection("research_memory")
            self.collection = self.client.get_or_create_collection(
                name="research_memory",
                metadata={"description": "Vector store for research data"}
            )
            return True
        except Exception as e:
            print(f"Clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        try:
            all_items = self.get_all()

            # Count by type
            type_counts = {}
            for item in all_items:
                item_type = item['metadata'].get('type', 'unknown')
                type_counts[item_type] = type_counts.get(item_type, 0) + 1

            # Count by agent
            agent_counts = {}
            for item in all_items:
                agent = item['metadata'].get('agent', 'unknown')
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

            return {
                "total_items": len(all_items),
                "type_counts": type_counts,
                "agent_counts": agent_counts,
                "oldest_timestamp": min(
                    (item['metadata'].get('timestamp', '') for item in all_items if item['metadata'].get('timestamp')),
                    default=''
                ),
                "newest_timestamp": max(
                    (item['metadata'].get('timestamp', '') for item in all_items if item['metadata'].get('timestamp')),
                    default=''
                )
            }

        except Exception as e:
            print(f"Stats error: {e}")
            return {"error": str(e)}