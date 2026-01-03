import asyncio
import aiohttp
from typing import List, Dict, Optional
import json
from utils.config import Config


class SearchTools:
    def __init__(self):
        self.tavily_key = Config.TAVILY_API_KEY

    async def search_tavily(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search using Tavily API"""
        if not self.tavily_key:
            print("⚠️ No Tavily API key, using mock data")
            return self._mock_search_results(query)

        url = "https://api.tavily.com/search"

        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": self.tavily_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic"
            }

            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        print(f"Tavily search failed: {response.status}")
                        return self._mock_search_results(query)
            except Exception as e:
                print(f"Tavily search error: {e}")
                return self._mock_search_results(query)

    async def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web using Tavily API"""
        return await self.search_tavily(query, max_results)

    async def search_arxiv(self, query: str, max_results: int = 3, sort_by: str = "relevance") -> List[Dict]:
        """Search arXiv for recent papers"""
        try:
            import arxiv
            client = arxiv.Client()

            sort_criterion = arxiv.SortCriterion.SubmittedDate if sort_by == "date" else arxiv.SortCriterion.Relevance

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "summary": paper.summary[:500],  # Limit summary
                    "published": str(paper.published),
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories
                })

            return results
        except ImportError:
            print("⚠️ arXiv package not installed, using mock data")
            return self._mock_arxiv_results(query)
        except Exception as e:
            print(f"arXiv search error: {e}")
            return self._mock_arxiv_results(query)

    async def search_github(self, query: str, max_results: int = 5, sort: str = "stars") -> List[Dict]:
        """Search GitHub repositories (mock for now)"""
        print(f"⚠️ GitHub search (mock): {query}")
        return [
            {
                "name": f"{query}-repo",
                "description": f"Repository about {query}",
                "html_url": f"https://github.com/example/{query}",
                "stargazers_count": 150,
                "forks_count": 25,
                "updated_at": "2024-01-15",
                "language": "Python"
            }
            for i in range(min(max_results, 3))
        ]

    async def search_datasets(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for datasets (mock for now)"""
        print(f"⚠️ Dataset search (mock): {query}")
        return [
            {
                "title": f"Dataset: {query} {i + 1}",
                "description": f"Comprehensive dataset for {query} research",
                "url": f"https://kaggle.com/datasets/{query}-{i + 1}",
                "platform": "Kaggle",
                "size": "500MB",
                "records": 10000,
                "format": "CSV",
                "license": "CC BY 4.0"
            }
            for i in range(min(max_results, 2))
        ]

    def _mock_search_results(self, query: str) -> List[Dict]:
        """Return mock results for testing"""
        return [
            {
                "title": f"Emerging research in {query}",
                "url": "https://example.com/emerging-research",
                "snippet": f"Recent breakthroughs in {query} show promising results in 2024...",
                "score": 0.9,
                "content": f"This is mock content about {query} for testing purposes."
            }
        ]

    def _mock_arxiv_results(self, query: str) -> List[Dict]:
        """Return mock arXiv results"""
        return [
            {
                "title": f"Advances in {query}: A 2024 Perspective",
                "authors": ["Researcher A", "Researcher B"],
                "summary": f"This paper discusses recent advances in {query} with focus on 2024 developments.",
                "published": "2024-01-15",
                "pdf_url": f"https://arxiv.org/pdf/2401.12345.pdf",
                "categories": ["cs.AI", "cs.LG"]
            }
        ]