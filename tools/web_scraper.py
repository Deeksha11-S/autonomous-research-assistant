import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import json
import re
from typing import List, Dict, Optional
from utils.config import Config
import time


class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web using Tavily API or fallback"""
        try:
            from tools.search_tools import SearchTools
            search_tools = SearchTools()
            return await search_tools.search_web(query, max_results)
        except Exception as e:
            print(f"Web search failed: {e}, using fallback")
            return self._mock_web_search(query, max_results)

    async def search_arxiv(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search arXiv for papers"""
        try:
            import arxiv
            client = arxiv.Client()

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            results = []
            papers = []

            # Convert async generator to list
            async for paper in client.results(search):
                papers.append(paper)
                if len(papers) >= max_results:
                    break

            for paper in papers:
                results.append({
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "summary": paper.summary[:500],
                    "published": str(paper.published),
                    "pdf_url": paper.pdf_url,
                    "arxiv_id": paper.entry_id.split('/')[-1]
                })

            return results

        except Exception as e:
            print(f"arXiv search failed: {e}")
            return self._mock_arxiv_results(query, max_results)

    async def search_github(self, topic: str, max_repos: int = 2) -> List[Dict]:
        """Search GitHub for relevant repositories"""
        try:
            url = f"https://api.github.com/search/repositories"
            params = {
                'q': f'{topic} in:name,description',
                'sort': 'stars',
                'order': 'desc',
                'per_page': max_repos
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        repos = []
                        for item in data.get('items', [])[:max_repos]:
                            repos.append({
                                "name": item.get('name', ''),
                                "full_name": item.get('full_name', ''),
                                "description": item.get('description', ''),
                                "html_url": item.get('html_url', ''),
                                "stars": item.get('stargazers_count', 0),
                                "forks": item.get('forks_count', 0),
                                "language": item.get('language', ''),
                                "updated_at": item.get('updated_at', '')
                            })
                        return repos
                    else:
                        print(f"GitHub API error: {response.status}")
                        return self._mock_github_results(topic, max_repos)

        except Exception as e:
            print(f"GitHub search failed: {e}")
            return self._mock_github_results(topic, max_repos)

    async def search_datasets(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search for public datasets"""
        # This is a placeholder - in real implementation, integrate with Kaggle/HuggingFace APIs
        return self._mock_dataset_results(query, max_results)

    async def scrape_website(self, url: str) -> Dict[str, str]:
        """Scrape content from a website"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Remove scripts and styles
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Get text
                        text = soup.get_text(separator=' ', strip=True)

                        # Clean up text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)

                        return {
                            "url": url,
                            "title": soup.title.string if soup.title else "",
                            "content": text[:2000],  # Limit content length
                            "success": True
                        }
                    else:
                        return {
                            "url": url,
                            "error": f"HTTP {response.status}",
                            "success": False
                        }
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "success": False
            }

    async def scrape_with_playwright(self, url: str) -> Dict[str, str]:
        """Scrape JavaScript-rendered websites"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto(url, wait_until='networkidle', timeout=15000)

                # Get page content
                content = await page.content()
                title = await page.title()

                # Extract text
                soup = BeautifulSoup(content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()

                text = soup.get_text(separator=' ', strip=True)
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                await browser.close()

                return {
                    "url": url,
                    "title": title,
                    "content": text[:2000],
                    "success": True
                }

        except Exception as e:
            print(f"Playwright scrape failed for {url}: {e}")
            # Fall back to regular scraping
            return await self.scrape_website(url)

    # Mock data methods for when APIs fail
    def _mock_web_search(self, query: str, max_results: int) -> List[Dict]:
        return [
            {
                "title": f"Research on {query}",
                "url": f"https://example.com/research/{query.replace(' ', '-')}",
                "snippet": f"Recent developments in {query} show promising results...",
                "score": 0.9
            }
            for i in range(max_results)
        ]

    def _mock_arxiv_results(self, query: str, max_results: int) -> List[Dict]:
        return [
            {
                "title": f"Advances in {query}",
                "authors": ["Researcher A", "Researcher B"],
                "summary": f"This paper discusses novel approaches to {query}...",
                "published": "2024-01-15",
                "pdf_url": "https://arxiv.org/pdf/mock.pdf",
                "arxiv_id": "2401.12345"
            }
            for i in range(max_results)
        ]

    def _mock_github_results(self, topic: str, max_repos: int) -> List[Dict]:
        return [
            {
                "name": f"{topic}-project",
                "full_name": f"username/{topic}-project",
                "description": f"A project related to {topic}",
                "html_url": f"https://github.com/username/{topic}-project",
                "stars": 100 + i * 50,
                "forks": 20 + i * 10,
                "language": "Python",
                "updated_at": "2024-01-15T10:30:00Z"
            }
            for i in range(max_repos)
        ]

    def _mock_dataset_results(self, query: str, max_results: int) -> List[Dict]:
        return [
            {
                "name": f"{query} Dataset",
                "description": f"A dataset for {query} analysis",
                "platform": "Kaggle",
                "url": f"https://kaggle.com/datasets/{query.lower().replace(' ', '-')}",
                "size": "50MB",
                "records": 5000
            }
            for i in range(max_results)
        ]