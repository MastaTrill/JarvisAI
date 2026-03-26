"""
Jarvis AI - External API Integrations
Pre-built connectors for popular external services
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging

from .data_connectors import RESTAPIConnector, RESTAPIConfig, DataIntegrationManager

logger = logging.getLogger(__name__)


# =============================================================================
# OPENAI INTEGRATION
# =============================================================================

class OpenAIConnector(RESTAPIConnector):
    """Connector for OpenAI API"""
    
    def __init__(self, api_key: str, organization: str = None):
        config = RESTAPIConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            auth_type="bearer",
            auth_config={"token": api_key},
            headers={
                "Content-Type": "application/json",
                **({"OpenAI-Organization": organization} if organization else {})
            },
            rate_limit=60
        )
        super().__init__(config)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate chat completion"""
        response = await self.fetch({
            "endpoint": "/chat/completions",
            "method": "POST",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        })
        return response[0] if response else {}
    
    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Generate text embedding"""
        response = await self.fetch({
            "endpoint": "/embeddings",
            "method": "POST",
            "body": {
                "model": model,
                "input": text
            }
        })
        if response:
            return response[0].get("data", [{}])[0].get("embedding", [])
        return []


# =============================================================================
# HUGGING FACE INTEGRATION
# =============================================================================

class HuggingFaceConnector(RESTAPIConnector):
    """Connector for Hugging Face Inference API"""
    
    def __init__(self, api_key: str):
        config = RESTAPIConfig(
            name="huggingface",
            base_url="https://api-inference.huggingface.co",
            auth_type="bearer",
            auth_config={"token": api_key},
            rate_limit=100
        )
        super().__init__(config)
    
    async def inference(
        self,
        model: str,
        inputs: Any,
        parameters: Dict[str, Any] = None
    ) -> Any:
        """Run inference on a Hugging Face model"""
        response = await self.fetch({
            "endpoint": f"/models/{model}",
            "method": "POST",
            "body": {
                "inputs": inputs,
                **({"parameters": parameters} if parameters else {})
            }
        })
        return response[0] if response else {}
    
    async def text_generation(
        self,
        model: str,
        prompt: str,
        max_new_tokens: int = 100
    ) -> str:
        """Generate text using a language model"""
        result = await self.inference(
            model=model,
            inputs=prompt,
            parameters={"max_new_tokens": max_new_tokens}
        )
        if isinstance(result, dict):
            return result.get("generated_text", "")
        return ""


# =============================================================================
# WEATHER API INTEGRATION
# =============================================================================

class OpenWeatherConnector(RESTAPIConnector):
    """Connector for OpenWeatherMap API"""
    
    def __init__(self, api_key: str):
        config = RESTAPIConfig(
            name="openweather",
            base_url="https://api.openweathermap.org/data/2.5",
            auth_type="none",
            rate_limit=60
        )
        super().__init__(config)
        self._api_key = api_key
    
    async def get_current_weather(
        self,
        city: str = None,
        lat: float = None,
        lon: float = None,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Get current weather data"""
        params = {"appid": self._api_key, "units": units}
        
        if city:
            params["q"] = city
        elif lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        else:
            raise ValueError("Either city or lat/lon must be provided")
        
        response = await self.fetch({
            "endpoint": "/weather",
            "params": params
        })
        return response[0] if response else {}
    
    async def get_forecast(
        self,
        city: str,
        units: str = "metric",
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """Get weather forecast"""
        response = await self.fetch({
            "endpoint": "/forecast",
            "params": {
                "q": city,
                "appid": self._api_key,
                "units": units,
                "cnt": count
            }
        })
        if response:
            return response[0].get("list", [])
        return []


# =============================================================================
# FINANCIAL DATA INTEGRATION
# =============================================================================

class AlphaVantageConnector(RESTAPIConnector):
    """Connector for Alpha Vantage financial data API"""
    
    def __init__(self, api_key: str):
        config = RESTAPIConfig(
            name="alphavantage",
            base_url="https://www.alphavantage.co",
            auth_type="none",
            rate_limit=5  # Free tier: 5 requests/minute
        )
        super().__init__(config)
        self._api_key = api_key
    
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote"""
        response = await self.fetch({
            "endpoint": "/query",
            "params": {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self._api_key
            }
        })
        if response:
            return response[0].get("Global Quote", {})
        return {}
    
    async def get_time_series(
        self,
        symbol: str,
        interval: str = "daily",
        outputsize: str = "compact"
    ) -> List[Dict[str, Any]]:
        """Get historical stock data"""
        function_map = {
            "intraday": "TIME_SERIES_INTRADAY",
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",
            "monthly": "TIME_SERIES_MONTHLY"
        }
        
        params = {
            "function": function_map.get(interval, "TIME_SERIES_DAILY"),
            "symbol": symbol,
            "apikey": self._api_key,
            "outputsize": outputsize
        }
        
        if interval == "intraday":
            params["interval"] = "5min"
        
        response = await self.fetch({
            "endpoint": "/query",
            "params": params
        })
        
        if response:
            # Parse time series data
            data = response[0]
            series_key = [k for k in data.keys() if "Time Series" in k]
            if series_key:
                time_series = data[series_key[0]]
                return [
                    {"date": date, **values}
                    for date, values in time_series.items()
                ]
        return []


# =============================================================================
# NEWS API INTEGRATION
# =============================================================================

class NewsAPIConnector(RESTAPIConnector):
    """Connector for NewsAPI"""
    
    def __init__(self, api_key: str):
        config = RESTAPIConfig(
            name="newsapi",
            base_url="https://newsapi.org/v2",
            auth_type="api_key",
            auth_config={"header": "X-Api-Key", "key": api_key},
            rate_limit=100
        )
        super().__init__(config)
    
    async def get_headlines(
        self,
        country: str = "us",
        category: str = None,
        query: str = None
    ) -> List[Dict[str, Any]]:
        """Get top headlines"""
        params = {"country": country}
        if category:
            params["category"] = category
        if query:
            params["q"] = query
        
        response = await self.fetch({
            "endpoint": "/top-headlines",
            "params": params
        })
        if response:
            return response[0].get("articles", [])
        return []
    
    async def search_news(
        self,
        query: str,
        from_date: str = None,
        to_date: str = None,
        sort_by: str = "relevancy"
    ) -> List[Dict[str, Any]]:
        """Search for news articles"""
        params = {"q": query, "sortBy": sort_by}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        response = await self.fetch({
            "endpoint": "/everything",
            "params": params
        })
        if response:
            return response[0].get("articles", [])
        return []


# =============================================================================
# SOCIAL MEDIA INTEGRATION (Twitter/X)
# =============================================================================

class TwitterConnector(RESTAPIConnector):
    """Connector for Twitter/X API v2"""
    
    def __init__(self, bearer_token: str):
        config = RESTAPIConfig(
            name="twitter",
            base_url="https://api.twitter.com/2",
            auth_type="bearer",
            auth_config={"token": bearer_token},
            rate_limit=300
        )
        super().__init__(config)
    
    async def search_tweets(
        self,
        query: str,
        max_results: int = 10,
        tweet_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for tweets"""
        params = {
            "query": query,
            "max_results": max_results
        }
        if tweet_fields:
            params["tweet.fields"] = ",".join(tweet_fields)
        
        response = await self.fetch({
            "endpoint": "/tweets/search/recent",
            "params": params
        })
        if response:
            return response[0].get("data", [])
        return []
    
    async def get_user_tweets(
        self,
        user_id: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Get tweets from a user"""
        response = await self.fetch({
            "endpoint": f"/users/{user_id}/tweets",
            "params": {"max_results": max_results}
        })
        if response:
            return response[0].get("data", [])
        return []


# =============================================================================
# GEOCODING INTEGRATION
# =============================================================================

class GeocodingConnector(RESTAPIConnector):
    """Connector for geocoding services"""
    
    def __init__(self, api_key: str, provider: str = "google"):
        if provider == "google":
            config = RESTAPIConfig(
                name="geocoding",
                base_url="https://maps.googleapis.com/maps/api",
                auth_type="none",
                rate_limit=50
            )
        else:
            config = RESTAPIConfig(
                name="geocoding",
                base_url="https://api.opencagedata.com",
                auth_type="none",
                rate_limit=50
            )
        
        super().__init__(config)
        self._api_key = api_key
        self._provider = provider
    
    async def geocode(self, address: str) -> Dict[str, Any]:
        """Convert address to coordinates"""
        if self._provider == "google":
            response = await self.fetch({
                "endpoint": "/geocode/json",
                "params": {"address": address, "key": self._api_key}
            })
            if response and response[0].get("results"):
                location = response[0]["results"][0]["geometry"]["location"]
                return {
                    "lat": location["lat"],
                    "lng": location["lng"],
                    "formatted_address": response[0]["results"][0]["formatted_address"]
                }
        return {}
    
    async def reverse_geocode(self, lat: float, lng: float) -> Dict[str, Any]:
        """Convert coordinates to address"""
        if self._provider == "google":
            response = await self.fetch({
                "endpoint": "/geocode/json",
                "params": {"latlng": f"{lat},{lng}", "key": self._api_key}
            })
            if response and response[0].get("results"):
                return {
                    "formatted_address": response[0]["results"][0]["formatted_address"],
                    "components": response[0]["results"][0]["address_components"]
                }
        return {}


# =============================================================================
# FACTORY AND MANAGER
# =============================================================================

def create_external_connector(
    service: str,
    api_key: str,
    **kwargs
) -> RESTAPIConnector:
    """Factory function to create external service connectors"""
    connectors = {
        "openai": lambda: OpenAIConnector(api_key, kwargs.get("organization")),
        "huggingface": lambda: HuggingFaceConnector(api_key),
        "openweather": lambda: OpenWeatherConnector(api_key),
        "alphavantage": lambda: AlphaVantageConnector(api_key),
        "newsapi": lambda: NewsAPIConnector(api_key),
        "twitter": lambda: TwitterConnector(api_key),
        "geocoding": lambda: GeocodingConnector(api_key, kwargs.get("provider", "google"))
    }
    
    if service not in connectors:
        raise ValueError(f"Unknown service: {service}. Available: {list(connectors.keys())}")
    
    return connectors[service]()


class ExternalServicesManager:
    """Manager for external service integrations"""
    
    def __init__(self):
        self._connectors: Dict[str, RESTAPIConnector] = {}
    
    def add_service(self, service: str, api_key: str, **kwargs) -> None:
        """Add an external service"""
        connector = create_external_connector(service, api_key, **kwargs)
        self._connectors[service] = connector
    
    def get_service(self, service: str) -> Optional[RESTAPIConnector]:
        """Get a service connector"""
        return self._connectors.get(service)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all services"""
        results = {}
        for name, connector in self._connectors.items():
            results[name] = await connector.connect()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all services"""
        for connector in self._connectors.values():
            await connector.disconnect()


__all__ = [
    "OpenAIConnector",
    "HuggingFaceConnector",
    "OpenWeatherConnector",
    "AlphaVantageConnector",
    "NewsAPIConnector",
    "TwitterConnector",
    "GeocodingConnector",
    "create_external_connector",
    "ExternalServicesManager",
]
