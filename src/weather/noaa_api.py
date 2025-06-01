#!/usr/bin/env python3
"""
NOAA Weather API Integration
Weather data fetching with smart caching and offline fallback
Part of Weather-Driven Viscous Particle Art System
"""

import json
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherObservation:
    """Complete weather data structure for particle system"""
    timestamp: datetime
    temperature: float      # Celsius
    pressure: float        # hPa (millibars)
    humidity: float        # Percentage 0-100
    wind_speed: float      # m/s
    wind_direction: float  # degrees (0-360)
    uv_index: float       # 0-11+
    cloud_cover: float    # Percentage 0-100
    precipitation: float  # mm/hr
    visibility: float     # km
    
    # Computed properties for physics
    @property
    def temperature_kelvin(self) -> float:
        """Temperature in Kelvin for physics calculations"""
        return self.temperature + 273.15
    
    @property
    def pressure_pascals(self) -> float:
        """Pressure in Pascals for physics calculations"""
        return self.pressure * 100.0
    
    @property
    def wind_vector(self) -> np.ndarray:
        """Wind as 3D vector [x, y, z] in m/s"""
        # Convert wind direction to radians
        angle_rad = np.radians(self.wind_direction)
        # Wind is primarily horizontal
        return np.array([
            self.wind_speed * np.cos(angle_rad),
            self.wind_speed * np.sin(angle_rad),
            0.0  # Minimal vertical component
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeatherObservation':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class NOAAWeatherAPI:
    """NOAA Weather API client with caching and offline fallback"""
    
    def __init__(self, cache_dir: str = "weather_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(minutes=5)  # 5-minute cache
        
        # NOAA API endpoints
        self.points_url = "https://api.weather.gov/points/{lat},{lon}"
        self.forecast_url = None
        self.observation_url = None
        
        # Demo weather patterns for offline mode
        self.demo_patterns = self._create_demo_patterns()
        
    def _create_demo_patterns(self) -> Dict[str, WeatherObservation]:
        """Create interesting weather patterns for demos"""
        now = datetime.now()
        
        patterns = {
            'calm': WeatherObservation(
                timestamp=now,
                temperature=22.0,
                pressure=1013.25,
                humidity=50.0,
                wind_speed=2.0,
                wind_direction=180.0,
                uv_index=5.0,
                cloud_cover=20.0,
                precipitation=0.0,
                visibility=10.0
            ),
            'storm': WeatherObservation(
                timestamp=now,
                temperature=10.0,
                pressure=985.0,
                humidity=90.0,
                wind_speed=25.0,
                wind_direction=45.0,
                uv_index=1.0,
                cloud_cover=95.0,
                precipitation=15.0,
                visibility=2.0
            ),
            'heat_wave': WeatherObservation(
                timestamp=now,
                temperature=38.0,
                pressure=1020.0,
                humidity=20.0,
                wind_speed=0.5,
                wind_direction=270.0,
                uv_index=11.0,
                cloud_cover=0.0,
                precipitation=0.0,
                visibility=15.0
            ),
            'fog': WeatherObservation(
                timestamp=now,
                temperature=12.0,
                pressure=1015.0,
                humidity=98.0,
                wind_speed=0.2,
                wind_direction=0.0,
                uv_index=2.0,
                cloud_cover=100.0,
                precipitation=0.1,
                visibility=0.5
            ),
            'hurricane': WeatherObservation(
                timestamp=now,
                temperature=25.0,
                pressure=950.0,
                humidity=95.0,
                wind_speed=45.0,
                wind_direction=120.0,
                uv_index=0.0,
                cloud_cover=100.0,
                precipitation=50.0,
                visibility=1.0
            )
        }
        
        return patterns
    
    def _get_cache_path(self, location: str) -> Path:
        """Get cache file path for location"""
        safe_location = location.replace(',', '_').replace(' ', '_')
        return self.cache_dir / f"weather_{safe_location}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        # Check age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < self.cache_duration
    
    def _save_to_cache(self, location: str, weather: WeatherObservation):
        """Save weather data to cache"""
        cache_path = self._get_cache_path(location)
        with open(cache_path, 'w') as f:
            json.dump(weather.to_dict(), f, indent=2)
        logger.info(f"Saved weather data to cache: {cache_path}")
    
    def _load_from_cache(self, location: str) -> Optional[WeatherObservation]:
        """Load weather data from cache if valid"""
        cache_path = self._get_cache_path(location)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded weather from cache: {cache_path}")
                return WeatherObservation.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        
        return None
    
    def _fetch_from_noaa(self, lat: float, lon: float) -> Optional[WeatherObservation]:
        """Fetch real weather data from NOAA API"""
        try:
            # First, get the grid point for this location
            response = requests.get(
                self.points_url.format(lat=lat, lon=lon),
                timeout=5.0
            )
            response.raise_for_status()
            
            point_data = response.json()
            properties = point_data['properties']
            
            # Get observation station
            stations_url = properties['observationStations']
            response = requests.get(stations_url, timeout=5.0)
            response.raise_for_status()
            
            stations = response.json()
            if not stations['features']:
                logger.error("No observation stations found")
                return None
            
            # Get latest observation from nearest station
            station_id = stations['features'][0]['properties']['stationIdentifier']
            obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
            
            response = requests.get(obs_url, timeout=5.0)
            response.raise_for_status()
            
            obs_data = response.json()
            props = obs_data['properties']
            
            # Parse NOAA data into our format
            weather = WeatherObservation(
                timestamp=datetime.fromisoformat(props['timestamp'].replace('Z', '+00:00')),
                temperature=self._safe_float(props.get('temperature', {}).get('value'), 20.0),
                pressure=self._safe_float(props.get('barometricPressure', {}).get('value'), 1013.25) / 100.0,  # Pa to hPa
                humidity=self._safe_float(props.get('relativeHumidity', {}).get('value'), 50.0),
                wind_speed=self._safe_float(props.get('windSpeed', {}).get('value'), 5.0),
                wind_direction=self._safe_float(props.get('windDirection', {}).get('value'), 180.0),
                uv_index=5.0,  # NOAA doesn't provide UV index in observations
                cloud_cover=self._estimate_cloud_cover(props.get('cloudLayers', [])),
                precipitation=self._safe_float(props.get('precipitationLastHour', {}).get('value'), 0.0),
                visibility=self._safe_float(props.get('visibility', {}).get('value'), 10000.0) / 1000.0  # m to km
            )
            
            logger.info(f"Successfully fetched weather from NOAA")
            return weather
            
        except requests.RequestException as e:
            logger.error(f"NOAA API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}")
            return None
    
    def _safe_float(self, value: any, default: float) -> float:
        """Safely convert value to float with default"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _estimate_cloud_cover(self, cloud_layers: list) -> float:
        """Estimate cloud cover percentage from NOAA cloud layers"""
        if not cloud_layers:
            return 0.0
        
        # NOAA provides cloud cover in oktas (0-8 scale)
        cover_map = {
            'CLR': 0, 'SKC': 0,      # Clear
            'FEW': 12.5,             # Few clouds
            'SCT': 37.5,             # Scattered
            'BKN': 62.5,             # Broken
            'OVC': 100.0,            # Overcast
            'VV': 100.0              # Vertical visibility (obscured)
        }
        
        max_cover = 0.0
        for layer in cloud_layers:
            amount = layer.get('amount', 'CLR')
            max_cover = max(max_cover, cover_map.get(amount, 50.0))
        
        return max_cover
    
    def get_weather(self, location: str = "39.2904,-76.6122") -> WeatherObservation:
        """
        Get weather data with caching and fallback
        
        Args:
            location: Either "lat,lon" string or demo pattern name
            
        Returns:
            WeatherObservation with complete weather data
        """
        start_time = time.time()
        
        # Check if requesting demo pattern
        if location in self.demo_patterns:
            logger.info(f"Using demo pattern: {location}")
            return self.demo_patterns[location]
        
        # Try cache first
        cached = self._load_from_cache(location)
        if cached:
            fetch_time = time.time() - start_time
            logger.info(f"Weather fetch time: {fetch_time:.3f}s (from cache)")
            return cached
        
        # Try live API
        if ',' in location:
            try:
                lat, lon = map(float, location.split(','))
                weather = self._fetch_from_noaa(lat, lon)
                
                if weather:
                    self._save_to_cache(location, weather)
                    fetch_time = time.time() - start_time
                    logger.info(f"Weather fetch time: {fetch_time:.3f}s (from API)")
                    return weather
                    
            except ValueError:
                logger.error(f"Invalid location format: {location}")
        
        # Fallback to demo data
        logger.warning("Using fallback demo weather data")
        fetch_time = time.time() - start_time
        logger.info(f"Weather fetch time: {fetch_time:.3f}s (demo fallback)")
        return self.demo_patterns['calm']
    
    def get_weather_sequence(self, pattern: str = 'storm_approach') -> list[WeatherObservation]:
        """Get a sequence of weather observations for testing dynamics"""
        sequences = {
            'storm_approach': [
                self.demo_patterns['calm'],
                self._interpolate_weather(self.demo_patterns['calm'], self.demo_patterns['storm'], 0.33),
                self._interpolate_weather(self.demo_patterns['calm'], self.demo_patterns['storm'], 0.67),
                self.demo_patterns['storm']
            ],
            'clearing': [
                self.demo_patterns['storm'],
                self._interpolate_weather(self.demo_patterns['storm'], self.demo_patterns['calm'], 0.5),
                self.demo_patterns['calm']
            ],
            'day_cycle': [
                self.demo_patterns['fog'],
                self.demo_patterns['calm'],
                self.demo_patterns['heat_wave'],
                self.demo_patterns['calm']
            ]
        }
        
        return sequences.get(pattern, [self.demo_patterns['calm']])
    
    def _interpolate_weather(self, w1: WeatherObservation, w2: WeatherObservation, t: float) -> WeatherObservation:
        """Linearly interpolate between two weather observations"""
        return WeatherObservation(
            timestamp=datetime.now(),
            temperature=w1.temperature + (w2.temperature - w1.temperature) * t,
            pressure=w1.pressure + (w2.pressure - w1.pressure) * t,
            humidity=w1.humidity + (w2.humidity - w1.humidity) * t,
            wind_speed=w1.wind_speed + (w2.wind_speed - w1.wind_speed) * t,
            wind_direction=w1.wind_direction + (w2.wind_direction - w1.wind_direction) * t,
            uv_index=w1.uv_index + (w2.uv_index - w1.uv_index) * t,
            cloud_cover=w1.cloud_cover + (w2.cloud_cover - w1.cloud_cover) * t,
            precipitation=w1.precipitation + (w2.precipitation - w1.precipitation) * t,
            visibility=w1.visibility + (w2.visibility - w1.visibility) * t
        )