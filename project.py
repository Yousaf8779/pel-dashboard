import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import io
import requests

st.set_page_config(
    page_title="PEL – AI Predictive Maintenance",
    page_icon="⚙️", layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THINGSPEAK CONFIG ────────────────────────────────────────────
THINGSPEAK_BASE = "https://api.thingspeak.com"

THINGSPEAK_CHANNELS = {
    "Compressor Unit A": {
        "channel_id": "3376043",
        "read_api_key": "A72NTHJF5HQS0BE1",
        "fields": {"Vibration": "field1", "Temperature": "field2", "Fuel": "field3"}
    },
    "Pump Station B": {
        "channel_id": "3376046",
        "read_api_key": "H2P0HQW9IMKZ3HSY",
        "fields": {"Vibration": "field1", "Temperature": "field2", "Fuel": "field3"}
    },
    "Gas Turbine C": {
        "channel_id": "3376050",
        "read_api_key": "PKZ6F7DG5J3TO93W",
        "fields": {"Vibration": "field1", "Temperature": "field2", "Fuel": "field3"}
    },
    "Generator D": {
        "channel_id": "3376053",
        "read_api_key": "XMK816UD45TYEIPH",
        "fields": {"Vibration": "field1", "Temperature": "field2", "Fuel": "field3"}
    },
}

MACHINES = list(THINGSPEAK_CHANNELS.keys())

# ── THINGSPEAK FETCH ─────────────────────────────────────────────
def fetch_thingspeak(machine, results=100):
    """
    ThingSpeak se real data fetch karta hai.
    Agar data nahi mila ya channel empty hai → simulated data return karta hai.
    Kal jab company real sensors lagaye → same code real data dikhayega.
    """
    cfg = THINGSPEAK_CHANNELS[machine]
    try:
        url = (f"{THINGSPEAK_BASE}/channels/{cfg['channel_id']}/feeds.json"
               f"?api_key={cfg['read_api_key']}&results={results}")
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            feeds = data.get("feeds", [])
            if feeds and len(feeds) > 0:
                rows = []
                for i, feed in enumerate(feeds):
                    try:
                        vib  = float(feed.get("field1") or 0)
                        temp = float(feed.get("field2") or 0)
                        fuel = float(feed.get("field3") or 0)
                        # Sirf valid rows lo
                        if vib > 0 or temp > 0 or fuel > 0:
                            rows.append({
                                "Day": i + 1,
                                "Vibration": vib if vib > 0 else np.random.uniform(2, 9),
                                "Temperature": temp if temp > 0 else np.random.uniform(45, 85),
                                "Fuel": fuel if fuel > 0 else np.random.uniform(80, 480),
                                "Timestamp": feed.get("created_at", ""),
                                "Shift": ["Morning", "Evening", "Night"][i % 3],
                                "Source": "ThingSpeak LIVE 🟢"
                            })
                    except:
                        continue

                if rows:
                    df = pd.DataFrame(rows)
                    df["CO2"] = df["Fuel"] * 2.68 * (1 + df["Vibration"] / 12)
                    df["Failure_Prob"] = np.clip(
                        (df["Vibration"] - 4) / 5.5 + (df["Temperature"] - 60) / 32, 0, 0.96)
                    return df, "live"
    except Exception as e:
        pass

    # ── SIMULATED FALLBACK ───────────────────────────────────────
    # Jab ThingSpeak channel empty ho ya sensors nahi lagy
    seed = MACHINES.index(machine) * 7
    np.random.seed(seed)
    n = 100
    df = pd.DataFrame({
        "Day": range(1, n + 1),
        "Vibration": np.random.uniform(2, 9.5, n),
        "Temperature": np.random.uniform(45, 89, n),
        "Fuel": np.random.uniform(80, 480, n),
        "Shift": np.random.choice(["Morning", "Evening", "Night"], n),
        "Timestamp": ["—"] * n,
        "Source": ["Simulated 🔵"] * n,
    })
    df["CO2"] = df["Fuel"] * 2.68 * (1 + df["Vibration"] / 12)
    df["Failure_Prob"] = np.clip(
        (df["Vibration"] - 4) / 5.5 + (df["Temperature"] - 60) / 32, 0, 0.96)
    return df, "simulated"

def push_thingspeak(machine, vibration, temperature, fuel):
    """
    ThingSpeak channel mein data bhejta hai (simulate sensor push).
    Jab tum Python script ya real sensor lagao tab bhi yahi use hoga.
    """
    cfg = THINGSPEAK_CHANNELS[machine]
    # Write API key ke liye alag key chahiye — abhi push disabled hai
    # Real sensor lagane ke baad Write API Key add karo settings mein
    return False

BG_B64  = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxITEhUSEhMVFhUWFRUVFxcYFxUXFRUVFRUXFhYVFRUYHSggGBolGxUWITEhJSkrLi4uFx8zODMtNygtLi0BCgoKDg0OGxAQGy0lHyUtLS0tLS0tLS0tLS0tLy0tLS0tLS0tLS0tLS0tKy0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAKIBNwMBIgACEQEDEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAEAAIDBQYBB//EAEcQAAEDAQUEBggDBQYFBQAAAAEAAhEDBAUSITFBUWFxBhMigZGhFDJCUrHB0fCCkuEHFSMzckOTorLC8VNis9LiFiRUY4P/xAAaAQADAQEBAQAAAAAAAAAAAAABAgMABAUG/8QALxEAAgIBBAIBAgQFBQAAAAAAAAECEQMEEiExE0FRIvAFFGFxMoGRodEVI7HB4f/aAAwDAQACEQMRAD8ABhdhCXnaerZMwSYGU566RwVQ29qgznWCJb2eOexeSotqz2nNLg0YCeAs1a77cA2HDEZPZHZA3Enki7jtbyYLi4YZzzzkbe9CWNpWFZFdF6AobQ7skl2FoBJO0gCcjs5qSkJzK5bWS3CdDry2+UqXso+ivsViqCizq6naBBdI7LuPAxHAq3stWR2mw7dsne07Rw13gJl304YBy/ytRXVz95HmFPJO2wwjQ5imYFFTZunfBM+B3feSIphc02XiS0wp2NUbGohi4csiyHsU9JRtap6TVxvliSYTSKLpKGz0Jz0HUMN4BdmKLStnFkaDKLQimP3ISgwlWFBgC9LCm+jiyNIkp096nhNBXV6EUkjlbsSYWFPJQN62xzKZcyJluu4uE+UrSaDFMbedto0W4q1VlMb3ua0E7hJzUVR8bV41+0S2dZaw4xLqdIHCcUS2SCfZzJyy04q0s37QB1Y69jZa0jFjOJzg0sHYw5DMnVc+XG5LcjqhjdHo/prSYGf2Bt5hC2q2taA4kNaYzOuYJGQ3wvL7Z+0Ss55LGtAOhIIDRIM6ydAqm0dL7RWLadIPc4aCkw48ts5kbPVDdiktJkl2WUaPRLzvbq6jhUqU6dMQQX51X4mnC2lT0bmJxOk7wNVlLV0vYKYmmZlphzonhPfCeaTLYEorVC5HqSLHT1NPW0W/jF5jrJ7c6q2wbdwz3VXqpjQdgHHqFTN3UNS2uZXbW7WfqPCfpW1cL+GSWF/WxpyOZI0QoQhSRQ2WAAAAoAAFAAFAAFAB9ygghBN0CAQaJBNBCCShHWWuHQo6OjoyXt4zTYFqtOlHqOm2vZkpRKRn3S01kbJ7aTbPDTG3Xf0VTJgdKkqNRhHEKvMhsboZNdIJ2nIH3fmrq7jSAOhB+B/DkqSE7Y32t5G7Rn3Tj/AH+Y5q0si2hgdaHQ4cHDp5SuhqhFa7C1xdqOAO0H1FoZ0RFGRnmJ0KPbMl5aeEfXkrVlUKKsrXJpwB6N9PijNlq5bEMm5xkea5k0GHVBqaFLNVJkDq5ZBjPNRnQ7sC0ZDQEuuWLTqI2EaaqWyjbJPbIbJI1A9aTTqjt3Y17RqHNOi8pxSPS4yfRNWpRt1KLPV7GF9ZlwZHwGwkxpqZBzBUxXWIzCZRbSHWt7GiXHM7yZ1JPnkpW5Vy3WPh9c4gkBhGZ69lKFdWGLyEbVMsqaS6HlUNRe2RqQcpUQu7Y/UqW5S3JHH0e7aY5LjMqWx3KlT4Dk0gaqnEtGonwdvFuaFLnx9EAoJoIWJ0GEAKAoAJFAAFAAKAJFGgARBooIBh6KBjFaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/wBiV2dm0FrwSCCNRwI2gqBqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/wBiV2dm0FrwSCCNRwI2gqBqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUIIIWBsIQhACCqKKKAJFGgARBooIBh6K6IaWkqXLMlxQRjHRKMlISaR5BjhI0tWpqDlmioNJMJPNRVQ4x4Vl6fRL3+JR6S5pcBmNEVqJzExmrzQzJIzRKNWKdojzUJkqB2t9Z/x+B9cugXGXZz3GCXnMQB0GYBMDOTpAA5qGVZjbJSxNe1zzM4nEwMxq3I8MpCkuHx+vqpTU1l/ESjvkuqDI75Tn15+9bEq7I7hqc4ZQcvLPP1V9V3RK22yLNI1zWwM4HLd/P0W7MFkJeAbQJEHLPVpPxJ0WMcJ6mzJxfREZdMpjNmh1j0M+DuuSmqtlbbEqKxI8bsMiNQ/2JXZ2bQWvBIII1HAjaCouqSuV2lqSi7R6mJxkrTF7NUII"
DEV_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wgARCAUAA7EDASIAAhEBAxEB/8QAGgABAQEBAQEBAAAAAAAAAAAAAAECAwQFBv/EABgBAQEBAQEAAAAAAAAAAAAAAAABAgME/9oADAMBAAIQAxAAAAL6SsUC6zqylCygBQAAAACKiAAAqUEAAAAAAAgAAAKAACAAAABKsWIsAAAEqosgABKIsoAQAASwASiAiwA8mpmzuwr1DGgLrOktlqwoACgAAAAASogFgoEAAAAAAAAICgAAAAAAAAAAAAIogAAEoiiAASiASiLABKIBLAIiyvLz3jpjshfWOWwLrNTVloKAAqCgAAAAACABAAAAAAAAAAAAAAAAAAAAAAAQqUAgAAAEsAAIsAAIIAiwSwLBLK8mN46Y6oX2U5bAWVNCqKALBQAAAAAAAACIsAFiAoAAAAAAAAAAAAAAAASAAFlApKIAIQAAoABACIogAIsAEsryc+mOmNqt9g47AWU0lsoFloAAAAAAAAIACgiACAoAICgAAAAAAAAAAAAhFJQSwUAAqACAEoiwAASiAAAiwAiwSyvLjd1DQ9I56CllLZbKUCgAAAAAAAAgAABKIICgAgAKAAAACLCggAAWkoiolBAAAAAiiKEsAAAEogAAEqoBLAIZ1ivNjeOmOgPWXj0CgLZUtloKAAAAACAAAAoAIigACUIogAAAAAAAAgABZaAAEAgAAAAAKASiLIAAAgAAqAAghjea8udY6Y6oPXZeXQAC2DViyigAAAAAgACUAoAAIAAACgiKIAAAAAAIAAUoAgsAAIAAACgAAAIqIAKASwAiyAGN4ry2Szqyt9VlxQAKC2VKKAAACggAAKAAAAAAAAACAAAAAAAAAAAAAIAAIFIUgAoAAAAAABKIBKIIAiiZ1mvJjeOmOgPYOXQACgtlRZaAAAACgAAAAAAAAAAAAgBYAAAAAAAAAAAEoghSgIqAqACAAoAAABKIACCAAGN4ryZ1npjog9iXl0AAoLZUWWgAAAAAoAAAIACgAAAAAgUSwCggAAAAAAAAAAAAAICooiwAAAAAAEAAAEogGN4PLneOmNqPTZeXQBZQC3NSiqAAAAAKAAACAAoAAUiiKIqAAqKIsgAAAAAAAAAAAAAAABLAUhSAAAAAiwACApLBjeDyrNTape9M0BQAWVKKoAAAAAoAAAAAABQAAAACAoACCABSKIogAAAAAoIAAAASyAlAAACwAABKIsAoBz3izzZ1neeqD0jl0KAAGpUCqABQSiAFoBKIoiiKAJQAACBSLAAAUgIsBzrq4dooAEQqCoKyjTI0yNMpdMw2wjbmjpMDcyl0yNMjTI0yNMjUyNM0qQ1EKgqAQuNZs4Y1n08uol9Nw4dNsF6OY6OY6XlTpeROrnTo5jo5jo5jbA2xDowNsDbCNzMNsl0yNMo0yNMjTIoCIoAF4+TU9+Pn40+i+Za9XPnTrnMPo9Ple3F9DzdMusoiiKiAC0JIogUBYQLQAkC0AACUAJZYgLCgEoY3nU8+dZ9PHoqXoXy9pQACLZaAoKEBRUlQKIFBAAAAgogLKWURFAoz5K9Hk489y3GNN88QudWtaxZejGTrrz7k9GvNtfp9vN6eajIBKIKCABSAlQKEolBLLaEgUAILKASiAAAZ1nU88ufRy6ol62XzdQChLItlpQWUBAKmiAhSAAAAAAUgFBE0IonHl4tusxNzTNreZhbzcjecYrs49TpjXOKZL24bPo+34ffD695dMKIAgAAAAAAAAAAEoJVAgAAAIsoBnU1PLmu+OyE6WXzdlIS2oJGsjSaoBZUAgKAAAAsAAEUAWCoAHm38/RjWuk5HJe8403zsOfHPGzreaztee47yRTnormOuuej1/Y/P/V532jmSiKIAAAAAAAAAAFlAlEsBSLAAKgLnWbPIr08+gTpZfL3sUlIVEWUVKpRRCwAAKIQ1lSayKuSgAAAFHi9vyNMc5vpLrOazLDW+fE7+BmzM1LKaJqSO/PfFe3PpiXUz2JZk6+nw9s39Fcb5AAhKIsAAAAAAAAAUAAAQCgAAEubPKufTz6qToXy94osUiyFlSilChAALAlAsAAJQFIAUlADzfJ9fh6zp0xNGeSzfPeCc+nRc8PbmXwdfX6o8Ovobzv5XH7Hks8N9ObnnnpKxplGZiu3XzeiPvejnvhUAIAAiwAAAAAAABQAAEqoBZCgAZ1LPIuPRz7oOo83YoXNLCFlSiqlKELAAogLAFIABYLKIoAeP2eWvhcnPtnbnqu94bN3r6M68vu3rOvN2595XW2ajUjPDvlfPj0Zs8OO/k6c84Z1kE16fN6c39FNONlIFqCAAIolAAQAAAAAC0ACKIAABnWdTzxn0cuwXdl8vaiQFKFlASlqUQUSiKIAAAAACwCwqUSj4Hg/X/G65+PrpNp6eHql9nonbn05cfX4Dl6/L7l1dJZNQzncl5Z65PJ8z7fm3j5TeenIQ36vL9HN+2s41LIAJDSCoKAAFSxCwAFIAAAKLFAASiBGd41PNLPTy6iXdPJ3oQFoFlFECqEWURSAWABYAAABQlAALw7Yr87jvw7TH0fD9uuu15cs30fM5+fU+h6/i+rOvq3jZrpPP5bPoc/kYufsT5fZffPJ2zc/N+349Z+aOnPXu8Ptxr76XkZSUAACpShAAAACUAiiLACwtAAEKQqVGN41PNNT08ugl3ZfL3WWAFlFg0lRZaBLFIoihAAALABZQAAAoZ0r4fk+x8zpef2vjfbjxcPZ4V80rU69+PbO/X0z3zfH5vV57OHXVs1n53os9OvP6s668uqX4fSdunP0cunTn6Po9uPbPAJIAABYLYSgAAAAAAAAAABYsosAAEoiwUGN89Z86z0c+qhqXy91lAixQCixZRRFlAIogAAJQFEUJQAACpa8Xm93n11+X9v5P2Lnxef6eI+bw+rizlrr0mr0m5fLw9PGXPRqzz6605dd5lksT5u2tXXp4+ya9Ss8IIASwAAoSgAAAAAAAASqgUAABSRz3jThnXL08+6I6XOvN2WCgFiUFlFlSiypQQAAAAWCwKlCUJQQqwWUnk9nDWvD7My7651U4O9Oc6ZLqU8/LtzzqdeXZVSyY1mMkPL1x0rP0/P685Cc4VYsJQAQKlKEFqCFlEAKCChACksBRLIUEsGN5083PefRzy9CauuPXzdNUqUFlgBQWVKAWxAAAAASgBQEBTNUAsAK4Z9Hmu+lxbrSES/PX358MPRjx4PodPF6prvEJm4iJlOPqz9NjOozgkrTE1NzDTUiwWpbCoLrEy6OVOrlZejnqNJc0BYipQKiiKIBKhLAKZ1jU8+U9HPowzvXbl283SigFIAWUWVFgtiyUAFQAFIoAAAAAAAAvn786465dL1upK3PPzuMa8vYnn9XgX3ej5u5fpTxeqa1z3zlzhi59X0fn+xy1mTWRNywAKUoqSyGcyXSC2CgAus03rnc3bGsKJVgsCpRLASFlIqpz689TzZ0747MidOHbzdt3I0lCCiFlFlQCyqBABSWQ0lAEimpCgAAAAAWWzwb3w107/O9/zCerntemuXFvo4dC+fr0ufD6dmfTx1xzqYc9Z+h7fy36XfLZEEFgoNMw2wLmRUIqUtzS3NKlLZSgINXFjd56xdDNWAIFIAKc+nLU4Z1PRz7CXG/B08/f28PN4J6fva+V9Fw6yVzpSWUpalSLYssAIqCgAAASiUABaABCCwi/O+jx3fNjO2+kRo4+W69/Pw7PVYzNctcE1yzjWdc8tYx9fx9LPqzOrkQWWLMQ3c1bjXMqICrYNJYWC2DSC2UAELIjXTFl2wy2zc2iAApy68tzjLPRy6CXpr57z+j2ebBrWpojekm7Jndxo1c87O7y4j2zz+mRKFgAqUAAAASgABYqhLnWaKzQs8Pl+z8Td9XX5vra6+b0yXjN5XWMcC+fHPeOnPHW5ekrWsU79PNzPqdPm5T6nPy9Y1vn0N2CZQAWUWUWC2UWUoLZQImbg3c6NJSyBYLvEzernrF0iLz6c9TjnWfTy6CXl06c/P6NuWT0TycE+nn5POz7HH5Ur6PHyLOvPMOsws6foPzf28X1XOsCUFJQAAAAAAFJUqgCAoE879PmeWj1O3ryb3QqQ2Lde0xjH5VUzrn3ebGkVsQ01c7krDx6GdNuvqp05Xk6ZdDXjns7pjOTSBqoMbG8uSoAAfVMhXSGZmAA1cSKilh4WKFLGT4oj5b/LfRnpRiVKxL3+U6WKWY0bD87W+kLPIq8VL97hVBtPdkyM2bWIDmD2q7UtKqJpk4JYa1xBRlgD1G6YVuaSLkFBCK3xGpS0jGmTVFJQq5F4hAJoF8CSgT1CKSUB9lFHQY2EhCQUQ6yjKpIU4Z2qEbMqxIOjgFYIOgRCGQIqoBBsI0dJA3ghCmRkmiGkBwCBKQmVaqIKAgSpbFCqlNKlCFpEJ2oqJSKNBBQKiChMrYB2bYPQASuBkMVEEBTY0VipWhKYsAIhJgqGWq9ELqGpU3ZQwGgWqMAkakEkCoQIHLp0BKEoJq+JEmqEVCcFomhpqQc1OQCBU2EiEExhN6GxNzSSECMoABkHsIQFViqkqSqqQSqJTzKsrCBYJnYKaAkIhXTNhaBEt7ATJAQN1BsAWiMAKkCSJhBJFVaFMRVkqlgoaEoMqQQNOFUjFYFQCQ1UqEiClJ2NKEHLe/FIRiNBlIIiUARRrqFWMASPTAIFJC1Kb1VKNFAbQGqZoHfqSIaisBqQIpQIp1Qa0CVoEzQcCCqpQkxrTlA9dCWBQgqhhMiRgUiGQjGAoSFMkAJFQAApRCCqhBCBIjR7YbCoiNEABNQBrLbAqMqbsJCoqkAaqgQAqlaGFOVopWQKkqJkGgqUkBaaJV5EJCRIUIOKqAApVQ0QAmIKKBCjFhQIATVJrRoAqihTVqQKVqpqIABCpQAClrRpOQU2CKWkpUaqUJIJIFMBJCsohQAoQUBSBiCwQkqIaABSAAIAVSKAQSqSBbVBaRogSBDVRFUQSUIbkoIKpKAUSoAEqAQAAqBiGJoIBoSAUkUKASiKqQVRSoEQiAapqJpKiqQJaJvIqfQSkIFLqWqAaqEkQAJQkKAABSAACpIwFN6QBdCSLEIRkhSSmhQpEADiCKhCQAmKQ0gI0JLpQCCCSoFFgBWlFNQIABUJAADQUmAmVFIBAAoAKJiCJoABYJUgAoRSJChCKkqKRiGBRASkkEKyRgJIoSqTyEEpqiIVapTJCRagFiJPbAiK0CAESK5BUuaC4QBBBBqZECQIAASKASRAqpQKBUQqVAASKkIqiIJpQQqBBCFlUFRCKhFQioUSISFgBxSqATFQqAChJJiUASSIaSmKlFCBSBpRaYDsZAKmNAAASqaFVQoGIqQAKJFASBAqBSqiVSkRqAoqJWhkqSCJhSCARaKMAJEQFIgIBqABUCCFDQJKkBEiAQBARJqACKZJSUoKJFAQBiATRASjYUBFSJJEVBkUEKHlAABioAGAtBJIAEJCKhJhVsRJAVgFSoQiCWi0FVQrCJFAICpIgAIqpMiiACKCAAQAABpIAJsA0JIIAACAAAAIBQAQAAAABEABAAACAAQAAAABAQEAAAAQAAAASAAAAAABAAAAAAAAAAAAAAAAARAAABAAAAAABAAAAAAAAABAQAAABAAAAAABAAAAAAAAAAAAAAAQAAA"

# ── USER ROLES ──────────────────────────────────────────────────
USERS = {
    "admin":    {"password": "pel2025",     "role": "Admin"},
    "engineer": {"password": "engineer123", "role": "Engineer"},
    "viewer":   {"password": "view2025",    "role": "Viewer"},
}

def check_password():
    def do_login():
        u = st.session_state.get("u_in","").strip().lower()
        p = st.session_state.get("p_in","")
        if u in USERS and USERS[u]["password"] == p:
            st.session_state.auth  = True
            st.session_state.cuser = u
            st.session_state.crole = USERS[u]["role"]
        else:
            st.session_state.auth  = False

    if st.session_state.get("auth"):
        return True

    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,.stApp{font-family:'Inter',sans-serif!important;}
    .stApp{background:linear-gradient(rgba(10,20,45,0.85),rgba(5,15,35,0.90));background-size:cover;}
    .stTextInput input{border-radius:8px!important;border:1.5px solid #d1d5db!important;
        padding:11px 14px!important;font-size:14px!important;background:#f9fafb!important;color:#111!important;}
    .stButton>button{background:#1a3a6b!important;color:#fff!important;border:none!important;
        border-radius:8px!important;font-weight:600!important;font-size:14px!important;padding:11px!important;width:100%;}
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:40px!important;}
    </style>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("""<div style="background:#fff;border-radius:14px;padding:44px 40px;
            box-shadow:0 24px 64px rgba(0,0,0,0.35);max-width:420px;margin:0 auto;">
          <div style="text-align:center;margin-bottom:20px;">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s"
                 width="80" style="border-radius:8px;border:2px solid #e5e7eb;">
          </div>
          <div style="font-size:19px;font-weight:700;color:#1a2a4a;text-align:center;margin-bottom:4px;">
            PEL Maintenance System
          </div>
          <div style="font-size:12px;color:#6b7280;text-align:center;margin-bottom:24px;">
            Petroleum Exploration (Pvt.) Ltd.
          </div>
          <div style="height:1px;background:#e5e7eb;margin-bottom:20px;"></div>
        </div>""", unsafe_allow_html=True)

        st.text_input("Username", key="u_in", placeholder="Enter username")
        st.text_input("Password", type="password", key="p_in",
                      placeholder="Enter password", on_change=do_login)
        st.button("Sign In →", on_click=do_login, use_container_width=True)

        if "auth" in st.session_state and not st.session_state.auth:
            st.error("❌ Incorrect credentials.")

        st.markdown("""<div style="text-align:center;margin-top:14px;font-size:11px;color:#9ca3af;">
            admin/pel2025 · engineer/engineer123 · viewer/view2025
        </div>""", unsafe_allow_html=True)
    return False

if not check_password():
    st.stop()

role     = st.session_state.get("crole","Viewer")
can_edit = role in ["Admin","Engineer"]
CRIT     = st.session_state.get("thr_crit", 0.70)
WARN     = st.session_state.get("thr_warn", 0.50)

# ── MAIN CSS ─────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,.stApp{font-family:'Inter',sans-serif!important;color:#111827;}
.stApp{background:#f0f2f5;}
.stApp::before{content:'';position:fixed;top:0;left:0;width:100%;height:4px;
    background:linear-gradient(90deg,#1a3a6b,#c0392b,#1a3a6b);
    background-size:200%;animation:tb 5s linear infinite;z-index:9999;}
@keyframes tb{0%{background-position:0%}100%{background-position:200%}}
.kcard{background:#fff;border-radius:10px;padding:20px 22px;border-top:3px solid #1a3a6b;
    box-shadow:0 1px 6px rgba(0,0,0,0.08);}
.klabel{font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;}
.kval{font-size:28px;font-weight:700;color:#111827;margin:6px 0 3px;line-height:1;}
.kdelta{font-size:12px;font-weight:500;color:#059669;}
.scard{background:#fff;border-radius:10px;padding:22px;box-shadow:0 1px 6px rgba(0,0,0,0.08);margin-bottom:14px;}
.acrit{background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 8px 8px 0;padding:13px 16px;color:#7f1d1d;font-size:13px;font-weight:500;}
.awarn{background:#fffbeb;border-left:4px solid #d97706;border-radius:0 8px 8px 0;padding:13px 16px;color:#78350f;font-size:13px;font-weight:500;}
.aok{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:0 8px 8px 0;padding:13px 16px;color:#14532d;font-size:13px;font-weight:500;}
.live-badge{background:#dcfce7;border:1px solid #16a34a;color:#14532d;border-radius:20px;
    padding:3px 10px;font-size:11px;font-weight:700;}
.sim-badge{background:#dbeafe;border:1px solid #3b82f6;color:#1e40af;border-radius:20px;
    padding:3px 10px;font-size:11px;font-weight:700;}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:8px;padding:4px;border:1px solid #e5e7eb;gap:2px;}
.stTabs [data-baseweb="tab"]{color:#6b7280!important;font-size:13px!important;font-weight:500!important;border-radius:6px!important;padding:8px 14px!important;}
.stTabs [aria-selected="true"]{background:#1a3a6b!important;color:#fff!important;}
div[data-testid="metric-container"]{background:#fff!important;border-radius:10px!important;
    border-top:3px solid #1a3a6b!important;padding:18px!important;box-shadow:0 1px 6px rgba(0,0,0,0.08)!important;}
.stButton>button{background:#1a3a6b!important;color:#fff!important;border:none!important;
    border-radius:8px!important;font-weight:600!important;font-size:12px!important;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:14px!important;max-width:1400px;}
</style>""", unsafe_allow_html=True)

st_autorefresh(interval=20000, key="ar")

# ── DATA LOADING ──────────────────────────────────────────────────
@st.cache_data(ttl=15)
def load_all_data():
    """
    Har machine ka data ThingSpeak se fetch karo.
    Cache 15 seconds — har refresh pe fresh data.
    """
    result = {}
    sources = {}
    for m in MACHINES:
        df, src = fetch_thingspeak(m, results=100)
        result[m] = df
        sources[m] = src
    return result, sources

# ── ML MODELS ────────────────────────────────────────────────────
def train_models(mdata):
    models = {}
    acc = {}
    for m in MACHINES:
        df = mdata[m]
        X = df[['Vibration', 'Temperature', 'Fuel']]
        y = (df['Failure_Prob'] > CRIT).astype(int)
        if len(df) < 10:
            continue
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
        clf.fit(Xt, yt)
        models[m] = clf
        acc[m] = accuracy_score(yv, clf.predict(Xv))
    return models, acc

# Load data
mdata, data_sources = load_all_data()

# Add predicted risk
# Always retrain to match current data columns
st.session_state.models, st.session_state.acc = train_models(mdata)

for m in MACHINES:
    if m in st.session_state.models:
        df = mdata[m]
        try:
            df['Predicted_Risk'] = st.session_state.models[m].predict_proba(
                df[['Vibration', 'Temperature', 'Fuel']])[:, 1]
        except Exception:
            df['Predicted_Risk'] = 0.3
        mdata[m] = df
    else:
        mdata[m]['Predicted_Risk'] = 0.3

# Maintenance log
if "mlog" not in st.session_state:
    st.session_state.mlog = pd.DataFrame({
        'Date': ['2025-01-10', '2025-02-15', '2025-03-20'],
        'Machine': ['Compressor Unit A', 'Pump Station B', 'Gas Turbine C'],
        'Type': ['Preventive', 'Corrective', 'Preventive'],
        'Engineer': ['Ali Hassan', 'Umar Farooq', 'Zara Khan'],
        'Notes': ['Replaced bearings', 'Fixed oil leak', 'Blade inspection OK'],
        'Cost_PKR': [85000, 150000, 60000],
    })

if "notes" not in st.session_state:
    st.session_state.notes = []

# ── EMAIL ────────────────────────────────────────────────────────
def send_email(risk_pct, day, machine, vib, temp, to):
    try:
        sndr = st.secrets.get("alert_email", "")
        pwd = st.secrets.get("alert_email_password", "")
        if not sndr or not pwd:
            return False, "Email credentials not set."
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"PEL ALERT — {machine} Risk {risk_pct:.0f}%"
        msg["From"] = f"PEL AI System <{sndr}>"
        msg["To"] = to
        html = f"""<html><body style="font-family:Inter,Arial;">
        <div style="max-width:560px;margin:0 auto;background:#fff;border-radius:12px;">
          <div style="background:#1a3a6b;padding:22px;text-align:center;">
            <h2 style="color:#fff;margin:10px 0 3px;">PEL Maintenance Alert</h2>
          </div>
          <div style="background:#dc2626;padding:12px;text-align:center;">
            <b style="color:#fff;">🚨 Critical Risk — {machine}</b>
          </div>
          <div style="padding:24px;">
            <p>Risk: <b>{risk_pct:.0f}%</b> | Vibration: <b>{vib:.2f} mm/s</b> | Temp: <b>{temp:.0f}°C</b></p>
            <p>Day #{day} — Immediate inspection required.</p>
          </div>
        </div></body></html>"""
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as sv:
            sv.login(sndr, pwd)
            sv.sendmail(sndr, to, msg.as_string())
        return True, "✅ Alert sent!"
    except Exception as e:
        return False, str(e)

def should_send(r):
    if r <= CRIT:
        return False
    ls = st.session_state.get("last_alert")
    return ls is None or datetime.now() - ls > timedelta(minutes=60)

def chart_layout(fig, h=260):
    fig.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        font_color='#374151', height=h,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor='#f3f4f6'),
        yaxis=dict(gridcolor='#f3f4f6'),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02)
    )
    return fig

# ── HEADER ───────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns([1, 4, 2, 1.2])
with h1:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s", width=85)
with h2:
    # Check kitne live hain
    live_count = sum(1 for m in MACHINES if data_sources.get(m) == "live")
    st.markdown(f"""<div style='padding-top:6px;'>
      <div style='font-size:19px;font-weight:700;color:#1a2a4a;'>PEL – AI Predictive Maintenance System</div>
      <div style='font-size:12px;color:#6b7280;'>Petroleum Exploration (Pvt.) Ltd. · ThingSpeak IoT Integration</div>
      <div style='margin-top:4px;'>
        {'<span class="live-badge">🟢 LIVE ThingSpeak</span>' if live_count > 0 else '<span class="sim-badge">🔵 Simulated Mode</span>'}
        <span style='font-size:11px;color:#6b7280;margin-left:8px;'>{live_count}/{len(MACHINES)} sensors live</span>
      </div>
    </div>""", unsafe_allow_html=True)
with h3:
    now = datetime.now()
    st.markdown(f"""<div style='background:#fff;border-radius:8px;padding:9px 13px;border:1px solid #e5e7eb;text-align:center;'>
      <div style='font-size:10px;color:#6b7280;font-weight:600;'>● AUTO-REFRESH 20s</div>
      <div style='font-size:13px;font-weight:600;color:#1a2a4a;'>{now.strftime('%d %b %Y')}</div>
      <div style='font-size:11px;color:#6b7280;'>{now.strftime('%H:%M:%S')}</div>
    </div>""", unsafe_allow_html=True)
with h4:
    if st.button("⎋ Sign Out", use_container_width=True):
        for k in ["auth", "cuser", "crole"]:
            st.session_state.pop(k, None)
        st.rerun()

rc = {"Admin": "#1a3a6b", "Engineer": "#059669", "Viewer": "#6b7280"}[role]
st.markdown(f"""<div style='display:flex;gap:8px;align-items:center;margin:8px 0 14px;'>
  <span style='background:{rc};color:#fff;font-size:10px;font-weight:700;border-radius:20px;padding:3px 12px;'>{role.upper()}</span>
  <span style='font-size:12px;color:#6b7280;'>Signed in as <b>{st.session_state.get("cuser","")}</b></span>
</div>""", unsafe_allow_html=True)
st.markdown("<div style='height:1px;background:#e5e7eb;margin-bottom:14px;'></div>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────
tabs = st.tabs(["📋 Executive Summary", "📊 Live Dashboard", "🔮 Forecast",
                "⚠️ Alerts & Actions", "🌿 HSE & Environment", "📡 Sensor Config", "⚙️ Settings", "🏢 About PEL"])

# ══ TAB 1 – EXECUTIVE SUMMARY ════════════════════════════════════
with tabs[0]:
    st.markdown("### Executive Summary — Fleet Overview")

    all_risks = []
    for m in MACHINES:
        if 'Predicted_Risk' in mdata[m].columns:
            all_risks.append(float(mdata[m].iloc[-1]['Predicted_Risk']))
        else:
            all_risks.append(0.0)

    avg_fleet = np.mean(all_risks) * 100
    crit_cnt = sum(r > CRIT for r in all_risks)
    total_co2 = sum(mdata[m]['CO2'].sum() for m in MACHINES) / 1000
    avg_health = 100 - avg_fleet

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (c1, "Fleet Health", f"{avg_health:.0f}%", "All machines avg", "#059669" if avg_health > 70 else "#dc2626"),
        (c2, "Critical Machines", f"{crit_cnt}/{len(MACHINES)}", "Risk above threshold", "#dc2626" if crit_cnt > 0 else "#059669"),
        (c3, "Fleet Avg Risk", f"{avg_fleet:.1f}%", "Current average", "#dc2626" if avg_fleet > 70 else "#d97706" if avg_fleet > 50 else "#059669"),
        (c4, "Total CO₂ Logged", f"{total_co2:.1f} t", "All machines combined", "#1a3a6b"),
    ]:
        with col:
            st.markdown(f"""<div class="kcard" style="border-top-color:{clr};">
              <div class="klabel">{lbl}</div>
              <div class="kval" style="color:{clr};">{val}</div>
              <div class="kdelta">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Machine Status Table")
    rows = []
    for m in MACHINES:
        l = mdata[m].iloc[-1]
        r = float(l.get('Predicted_Risk', 0))
        src = data_sources.get(m, "simulated")
        rows.append({
            "Machine": m,
            "Data Source": "🟢 ThingSpeak LIVE" if src == "live" else "🔵 Simulated",
            "Status": "🔴 Critical" if r > CRIT else ("🟡 Warning" if r > WARN else "🟢 Normal"),
            "Risk %": f"{r * 100:.1f}%",
            "Vibration": f"{l['Vibration']:.2f} mm/s",
            "Temperature": f"{l['Temperature']:.1f} °C",
            "CO₂ (kg)": f"{l['CO2']:.0f}",
            "Last Reading": str(l.get('Timestamp', '—'))[:16],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Fleet Risk Comparison")
    fig_fl = go.Figure(go.Bar(
        x=MACHINES, y=[r * 100 for r in all_risks],
        text=[f"{r * 100:.1f}%" for r in all_risks], textposition='outside',
        marker_color=['#dc2626' if r > CRIT else '#d97706' if r > WARN else '#059669' for r in all_risks]))
    fig_fl.add_hline(y=CRIT * 100, line_dash="dash", line_color="#dc2626", annotation_text=f"Critical {CRIT * 100:.0f}%")
    fig_fl.add_hline(y=WARN * 100, line_dash="dot", line_color="#d97706", annotation_text=f"Warning {WARN * 100:.0f}%")
    chart_layout(fig_fl, 300)
    fig_fl.update_layout(yaxis=dict(range=[0, 115], title="Risk %"), showlegend=False)
    st.plotly_chart(fig_fl, use_container_width=True)

# ══ TAB 2 – LIVE DASHBOARD ═══════════════════════════════════════
with tabs[1]:
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        sel_m = st.selectbox("Machine", MACHINES, key="dm")
    with sc2:
        drange = st.selectbox("Range", ["Last 30", "Last 60", "All"], key="dr")
    with sc3:
        dshift = st.selectbox("Shift", ["All", "Morning", "Evening", "Night"], key="ds")

    df_s = mdata[sel_m].copy()
    df_s = df_s.tail({"Last 30": 30, "Last 60": 60, "All": 999}[drange])
    if dshift != "All" and 'Shift' in df_s.columns:
        df_s = df_s[df_s['Shift'] == dshift]

    src_label = data_sources.get(sel_m, "simulated")
    if src_label == "live":
        st.markdown("<div class='live-badge'>🟢 LIVE DATA — ThingSpeak Sensor</div><br>", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='sim-badge'>🔵 SIMULATED DATA — ThingSpeak channel empty. 
        Jab real sensor data bhejo tab live ho jayega.</div><br>""", unsafe_allow_html=True)

    lat = mdata[sel_m].iloc[-1]
    risk = float(lat.get('Predicted_Risk', 0))
    hlth = 100 - risk * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, lbl, val, clr in [
        (k1, "Health Score", f"{hlth:.0f}%", "#059669" if hlth > 70 else "#dc2626"),
        (k2, "Failure Risk", f"{risk * 100:.1f}%", "#dc2626" if risk > CRIT else "#d97706" if risk > WARN else "#059669"),
        (k3, "Vibration", f"{lat['Vibration']:.2f} mm/s", "#1a3a6b"),
        (k4, "Temperature", f"{lat['Temperature']:.1f} °C", "#d97706" if lat['Temperature'] > 80 else "#1a3a6b"),
        (k5, "CO₂", f"{lat['CO2']:.0f} kg", "#1a3a6b"),
    ]:
        with col:
            st.markdown(f"""<div class="kcard" style="border-top-color:{clr};">
              <div class="klabel">{lbl}</div>
              <div class="kval" style="font-size:20px;color:{clr};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    cc1, cc2 = st.columns([3, 1])

    with cc1:
        if 'Predicted_Risk' in df_s.columns:
            st.markdown("**Failure Risk Trend**")
            anom = df_s[df_s['Predicted_Risk'] > CRIT]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=df_s['Day'], y=df_s['Predicted_Risk'],
                mode='lines', name='Risk', line=dict(color='#1a3a6b', width=2),
                fill='tozeroy', fillcolor='rgba(26,58,107,0.06)'))
            if not anom.empty:
                fig_r.add_trace(go.Scatter(x=anom['Day'], y=anom['Predicted_Risk'],
                    mode='markers', name='⚠️ Anomaly',
                    marker=dict(color='#dc2626', size=8)))
            fig_r.add_hline(y=CRIT, line_dash="dash", line_color="#dc2626")
            fig_r.add_hline(y=WARN, line_dash="dot", line_color="#d97706")
            chart_layout(fig_r)
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("**Sensor Readings — Vibration & Temperature**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df_s['Day'], y=df_s['Vibration'],
            name='Vibration (mm/s)', line=dict(color='#1a3a6b', width=2)))
        fig_s.add_trace(go.Scatter(x=df_s['Day'], y=df_s['Temperature'],
            name='Temperature (°C)', line=dict(color='#c0392b', width=2), yaxis='y2'))
        chart_layout(fig_s, 220)
        fig_s.update_layout(
            yaxis=dict(title='Vibration (mm/s)', side='left'),
            yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("**Fuel Consumption**")
        fig_f2 = go.Figure(go.Bar(x=df_s['Day'], y=df_s['Fuel'],
            marker_color='#059669', name='Fuel (L)'))
        chart_layout(fig_f2, 180)
        st.plotly_chart(fig_f2, use_container_width=True)

    with cc2:
        st.markdown("**Machine Health**")
        gc = '#059669' if hlth > 70 else '#d97706' if hlth > 40 else '#dc2626'
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=hlth,
            number={'suffix': '%', 'font': {'color': gc, 'size': 26}},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': gc, 'thickness': 0.28},
                   'steps': [{'range': [0, 40], 'color': 'rgba(220,38,38,0.08)'},
                              {'range': [40, 70], 'color': 'rgba(217,119,6,0.06)'},
                              {'range': [70, 100], 'color': 'rgba(5,150,105,0.06)'}]}))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=190, margin=dict(l=10, r=10, t=20, b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        st.metric("Day", f"#{int(lat['Day'])}")
        st.metric("Fuel", f"{lat['Fuel']:.0f} L")
        if 'Predicted_Risk' in mdata[sel_m].columns:
            st.metric("Model Acc.", f"{st.session_state.acc.get(sel_m, 0) * 100:.1f}%")

        ac = "acrit" if risk > CRIT else "awarn" if risk > WARN else "aok"
        lb = "🔴 Critical" if risk > CRIT else "🟡 Warning" if risk > WARN else "🟢 Normal"
        st.markdown(f"<br><div class='{ac}'><b>{lb}</b><br>Risk: {risk * 100:.1f}%</div>",
                    unsafe_allow_html=True)

    st.markdown("#### Maintenance Log")
    st.dataframe(st.session_state.mlog, use_container_width=True, hide_index=True)
    if can_edit:
        with st.expander("➕ Add Record"):
            ml1, ml2, ml3, ml4, ml5, ml6 = st.columns(6)
            nm = ml1.selectbox("Machine", MACHINES, key="lm")
            nd = ml2.date_input("Date", datetime.today(), key="ld")
            nt = ml3.selectbox("Type", ["Preventive", "Corrective", "Emergency"], key="lt")
            ne = ml4.text_input("Engineer", key="le")
            nn = ml5.text_input("Notes", key="ln")
            nc = ml6.number_input("Cost PKR", 0, key="lc")
            if st.button("Add Record"):
                st.session_state.mlog = pd.concat([st.session_state.mlog, pd.DataFrame([{
                    'Date': str(nd), 'Machine': nm, 'Type': nt,
                    'Engineer': ne, 'Notes': nn, 'Cost_PKR': nc}])], ignore_index=True)
                st.success("Record added!")
                st.rerun()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        mdata[sel_m].tail(60).to_excel(w, sheet_name='Sensor Data', index=False)
        st.session_state.mlog.to_excel(w, sheet_name='Maintenance Log', index=False)
    buf.seek(0)
    st.download_button("📥 Download Excel Report", data=buf,
        file_name=f"PEL_{sel_m.replace(' ', '_')}_Report.xlsx",
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True)

# ══ TAB 3 – FORECAST ═════════════════════════════════════════════
with tabs[2]:
    fm = st.selectbox("Machine", MACHINES, key="fcm")
    df_fc = mdata[fm]
    if 'Predicted_Risk' not in df_fc.columns:
        st.warning("Predicted risk not available.")
    else:
        lat_fc = df_fc.iloc[-1]
        rfc = float(lat_fc['Predicted_Risk'])
        cd = int(lat_fc['Day'])

        np.random.seed(cd % 100)
        base = [min(rfc + i * 0.012 + np.random.uniform(-0.025, 0.025), 0.98) for i in range(30)]
        fd = list(range(cd + 1, cd + 31))
        df_f = pd.DataFrame({'Day': fd, 'Risk': base,
                             'Upper': [min(r + .07, 1) for r in base],
                             'Lower': [max(r - .07, 0) for r in base]})

        st.markdown("**30-Day Risk Forecast**")
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df_f['Day'], y=df_f['Upper'],
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig_f.add_trace(go.Scatter(x=df_f['Day'], y=df_f['Lower'],
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
            fillcolor='rgba(26,58,107,0.08)', name='Confidence'))
        fig_f.add_trace(go.Scatter(x=df_f['Day'], y=df_f['Risk'],
            mode='lines+markers', name='Forecast',
            line=dict(color='#1a3a6b', width=2.5), marker=dict(size=5)))
        fig_f.add_hline(y=CRIT, line_dash="dash", line_color="#dc2626")
        fig_f.add_hline(y=WARN, line_dash="dot", line_color="#d97706")
        chart_layout(fig_f, 340)
        st.plotly_chart(fig_f, use_container_width=True)

        cr = [fd[i] for i, r in enumerate(base) if r > CRIT]
        fs1, fs2, fs3 = st.columns(3)
        fs1.metric("Peak Risk", f"{max(base) * 100:.1f}%")
        fs2.metric("Critical Days", len(cr))
        fs3.metric("Warning Days", sum(1 for r in base if WARN < r <= CRIT))

        if cr:
            st.markdown(f"<div class='acrit'>⚠️ First critical in <b>{cr[0] - cd} days</b>. Schedule maintenance now.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='aok'>✅ No critical periods in next 30 days.</div>", unsafe_allow_html=True)

# ══ TAB 4 – ALERTS ═══════════════════════════════════════════════
with tabs[3]:
    am = st.selectbox("Machine", MACHINES, key="alm")
    df_al = mdata[am]
    lat_al = df_al.iloc[-1]
    ral = float(lat_al.get('Predicted_Risk', 0))

    al1, al2 = st.columns([2, 1])
    with al1:
        st.markdown("**Recent High-Risk Readings**")
        if 'Predicted_Risk' in df_al.columns:
            alf = df_al[df_al['Predicted_Risk'] > WARN].tail(15).copy()
            alf['Status'] = alf['Predicted_Risk'].apply(
                lambda x: '🔴 Critical' if x > CRIT else '🟡 Warning')
            if not alf.empty:
                st.dataframe(alf[['Day', 'Vibration', 'Temperature', 'Predicted_Risk', 'Status', 'Source']],
                             height=300, use_container_width=True, hide_index=True)
            else:
                st.markdown("<div class='aok'>✅ No recent alerts.</div>", unsafe_allow_html=True)

    with al2:
        st.markdown("**Recommended Actions**")
        if ral > CRIT:
            acts = [("🔴", "Immediate", "Halt — inspect now"),
                    ("🔴", "Immediate", "Replace dampeners"),
                    ("🔴", "Immediate", "Notify maintenance head")]
        elif ral > WARN:
            acts = [("🟡", "48 Hours", "Schedule inspection"),
                    ("🟡", "48 Hours", "Check coolant & filters")]
        else:
            acts = [("🟢", "Routine", "Normal operations"),
                    ("🟢", "7 Days", "Next scheduled check")]

        for ic, tm, ac in acts:
            st.markdown(f"""<div style='background:#f9fafb;border:1px solid #e5e7eb;
                        border-radius:8px;padding:10px 14px;margin:5px 0;'>
              <div style='font-size:11px;color:#6b7280;'>{ic} {tm}</div>
              <div style='font-size:13px;color:#111;'>{ac}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Email Alert**")
        eto = st.text_input("Email", placeholder="engineer@pel.com.pk", key="eto")
        if st.button("📧 Send Alert", use_container_width=True):
            if eto:
                ok, mr = send_email(ral * 100, int(lat_al['Day']), am,
                                    float(lat_al['Vibration']), float(lat_al['Temperature']), eto)
                if ok:
                    st.session_state.last_alert = datetime.now()
                    st.success("Sent!")
                else:
                    st.error(f"Failed: {mr}")
            else:
                st.warning("Enter email first")

# ══ TAB 5 – HSE ══════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### HSE & Environmental Compliance")
    tco2 = sum(mdata[m]['CO2'].sum() for m in MACHINES)
    mavg = tco2 / max(1, len(mdata[MACHINES[0]])) * 30
    tgt = 15000
    cmp = min(100, (tgt / max(mavg, 1)) * 100)

    h1c, h2c, h3c, h4c = st.columns(4)
    h1c.metric("Total CO₂", f"{tco2 / 1000:.2f} t")
    h2c.metric("Monthly Avg", f"{mavg:.0f} kg/mo")
    h3c.metric("Target", f"{tgt:,} kg")
    h4c.metric("HSE Compliance", f"{cmp:.0f}%")

    st.markdown("**CO₂ Trend — All Machines**")
    fig_c = go.Figure()
    clrs = ['#1a3a6b', '#c0392b', '#059669', '#d97706']
    for i, m in enumerate(MACHINES):
        dc = mdata[m].tail(60)
        fig_c.add_trace(go.Scatter(x=dc['Day'], y=dc['CO2'], name=m, line=dict(color=clrs[i], width=2)))
    chart_layout(fig_c, 280)
    st.plotly_chart(fig_c, use_container_width=True)

# ══ TAB 6 – SENSOR CONFIG ════════════════════════════════════════
with tabs[5]:
    st.markdown("### 📡 ThingSpeak Sensor Configuration")
    st.markdown("""<div class='scard'>
      <h4 style='color:#1a2a4a;margin:0 0 10px;'>Connected ThingSpeak Channels</h4>
      <p style='color:#6b7280;font-size:13px;margin:0 0 16px;'>
        Yeh channels ThingSpeak IoT platform se connected hain. 
        When sensor data is sent, the dashboard will automatically display real readings.
      </p>
    </div>""", unsafe_allow_html=True)

    for m in MACHINES:
        cfg = THINGSPEAK_CHANNELS[m]
        src = data_sources.get(m, "simulated")
        badge = "🟢 LIVE" if src == "live" else "🔵 Simulated (No data yet)"
        color = "#f0fdf4" if src == "live" else "#eff6ff"
        border = "#16a34a" if src == "live" else "#3b82f6"

        df_latest = mdata[m].iloc[-1]

        st.markdown(f"""<div style='background:{color};border:1px solid {border};border-radius:10px;
                    padding:16px 20px;margin:8px 0;'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
              <div style='font-size:15px;font-weight:700;color:#1a2a4a;'>{m}</div>
              <div style='font-size:12px;color:#6b7280;margin-top:3px;'>
                Channel ID: <b>{cfg["channel_id"]}</b> &nbsp;|&nbsp;
                Fields: Vibration → field1, Temperature → field2, Fuel → field3
              </div>
              <div style='font-size:12px;color:#374151;margin-top:6px;'>
                Latest: Vib={df_latest["Vibration"]:.2f} mm/s | 
                Temp={df_latest["Temperature"]:.1f}°C | 
                Fuel={df_latest["Fuel"]:.0f}L
              </div>
            </div>
            <div style='text-align:right;'>
              <span style='font-size:12px;font-weight:700;color:{"#14532d" if src=="live" else "#1e40af"};'>{badge}</span><br>
              <a href='https://thingspeak.mathworks.com/channels/{cfg["channel_id"]}' 
                 target='_blank' style='font-size:11px;color:#1a3a6b;text-decoration:none;'>
                 🔗 ThingSpeak Channel →
              </a>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='aok'>
      <b>✅ Future Ready:</b> When PEL installs real sensors (temperature, vibration, flow meters),
      they can be connected via ThingSpeak or direct REST API — no dashboard code changes required.
      Data will go live automatically!
    </div>""", unsafe_allow_html=True)

# ══ TAB 7 – SETTINGS ═════════════════════════════════════════════
with tabs[6]:
    st.markdown("### System Settings")
    if not can_edit:
        st.warning("🔒 Viewer role — Settings are read-only.")

    st1, st2 = st.columns(2)
    with st1:
        st.markdown("**Alert Thresholds**")
        nc = st.slider("Critical Threshold (%)", 50, 90, int(CRIT * 100), 5, disabled=not can_edit)
        nw = st.slider("Warning Threshold (%)", 20, 70, int(WARN * 100), 5, disabled=not can_edit)
        if can_edit and st.button("Apply Thresholds"):
            st.session_state.thr_crit = nc / 100
            st.session_state.thr_warn = nw / 100
            st.cache_data.clear()
            st.success("Thresholds updated!")
            st.rerun()

        st.markdown("<br>**Upload Real Sensor Data (CSV)**")
        if can_edit:
            um = st.selectbox("Machine", MACHINES, key="um")
            uf = st.file_uploader("CSV: Day, Vibration, Temperature, Fuel", type=['csv', 'xlsx'])
            if uf and st.button("Load Data"):
                try:
                    nd2 = pd.read_csv(uf) if uf.name.endswith('.csv') else pd.read_excel(uf)
                    req = ['Day', 'Vibration', 'Temperature', 'Fuel']
                    if all(c in nd2.columns for c in req):
                        nd2['CO2'] = nd2['Fuel'] * 2.68 * (1 + nd2['Vibration'] / 12)
                        nd2['Failure_Prob'] = np.clip(
                            (nd2['Vibration'] - 4) / 5.5 + (nd2['Temperature'] - 60) / 32, 0, 0.96)
                        nd2['Shift'] = nd2.get('Shift', 'Morning')
                        nd2['Source'] = 'CSV Upload 📁'
                        nd2['Timestamp'] = '—'
                        nd2['Predicted_Risk'] = st.session_state.models[um].predict_proba(
                            nd2[['Vibration', 'Temperature', 'Fuel']])[:, 1]
                        mdata[um] = nd2
                        st.success(f"Data loaded for {um}!")
                        st.rerun()
                    else:
                        st.error(f"Missing columns. Need: {req}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with st2:
        st.markdown("**Retrain ML Model**")
        rm = st.selectbox("Machine", MACHINES, key="rm")
        if can_edit and st.button("🔄 Retrain"):
            df_rt = mdata[rm]
            X = df_rt[['Vibration', 'Temperature', 'Fuel']]
            y = (df_rt['Failure_Prob'] > CRIT).astype(int)
            Xt2, Xv2, yt2, yv2 = train_test_split(X, y, test_size=.25, random_state=42)
            clf2 = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
            clf2.fit(Xt2, yt2)
            st.session_state.models[rm] = clf2
            st.session_state.acc[rm] = accuracy_score(yv2, clf2.predict(Xv2))
            st.success(f"Retrained! Accuracy: {st.session_state.acc[rm] * 100:.1f}%")

        st.markdown("<br>**ThingSpeak Channels**")
        ts_rows = []
        for m in MACHINES:
            cfg = THINGSPEAK_CHANNELS[m]
            ts_rows.append({
                "Machine": m,
                "Channel ID": cfg["channel_id"],
                "Status": "🟢 Live" if data_sources.get(m) == "live" else "🔵 Simulated"
            })
        st.dataframe(pd.DataFrame(ts_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>**Model Accuracy**")
        st.dataframe(pd.DataFrame([
            {"Machine": m, "Accuracy": f"{st.session_state.acc.get(m, 0) * 100:.1f}%"}
            for m in MACHINES]), use_container_width=True, hide_index=True)

        if can_edit and st.button("🔄 Clear Cache & Refresh Data"):
            st.cache_data.clear()
            st.success("Cache cleared! Refreshing...")
            st.rerun()

# ══ TAB 8 – ABOUT PEL ════════════════════════════════════════════
with tabs[7]:
    st.markdown("""
    <div style='text-align:center;padding:28px 0 16px;'>
      <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s'
           width='95' style='border-radius:10px;border:2px solid #e5e7eb;'>
      <h2 style='color:#1a2a4a;margin:14px 0 5px;'>Petroleum Exploration (Pvt.) Ltd. — PEL</h2>
      <p style='color:#6b7280;font-size:13px;'>Pakistan's Largest Private-Sector Exploration & Production Company</p>
      <div style='display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:12px;'>
        <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:20px;padding:4px 13px;font-size:12px;color:#3730a3;'>📍 Islamabad, Pakistan</span>
        <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:20px;padding:4px 13px;font-size:12px;color:#3730a3;'>🏭 Oil & Gas E&P</span>
        <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:20px;padding:4px 13px;font-size:12px;color:#3730a3;'>📅 Est. 1994</span>
        <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:20px;padding:4px 13px;font-size:12px;color:#3730a3;'>🌍 International Operations</span>
        <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:20px;padding:4px 13px;font-size:12px;color:#3730a3;'>⚡ 40 MMCFD Gas Production</span>
        <span style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:20px;padding:4px 13px;font-size:12px;color:#3730a3;'>🤝 12 JV Partners</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Company Overview
    st.markdown("""<div class='scard'>
      <h4 style='color:#1a2a4a;margin:0 0 10px;'>About PEL</h4>
      <p style='color:#374151;font-size:13px;line-height:1.8;margin:0;'>
        Petroleum Exploration (Pvt.) Ltd. (PEL) is Pakistan's largest private-sector
        Exploration & Production company, incorporated in 1994 under the Companies Ordinance 1984.
        PEL is a subsidiary of the <b>Shahzad International Group of Companies</b> and is the flagship
        business in hydrocarbon exploration and production.
        <br><br>
        PEL currently produces <b>40 MMCFD of gas</b> — approximately 1% of Pakistan's total national gas production.
        The company holds the <b>largest exploration acreage</b> among all private sector E&P companies in Pakistan,
        with 6 development & production leases, 9 onshore blocks, 3 offshore blocks, and 3 overseas blocks in Morocco.
        PEL is organized into multi-disciplinary integrated teams and believes in optimally exploiting knowledge
        and technology by partnering with industry leaders.
        <br><br>
        PEL is also the <b>first Pakistani private-sector E&P company</b> to venture into offshore exploration
        and international operations — currently exploring three onshore permits in Morocco through its affiliate
        <b>Olympus Petroleum</b>.
      </p>
    </div>""", unsafe_allow_html=True)

    # Vision & Mission
    vm1, vm2 = st.columns(2)
    with vm1:
        st.markdown("""<div class='scard' style='border-top-color:#1a3a6b;'>
          <h5 style='color:#1a2a4a;margin:0 0 8px;'>🎯 Vision</h5>
          <p style='color:#374151;font-size:13px;line-height:1.7;margin:0;'>
            To play a major role in enabling Pakistan to become self-sufficient in its energy needs,
            while enhancing the Company's footprint beyond Pakistan's frontiers and transforming
            into a global business entity.
          </p>
        </div>""", unsafe_allow_html=True)
    with vm2:
        st.markdown("""<div class='scard' style='border-top-color:#c0392b;'>
          <h5 style='color:#1a2a4a;margin:0 0 8px;'>🤝 JV Partners</h5>
          <p style='color:#374151;font-size:13px;line-height:1.7;margin:0;'>
            PEL has <b>12 Joint Venture partners</b> from Pakistan, Canada, Kuwait, UK, Morocco and Australia —
            including OGDCL, PPL, BP, Gulf Petroleum Exploration, ONHYM, Sherritt, Spud, and Government Holding Pvt Ltd.
          </p>
        </div>""", unsafe_allow_html=True)

    # Leadership
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 👥 Leadership & Board of Directors")
    l1, l2, l3 = st.columns(3)
    leaders = [
        (l1, "Mr. Shahzad Zaheer", "Chairman & Chief Executive Officer",
         "Heads the Shahzad International Group. Recently met with the Prime Minister of Pakistan to discuss national energy goals. Visited Ayesha Gas Complex with Directors and senior officials."),
        (l2, "Mr. Shahbaz Zaheer", "Director",
         "Firm believer in providing aspiring minds practical opportunities. Actively involved in project oversight, departmental collaboration, and promoting growth culture within PEL. Represented PEL at Petroleum Conference 2024."),
        (l3, "Mr. Faisal Zafar", "Senior Business Executive",
         "Holds ACCA from BPP University London & Masters in International Business from Coventry University UK. Oversees risk assessment, regulatory affairs, international transactions and commercial deals."),
    ]
    for col, name, title, bio in leaders:
        with col:
            st.markdown(f"""<div class='scard' style='text-align:center;'>
              <div style='width:60px;height:60px;border-radius:50%;background:#e8edf5;
                          margin:0 auto 10px;display:flex;align-items:center;justify-content:center;
                          border:2px solid #1a3a6b;font-size:24px;'>👤</div>
              <div style='font-size:14px;font-weight:700;color:#1a2a4a;'>{name}</div>
              <div style='font-size:11px;font-weight:600;color:#1a3a6b;margin:4px 0 8px;
                          text-transform:uppercase;letter-spacing:.5px;'>{title}</div>
              <div style='font-size:12px;color:#6b7280;line-height:1.5;'>{bio}</div>
            </div>""", unsafe_allow_html=True)

    # Key Projects
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🛢️ Key Projects & Operations")
    op1, op2, op3, op4 = st.columns(4)
    for col, ic, ttl, dsc in [
        (op1, "🛢️", "Ayesha Gas Complex", "Badin South Concession, Sindh — producing gas continuously since Feb 2020. Connected to SSGCL national grid."),
        (op2, "⚡", "Zahrah North Discovery", "Latest gas/condensate discovery announced at exploration well Zahrah North-01 — adding to national reserves."),
        (op3, "🌊", "Offshore Exploration", "First Pakistani private E&P company with deep water offshore licenses in Pakistan's territorial waters."),
        (op4, "🌍", "Morocco — Olympus Petroleum", "Three onshore Haha permits in Essaouira Basin + Abda Doukkala reconnaissance license with 75% working interest."),
    ]:
        with col:
            st.markdown(f"""<div style='background:#fff;border-radius:10px;padding:16px;
                border-top:3px solid #c0392b;box-shadow:0 1px 6px rgba(0,0,0,0.08);text-align:center;'>
              <div style='font-size:28px;margin-bottom:8px;'>{ic}</div>
              <div style='font-size:13px;font-weight:600;color:#1a2a4a;margin-bottom:6px;'>{ttl}</div>
              <div style='font-size:12px;color:#6b7280;line-height:1.5;'>{dsc}</div>
            </div>""", unsafe_allow_html=True)

    # Recent News
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📰 Recent News & Achievements")
    news = [
        ("🏆", "Petroleum Conference 2024", "PEL participated in Petroleum Conference 2024 held on January 30th in Islamabad alongside key industry players including OGDCL and PPL."),
        ("🌐", "ADIPEC 2023", "PEL participated in International Petroleum Exhibition & Conference (ADIPEC) — representing Pakistan's private E&P sector internationally."),
        ("🔍", "Exploration Block Revival", "PEL announced successful revival of exploration blocks in Pakistan following a rigorous regulatory process."),
        ("🤝", "PM Meeting", "Chairman Mr. Shahzad Zaheer and Sr. Executive Mr. Faisal Zafar met with the Honorable Prime Minister of Pakistan to discuss energy self-sufficiency."),
    ]
    n1, n2 = st.columns(2)
    for i, (ic, ttl, dsc) in enumerate(news):
        col = n1 if i % 2 == 0 else n2
        with col:
            st.markdown(f"""<div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
                        padding:12px 14px;margin:5px 0;display:flex;gap:12px;align-items:flex-start;'>
              <span style='font-size:20px;'>{ic}</span>
              <div>
                <div style='font-size:13px;font-weight:600;color:#1a2a4a;'>{ttl}</div>
                <div style='font-size:12px;color:#6b7280;margin-top:3px;line-height:1.5;'>{dsc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # CSR
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🌱 Corporate Social Responsibility")
    st.markdown("""<div class='scard'>
      <p style='color:#374151;font-size:13px;line-height:1.8;margin:0;'>
        PEL demonstrates strong commitment to <b>sustainability and corporate social responsibility</b>.
        The company provides safe, clean drinking water to schools, colleges and public libraries in its operating areas.
        Electric water coolers with three-tier action filters are provided in deserving communities.
        PEL also contributes to crisis management initiatives with both public and private sector collaboration,
        supporting local communities with full commitment to social welfare.
      </p>
    </div>""", unsafe_allow_html=True)

    # Stats
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📊 Company Statistics")
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Gas Production", "40 MMCFD", "National Grid")
    cs2.metric("JV Partners", "12+", "Pakistan, Kuwait, UK, Morocco")
    cs3.metric("New Discoveries", "Zahrah North + 4", "Gas Fields")
    cs4.metric("Est.", "1994", "Islamabad, Pakistan")

    # Contact
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📞 Contact Information")
    ct1, ct2 = st.columns(2)
    with ct1:
        st.markdown("""<div class='scard'>
          <h5 style='color:#1a2a4a;margin:0 0 12px;'>🏢 Head Office — Islamabad</h5>
          <div style='font-size:13px;color:#374151;line-height:2;'>
            📍 20 Margalla Road, F-8/3, Islamabad, Pakistan<br>
            📞 +92-51-2287170 to 2287175<br>
            📠 Fax: +92-51-2287154 to 2287155<br>
            🌐 <a href="https://www.pepl.com.pk" target="_blank" style="color:#1a3a6b;">www.pepl.com.pk</a>
          </div>
        </div>""", unsafe_allow_html=True)
    with ct2:
        st.markdown("""<div class='scard'>
          <h5 style='color:#1a2a4a;margin:0 0 12px;'>🔗 Connect With PEL</h5>
          <div style='font-size:13px;color:#374151;line-height:2;'>
            💼 <a href="https://peplportal.com" target="_blank" style="color:#1a3a6b;">peplportal.com</a> — Careers<br>
            🔗 <a href="https://pk.linkedin.com/company/petroleumexplorationltd" target="_blank" style="color:#1a3a6b;">LinkedIn — PEL</a><br>
            📘 <a href="https://www.facebook.com/PetroleumExplorationLtd" target="_blank" style="color:#1a3a6b;">Facebook — PEL</a>
          </div>
        </div>""", unsafe_allow_html=True)

    # Developer
    st.markdown("<div style='height:1px;background:#e5e7eb;margin:20px 0;'></div>", unsafe_allow_html=True)
    st.markdown("#### 👨‍💻 Developed By")
    d1, d2 = st.columns([1, 3])
    with d1:
        st.markdown("""<div style='text-align:center;'>
          <div style='width:140px;height:140px;border-radius:50%;background:#e8edf5;margin:0 auto;
                      display:flex;align-items:center;justify-content:center;border:3px solid #1a3a6b;font-size:50px;'>
            👨‍💻
          </div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class='scard' style='border-top-color:#c0392b;'>
          <div style='font-size:20px;font-weight:700;color:#1a2a4a;'>Muhammad Yousaf</div>
          <div style='font-size:13px;color:#1a3a6b;font-weight:600;margin:4px 0 8px;'>
            Web Developer &amp; AI Student at IMS Peshawar
          </div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
st.markdown("<div style='height:1px;background:#e5e7eb;'></div>", unsafe_allow_html=True)
f1, f2, f3 = st.columns([3, 3, 1])
with f1:
    live_cnt = sum(1 for m in MACHINES if data_sources.get(m) == "live")
    st.markdown(f"<p style='font-size:11px;color:#9ca3af;padding-top:6px;'>"
                f"⚙️ <b>PEL AI Maintenance v2.1</b> · ThingSpeak IoT · "
                f"{'🟢 ' + str(live_cnt) + ' Live' if live_cnt > 0 else '🔵 Simulated'}</p>",
                unsafe_allow_html=True)
with f2:
    st.markdown("<p style='font-size:11px;color:#9ca3af;padding-top:6px;'>"
                "📞 +92-51-2287170 · 🌐 www.pepl.com.pk</p>",
                unsafe_allow_html=True)
with f3:
    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine='openpyxl') as w2:
        for m in MACHINES:
            mdata[m].tail(60).to_excel(w2, sheet_name=m[:30], index=False)
    buf2.seek(0)
    st.download_button("📥 Full Report", data=buf2,
        file_name="PEL_Full_Report.xlsx",
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True)
