"""Test activity fixtures with various scenarios."""

ACTIVITIES = {
    "work_hours_run": {
        "object_type": "activity",
        "object_id": 1001,
        "object_data": {
            "id": 1001,
            "name": "Morning Run",
            "type": "Run",
            "distance": 12000,
            "moving_time": 3000,
            "start_date_local": "2025-11-26T10:00:00Z",  # 10:00 UTC = 11:00 CET (work hours)
            "average_heartrate": 142,
        },
        "spotify_recently_played": {"items": []}
    },

    "evening_ride": {
        "object_type": "activity",
        "object_id": 1002,
        "object_data": {
            "id": 1002,
            "name": "Evening Ride",
            "type": "Ride",
            "distance": 53669,
            "moving_time": 5329,
            "start_date_local": "2025-11-26T18:00:00Z",  # 18:00 UTC = 19:00 CET (outside work)
            "average_heartrate": 148,
            "average_watts": 277
        },
        "spotify_recently_played": {
            "items": [
                {
                    "artist": "Saez",
                    "track": "Le Requin",
                    "album": "Apocalypse",
                    "played_at": "2025-11-26T18:05:00Z"
                }
            ]
        }
    },

    "weekend_long_run": {
        "object_type": "activity",
        "object_id": 1003,
        "object_data": {
            "id": 1003,
            "name": "Sunday Long Run",
            "type": "Run",
            "distance": 25000,
            "moving_time": 7200,
            "start_date_local": "2025-11-24T09:00:00Z",  # Sunday morning
            "average_heartrate": 135,
        },
        "spotify_recently_played": {"items": []}
    },

    "boundary_08_29": {
        # Edge case: 1 minute before work hours
        "object_type": "activity",
        "object_id": 1004,
        "object_data": {
            "id": 1004,
            "name": "Early Run",
            "type": "Run",
            "distance": 8000,
            "moving_time": 2400,
            "start_date_local": "2025-11-26T07:29:00Z",  # 07:29 UTC = 08:29 CET (just before work)
            "average_heartrate": 140,
        },
        "spotify_recently_played": {"items": []}
    },

    "boundary_08_30": {
        # Edge case: exactly start of work hours
        "object_type": "activity",
        "object_id": 1005,
        "object_data": {
            "id": 1005,
            "name": "Morning Run",
            "type": "Run",
            "distance": 8000,
            "moving_time": 2400,
            "start_date_local": "2025-11-26T07:30:00Z",  # 07:30 UTC = 08:30 CET exactly (work start)
            "average_heartrate": 140,
        },
        "spotify_recently_played": {"items": []}
    },

    "null_fields_activity": {
        # Edge case: activity with null fields
        "object_type": "activity",
        "object_id": 9999,
        "object_data": {
            "id": 9999,
            "name": None,  # Null name
            "type": "Run",
            "distance": 10000,
            "moving_time": 3000,
            "start_date_local": "2025-11-26T18:00:00Z",
            "average_heartrate": None,  # Null HR
            "average_watts": None       # Null power
        },
        "spotify_recently_played": None  # Null Spotify
    }
}
