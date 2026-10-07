"""Reference implementation for out_of_order_session_windows; copied into fresh run candidates."""

from __future__ import annotations

def _sessionize(problem):
    gap = problem["gap"]
    lateness = problem["allowed_lateness"]
    sessions = {}
    trace = []
    max_seen = None
    for event_id, key, timestamp in problem["events"]:
        max_seen = timestamp if max_seen is None else max(max_seen, timestamp)
        watermark = max_seen - lateness
        if timestamp < watermark:
            trace.append(("drop", event_id, key, timestamp, watermark))
        else:
            current = sessions.setdefault(key, [])
            touching = [
                session
                for session in current
                if timestamp <= session[1] + gap and timestamp >= session[0] - gap
            ]
            for session in touching:
                current.remove(session)
            start = min([timestamp] + [session[0] for session in touching])
            end = max([timestamp] + [session[1] for session in touching])
            count = 1 + sum(session[2] for session in touching)
            current.append([start, end, count])
            current.sort()
            trace.append(
                ("upsert", event_id, key, start, end, count, len(touching), watermark)
            )

        expired = []
        for expired_key, key_sessions in sessions.items():
            for session in key_sessions:
                if session[1] + gap < watermark:
                    expired.append((expired_key, session))
        for expired_key, session in sorted(expired, key=lambda item: (item[0], item[1])):
            sessions[expired_key].remove(session)
            trace.append(("emit", expired_key, session[0], session[1], session[2], watermark))

    for key in sorted(sessions):
        for start, end, count in sorted(sessions[key]):
            trace.append(("emit", key, start, end, count, None))
    return tuple(trace)


def solve(problem):
    return _sessionize(problem)
