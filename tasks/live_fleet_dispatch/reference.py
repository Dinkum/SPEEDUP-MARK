"""Reference implementation for live_fleet_dispatch; copied into fresh run candidates."""

from __future__ import annotations

import math


_INF = math.inf


def _add_label(labels, energy, time):
    """Retain every nondominated energy/time tradeoff."""
    if any(used <= energy and elapsed <= time for used, elapsed in labels):
        return False
    labels[:] = [(used, elapsed) for used, elapsed in labels
                 if not (energy <= used and time <= elapsed)]
    labels.append((energy, time))
    return True


def _forward_profiles(adjacency, source, budget, targets):
    """Positive energy makes the exact-energy state graph a DAG."""
    size = len(adjacency)
    distance = [[_INF] * size for _ in range(budget + 1)]
    distance[0][source] = 0
    for used in range(budget + 1):
        row = distance[used]
        for vertex, elapsed in enumerate(row):
            if elapsed == _INF:
                continue
            for neighbor, travel_time, energy in adjacency[vertex].values():
                next_energy = used + energy
                if next_energy <= budget:
                    proposal = elapsed + travel_time
                    if proposal < distance[next_energy][neighbor]:
                        distance[next_energy][neighbor] = proposal
    result = {}
    for target in targets:
        labels = []
        best_time = _INF
        for used, row in enumerate(distance):
            elapsed = row[target]
            if elapsed < best_time:
                labels.append((used, elapsed))
                best_time = elapsed
        result[target] = tuple(labels)
    return result


def _reference_route_options(vehicle, jobs, profiles):
    origin, capacity, skills, budget = vehicle
    eligible = [index for index, (_, required, _) in enumerate(jobs)
                if skills & required]
    options = {0: 0}
    states = {}
    for index in eligible:
        mask = 1 << index
        labels = [(energy, elapsed) for energy, elapsed
                  in profiles[origin][jobs[index][0]] if energy <= budget]
        if labels:
            states[mask, index] = labels

    for depth in range(1, min(capacity, len(eligible)) + 1):
        next_states = {}
        for (mask, last), labels in states.items():
            return_labels = profiles[jobs[last][0]][origin]
            for used, elapsed in labels:
                for energy, travel_time in return_labels:
                    if used + energy <= budget:
                        options[mask] = min(options.get(mask, _INF),
                                            elapsed + travel_time)
            if depth == capacity:
                continue
            for index in eligible:
                bit = 1 << index
                if mask & bit:
                    continue
                leg = profiles[jobs[last][0]][jobs[index][0]]
                target = next_states.setdefault((mask | bit, index), [])
                for used, elapsed in labels:
                    for energy, travel_time in leg:
                        if used + energy <= budget:
                            _add_label(target, used + energy,
                                       elapsed + travel_time)
        states = next_states
    return options


def _forward_allocation(options_by_vehicle, jobs):
    penalties = [job[2] for job in jobs]
    saved = [0] * (1 << len(jobs))
    for mask in range(1, len(saved)):
        bit = mask & -mask
        saved[mask] = saved[mask ^ bit] + penalties[bit.bit_length() - 1]
    # The state stores tour time minus penalties of served jobs.
    states = {0: 0}
    for options in options_by_vehicle:
        next_states = states.copy()
        for covered, previous_cost in states.items():
            for tour_mask, tour_time in options.items():
                if tour_mask and not covered & tour_mask:
                    combined = covered | tour_mask
                    proposal = previous_cost + tour_time - saved[tour_mask]
                    if proposal < next_states.get(combined, _INF):
                        next_states[combined] = proposal
        states = next_states
    return saved[-1] + min(states.values())


def _reference_dispatch(adjacency, vehicles, jobs):
    if not jobs:
        return 0
    budget = max((vehicle[3] for vehicle in vehicles), default=0)
    waypoints = {location for location, _, _, _ in vehicles}
    waypoints.update(job[0] for job in jobs)
    profiles = {source: _forward_profiles(adjacency, source, budget, waypoints)
                for source in waypoints}
    options = [_reference_route_options(vehicle, jobs, profiles)
               for vehicle in vehicles]
    return _forward_allocation(options, jobs)


def _visible_vehicles(events, cutoff, identifiers):
    vehicles = []
    for identifier in identifiers:
        current = None
        for event in events.get(identifier, ()):
            if event[0] <= cutoff and (current is None or event[:2] > current[:2]):
                current = event
        if current is not None:
            vehicles.append(current[2:])
    return tuple(vehicles)


def _process(problem):
    answers = []
    for scenario in problem['scenarios']:
        size = scenario['node_count']
        forward = [{} for _ in range(size)]
        reverse = None
        for start, end, travel_time, energy in scenario['edges']:
            forward[start][end] = (end, travel_time, energy)
        rounds = []
        events = {}
        for sequence, operation in enumerate(scenario['operations']):
            kind = operation[0]
            if kind == 'set':
                _, start, end, travel_time, energy = operation
                forward[start][end] = (end, travel_time, energy)
            elif kind == 'delete':
                _, start, end = operation
                forward[start].pop(end, None)
            elif kind == 'telemetry':
                _, identifier, timestamp, location, capacity, skills, battery = operation
                events.setdefault(identifier, []).append((timestamp, sequence, location, capacity, skills, battery))
            else:
                _, cutoff, identifiers, jobs = operation
                vehicles = _visible_vehicles(events, cutoff, identifiers)
                rounds.append(_reference_dispatch(forward, vehicles, jobs))
        answers.append(tuple(rounds))
    return tuple(answers)


def solve(problem):
    return _process(problem)
