"""Unit tests for orthogonal wire router including parallel aligned component wiring (H1)."""

from __future__ import annotations

from circuitsim.wiring.router import count_bends, route, segment_intersects_rect
from tests.test_wiring_router import *  # noqa: F401,F403


def test_h1_same_facing_aligned_pins_route_straight_without_uloop() -> None:
    """Unobstructed aligned terminals with identical outward direction must route as 2-point straight lines."""
    # Vertically aligned West-facing pins (e.g. left pins of stacked horizontal parts)
    west_route = route(
        start=(-40, -120),
        end=(-40, 0),
        start_dir=(-1, 0),
        end_dir=(-1, 0),
        obstacles=[(-40, -140, 40, -100), (-40, -20, 40, 20)],
    )
    assert west_route == ((-40, -120), (-40, 0))
    assert count_bends(west_route) == 0

    # Vertically aligned East-facing pins (e.g. right pins of stacked horizontal parts)
    east_route = route(
        start=(40, -120),
        end=(40, 0),
        start_dir=(1, 0),
        end_dir=(1, 0),
        obstacles=[(-40, -140, 40, -100), (-40, -20, 40, 20)],
    )
    assert east_route == ((40, -120), (40, 0))
    assert count_bends(east_route) == 0

    # Horizontally aligned North-facing and South-facing pins
    north_route = route(
        start=(0, -40),
        end=(120, -40),
        start_dir=(0, -1),
        end_dir=(0, -1),
        obstacles=[(-20, -40, 20, 40), (100, -40, 140, 40)],
    )
    assert north_route == ((0, -40), (120, -40))
    assert count_bends(north_route) == 0

    south_route = route(
        start=(0, 40),
        end=(120, 40),
        start_dir=(0, 1),
        end_dir=(0, 1),
        obstacles=[(-20, -40, 20, 40), (100, -40, 140, 40)],
    )
    assert south_route == ((0, 40), (120, 40))
    assert count_bends(south_route) == 0


def test_h1_same_facing_aligned_pins_detour_when_obstacle_blocks_direct_line() -> None:
    """When an intervening component body blocks the straight line, router detours around it."""
    obstacle = (-60, -80, 20, -40)
    detour = route(
        start=(-40, -120),
        end=(-40, 0),
        start_dir=(-1, 0),
        end_dir=(-1, 0),
        obstacles=[(-40, -140, 40, -100), obstacle, (-40, -20, 40, 20)],
    )
    assert detour[0] == (-40, -120)
    assert detour[-1] == (-40, 0)
    assert len(detour) > 2
    for idx in range(len(detour) - 1):
        assert not segment_intersects_rect(detour[idx], detour[idx + 1], obstacle)
