"""Unit tests for the SystemStateMachine — transitions, validation, listeners."""

from __future__ import annotations

import threading

import pytest

from core.state.state_machine import StateMachine, SystemState


class TestSystemStateEnum:
    def test_all_states_exist(self):
        expected = {
            "STARTING", "INITIALIZING", "READY", "LISTENING",
            "PROCESSING", "NAVIGATING", "PAUSED", "DEGRADED",
            "ERROR", "SHUTTING_DOWN", "STOPPED",
        }
        actual = {s.name for s in SystemState}
        assert actual == expected

    def test_state_count(self):
        assert len(SystemState) == 11


class TestStateMachineInit:
    def test_default_initial_state(self):
        sm = StateMachine()
        assert sm.state == SystemState.STARTING

    def test_custom_initial_state(self):
        sm = StateMachine(initial_state=SystemState.READY)
        assert sm.state == SystemState.READY

    def test_not_operational_at_start(self):
        sm = StateMachine()
        assert not sm.is_operational

    def test_not_stopped_at_start(self):
        sm = StateMachine()
        assert not sm.is_stopped


class TestStateMachineTransitions:
    def test_valid_transition_starting_to_initializing(self, state_machine):
        assert state_machine.transition(SystemState.INITIALIZING)
        assert state_machine.state == SystemState.INITIALIZING

    def test_valid_full_startup_sequence(self, state_machine):
        assert state_machine.transition(SystemState.INITIALIZING)
        assert state_machine.transition(SystemState.READY)
        assert state_machine.is_operational

    def test_invalid_transition_rejected(self, state_machine):
        """STARTING cannot go directly to READY (must go through INITIALIZING)."""
        assert not state_machine.transition(SystemState.READY)
        assert state_machine.state == SystemState.STARTING

    def test_invalid_transition_to_stopped_from_starting(self, state_machine):
        assert not state_machine.transition(SystemState.STOPPED)

    def test_ready_to_listening(self, ready_state_machine):
        assert ready_state_machine.transition(SystemState.LISTENING)
        assert ready_state_machine.state == SystemState.LISTENING

    def test_ready_to_navigating(self, ready_state_machine):
        assert ready_state_machine.transition(SystemState.NAVIGATING)

    def test_ready_to_paused(self, ready_state_machine):
        assert ready_state_machine.transition(SystemState.PAUSED)
        assert not ready_state_machine.is_operational

    def test_ready_to_shutting_down(self, ready_state_machine):
        assert ready_state_machine.transition(SystemState.SHUTTING_DOWN)
        assert not ready_state_machine.is_operational

    def test_shutting_down_to_stopped(self, ready_state_machine):
        ready_state_machine.transition(SystemState.SHUTTING_DOWN)
        assert ready_state_machine.transition(SystemState.STOPPED)
        assert ready_state_machine.is_stopped

    def test_stopped_is_terminal(self, ready_state_machine):
        ready_state_machine.transition(SystemState.SHUTTING_DOWN)
        ready_state_machine.transition(SystemState.STOPPED)
        assert not ready_state_machine.transition(SystemState.READY)
        assert not ready_state_machine.transition(SystemState.STARTING)

    def test_error_recovery_to_ready(self, ready_state_machine):
        ready_state_machine.transition(SystemState.PROCESSING)
        ready_state_machine.transition(SystemState.ERROR)
        assert ready_state_machine.transition(SystemState.READY)

    def test_error_to_degraded(self, ready_state_machine):
        ready_state_machine.transition(SystemState.PROCESSING)
        ready_state_machine.transition(SystemState.ERROR)
        assert ready_state_machine.transition(SystemState.DEGRADED)

    def test_degraded_recovery(self, ready_state_machine):
        ready_state_machine.transition(SystemState.DEGRADED)
        assert ready_state_machine.transition(SystemState.READY)
        assert ready_state_machine.is_operational


class TestStateMachineListeners:
    def test_listener_called_on_transition(self, state_machine):
        transitions = []
        state_machine.on_transition(
            lambda old, new: transitions.append((old, new))
        )
        state_machine.transition(SystemState.INITIALIZING)
        assert len(transitions) == 1
        assert transitions[0] == (SystemState.STARTING, SystemState.INITIALIZING)

    def test_listener_not_called_on_invalid_transition(self, state_machine):
        transitions = []
        state_machine.on_transition(
            lambda old, new: transitions.append((old, new))
        )
        state_machine.transition(SystemState.STOPPED)  # invalid
        assert len(transitions) == 0

    def test_multiple_listeners(self, state_machine):
        calls_a = []
        calls_b = []
        state_machine.on_transition(lambda o, n: calls_a.append(n))
        state_machine.on_transition(lambda o, n: calls_b.append(n))
        state_machine.transition(SystemState.INITIALIZING)
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_remove_listener(self, state_machine):
        calls = []
        listener = lambda o, n: calls.append(n)
        state_machine.on_transition(listener)
        state_machine.remove_listener(listener)
        state_machine.transition(SystemState.INITIALIZING)
        assert len(calls) == 0

    def test_listener_exception_does_not_block(self, state_machine):
        """A failing listener should not prevent the transition."""
        good_calls = []

        def bad_listener(old, new):
            raise ValueError("boom")

        def good_listener(old, new):
            good_calls.append(new)

        state_machine.on_transition(bad_listener)
        state_machine.on_transition(good_listener)
        assert state_machine.transition(SystemState.INITIALIZING)
        assert len(good_calls) == 1


class TestStateMachineHistory:
    def test_history_recorded(self, state_machine):
        state_machine.transition(SystemState.INITIALIZING)
        state_machine.transition(SystemState.READY)
        history = state_machine.history
        assert len(history) == 2
        assert history[0][0] == "STARTING"
        assert history[0][1] == "INITIALIZING"
        assert history[1][0] == "INITIALIZING"
        assert history[1][1] == "READY"


class TestStateMachineCanTransition:
    def test_can_transition_valid(self, state_machine):
        assert state_machine.can_transition(SystemState.INITIALIZING)

    def test_can_transition_invalid(self, state_machine):
        assert not state_machine.can_transition(SystemState.READY)


class TestStateMachineThreadSafety:
    def test_concurrent_transitions(self, ready_state_machine):
        """Multiple threads attempting transitions should not corrupt state."""
        results = []
        lock = threading.Lock()

        def transition_worker(target):
            result = ready_state_machine.transition(target)
            with lock:
                results.append((target, result))

        threads = [
            threading.Thread(
                target=transition_worker, args=(SystemState.LISTENING,)
            ),
            threading.Thread(
                target=transition_worker, args=(SystemState.NAVIGATING,)
            ),
            threading.Thread(
                target=transition_worker, args=(SystemState.PAUSED,)
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        # Exactly one should succeed (the first to acquire the lock)
        successes = [r for r in results if r[1]]
        assert len(successes) == 1
