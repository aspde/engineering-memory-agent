"""Agent runners — orchestrate agent graph invocations.

Sits between the API layer and the agent: owns the lifecycle of compiled
graphs (checkpointer, concurrency caps, tool selection) and the
entry points that drive them (interactive chat via the API routes,
scheduled patrols, vertical scenarios, event-driven analysis).

Dependency direction is one-way::

    api → runner → agent → service → db/providers

Runner modules may import from ``backend.agent`` and ``backend.service``;
nothing in ``service`` imports back into ``runner``.
"""
