"""Core business-logic package.

Contains the pure domain modules that decide *how* the application
matches lost items against found items and how it falls back across AI
providers. Code in this package has no I/O dependencies of its own and
no framework awareness — making it the easiest layer to test in isolation.
"""
