"""AI layer: talks to the machine's existing Ollama (never installs it).

A language model handles natural-language commands and chat; a vision model
captions bird photos. Everything here is best-effort — if Ollama is down the
server keeps running and the AI features degrade to a friendly "offline" reply.
"""
