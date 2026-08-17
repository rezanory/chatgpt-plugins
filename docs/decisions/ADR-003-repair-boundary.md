# ADR-003: Repair engine is provider-neutral

**Status:** accepted for V0.1

Kaggle reports execution evidence; it does not own AI patch generation. A separate repair engine
classifies the failure and selects retry/repair/wait/stop. Model-specific repair backends can be
added later without changing provider adapters.
