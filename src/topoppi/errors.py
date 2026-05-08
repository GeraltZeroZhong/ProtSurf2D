"""Project-specific exceptions for TopoPPI."""


class TopoPPIError(Exception):
    """Base exception for expected TopoPPI runtime failures."""


class ConfigurationError(TopoPPIError):
    """Raised when user-provided configuration is invalid."""


class InputDataError(TopoPPIError):
    """Raised when input structures or chain selections are invalid."""


class PipelineError(TopoPPIError):
    """Raised when a pipeline stage fails."""


class ExternalToolError(TopoPPIError):
    """Raised when a required external tool is missing or fails."""
