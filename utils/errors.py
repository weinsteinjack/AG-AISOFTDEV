class UtilsError(Exception):
    """Base exception for utils module."""
    pass

class ArtifactError(UtilsError):
    """Generic artifact operation error."""
    pass

class ArtifactSecurityError(ArtifactError):
    """Raised when a path escapes the artifacts directory."""
    pass

class ArtifactNotFoundError(ArtifactError):
    """Raised when an expected artifact is missing."""
    pass
