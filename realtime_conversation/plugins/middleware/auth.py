"""
Authentication middleware for conversation pipeline.

Validates user authentication and permissions for conversation access.
"""

import logging
from typing import Dict, Any, Set, Optional, Callable, Awaitable
from .base import BaseMiddleware
from ...core.interfaces import ConversationContext

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseMiddleware):
    """Middleware that validates user authentication and permissions."""
    
    def __init__(
        self,
        auth_validator: Optional[Callable[[Dict[str, Any]], Awaitable[bool]]] = None,
        required_permissions: Set[str] = None,
        allow_anonymous: bool = False,
        auth_error_message: str = "Authentication required"
    ):
        """
        Initialize authentication middleware.
        
        Args:
            auth_validator: Async function to validate authentication
            required_permissions: Set of required permissions
            allow_anonymous: Whether to allow anonymous access
            auth_error_message: Error message for authentication failures
        """
        super().__init__(name="authentication")
        self.auth_validator = auth_validator
        self.required_permissions = required_permissions or set()
        self.allow_anonymous = allow_anonymous
        self.auth_error_message = auth_error_message
        
        # Authentication statistics
        self.auth_attempts = 0
        self.auth_successes = 0
        self.auth_failures = 0
        
        logger.info(f"Authentication middleware initialized: anonymous_allowed={allow_anonymous}")
    
    async def _pre_process(self, context: ConversationContext) -> None:
        """Validate authentication before processing."""
        self.auth_attempts += 1
        
        # Get authentication data from user_data
        auth_data = context.user_data or {}
        
        # Check if anonymous access is allowed and no auth data provided
        if self.allow_anonymous and not auth_data.get("auth_token"):
            logger.debug("Anonymous access allowed, skipping authentication")
            return
        
        # Validate authentication
        is_authenticated = await self._validate_authentication(auth_data)
        
        if not is_authenticated:
            self.auth_failures += 1
            raise PermissionError(self.auth_error_message)
        
        # Check permissions if required
        if self.required_permissions:
            await self._check_permissions(auth_data)
        
        self.auth_successes += 1
        logger.debug("Authentication successful")
    
    async def _validate_authentication(self, auth_data: Dict[str, Any]) -> bool:
        """Validate user authentication."""
        try:
            # Use custom validator if provided
            if self.auth_validator:
                return await self.auth_validator(auth_data)
            
            # Default validation: check for auth_token
            auth_token = auth_data.get("auth_token")
            if not auth_token:
                logger.debug("No authentication token provided")
                return False
            
            # Basic token validation (this is a simple example)
            # In production, you would validate against a database or service
            return await self._validate_token(auth_token)
            
        except Exception as e:
            logger.error(f"Error during authentication validation: {e}")
            return False
    
    async def _validate_token(self, token: str) -> bool:
        """Validate authentication token."""
        # This is a placeholder implementation
        # In production, you would:
        # 1. Validate token signature
        # 2. Check token expiration
        # 3. Verify against user database
        # 4. Check if user is active/enabled
        
        if not token or len(token) < 10:
            return False
        
        # Example: reject obviously invalid tokens
        invalid_tokens = {"invalid", "expired", "revoked", "test"}
        if token.lower() in invalid_tokens:
            return False
        
        return True
    
    async def _check_permissions(self, auth_data: Dict[str, Any]) -> None:
        """Check user permissions."""
        user_permissions = set(auth_data.get("permissions", []))
        
        # Check if user has all required permissions
        missing_permissions = self.required_permissions - user_permissions
        
        if missing_permissions:
            logger.warning(
                f"Insufficient permissions. Required: {self.required_permissions}, "
                f"User has: {user_permissions}, Missing: {missing_permissions}"
            )
            raise PermissionError(
                f"Insufficient permissions. Missing: {', '.join(missing_permissions)}"
            )
        
        logger.debug(f"Permission check passed: {user_permissions}")
    
    # Configuration methods
    
    def set_auth_validator(
        self, 
        validator: Callable[[Dict[str, Any]], Awaitable[bool]]
    ) -> None:
        """Set custom authentication validator."""
        self.auth_validator = validator
        logger.info("Custom authentication validator configured")
    
    def set_required_permissions(self, permissions: Set[str]) -> None:
        """Set required permissions."""
        self.required_permissions = permissions
        logger.info(f"Required permissions set: {permissions}")
    
    def add_required_permission(self, permission: str) -> None:
        """Add a required permission."""
        self.required_permissions.add(permission)
        logger.info(f"Added required permission: {permission}")
    
    def remove_required_permission(self, permission: str) -> None:
        """Remove a required permission."""
        self.required_permissions.discard(permission)
        logger.info(f"Removed required permission: {permission}")
    
    def set_allow_anonymous(self, allow: bool) -> None:
        """Set whether to allow anonymous access."""
        self.allow_anonymous = allow
        logger.info(f"Anonymous access {'allowed' if allow else 'denied'}")
    
    # Statistics methods
    
    def get_auth_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        success_rate = (self.auth_successes / self.auth_attempts * 100) if self.auth_attempts > 0 else 0
        
        return {
            "total_attempts": self.auth_attempts,
            "successful_authentications": self.auth_successes,
            "failed_authentications": self.auth_failures,
            "success_rate_percent": round(success_rate, 2),
            "allow_anonymous": self.allow_anonymous,
            "required_permissions": list(self.required_permissions)
        }
    
    def reset_statistics(self) -> None:
        """Reset authentication statistics."""
        self.auth_attempts = 0
        self.auth_successes = 0
        self.auth_failures = 0
        logger.info("Authentication statistics reset")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current authentication configuration."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "allow_anonymous": self.allow_anonymous,
            "required_permissions": list(self.required_permissions),
            "has_custom_validator": self.auth_validator is not None,
            "total_attempts": self.auth_attempts
        }


class TokenAuthenticationMiddleware(AuthenticationMiddleware):
    """Authentication middleware that validates against a set of valid tokens."""
    
    def __init__(
        self,
        valid_tokens: Set[str],
        allow_anonymous: bool = False,
        token_key: str = "auth_token"
    ):
        """
        Initialize token-based authentication middleware.
        
        Args:
            valid_tokens: Set of valid authentication tokens
            allow_anonymous: Whether to allow anonymous access
            token_key: Key to look for token in user_data
        """
        super().__init__(allow_anonymous=allow_anonymous)
        self.valid_tokens = valid_tokens
        self.token_key = token_key
        
        logger.info(f"Token authentication middleware initialized: {len(valid_tokens)} valid tokens")
    
    async def _validate_token(self, token: str) -> bool:
        """Validate token against the valid tokens set."""
        return token in self.valid_tokens
    
    def add_valid_token(self, token: str) -> None:
        """Add a valid token."""
        self.valid_tokens.add(token)
        logger.info("Valid token added")
    
    def remove_valid_token(self, token: str) -> None:
        """Remove a valid token."""
        self.valid_tokens.discard(token)
        logger.info("Valid token removed")
    
    def get_valid_token_count(self) -> int:
        """Get number of valid tokens."""
        return len(self.valid_tokens)


# Factory functions

def create_simple_auth_middleware(
    valid_tokens: Set[str],
    allow_anonymous: bool = False
) -> TokenAuthenticationMiddleware:
    """
    Create a simple token-based authentication middleware.
    
    Args:
        valid_tokens: Set of valid tokens
        allow_anonymous: Whether to allow anonymous access
        
    Returns:
        Configured token authentication middleware
    """
    return TokenAuthenticationMiddleware(
        valid_tokens=valid_tokens,
        allow_anonymous=allow_anonymous
    )


def create_permission_auth_middleware(
    valid_tokens: Set[str],
    required_permissions: Set[str]
) -> TokenAuthenticationMiddleware:
    """
    Create an authentication middleware with permission checking.
    
    Args:
        valid_tokens: Set of valid tokens
        required_permissions: Set of required permissions
        
    Returns:
        Configured authentication middleware
    """
    middleware = TokenAuthenticationMiddleware(valid_tokens=valid_tokens)
    middleware.set_required_permissions(required_permissions)
    return middleware


def create_auth_middleware_from_config(config: Dict[str, Any]) -> AuthenticationMiddleware:
    """
    Create authentication middleware from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured authentication middleware
    """
    valid_tokens = set(config.get("valid_tokens", []))
    
    if valid_tokens:
        middleware = TokenAuthenticationMiddleware(
            valid_tokens=valid_tokens,
            allow_anonymous=config.get("allow_anonymous", False),
            token_key=config.get("token_key", "auth_token")
        )
    else:
        middleware = AuthenticationMiddleware(
            allow_anonymous=config.get("allow_anonymous", True)
        )
    
    # Set required permissions
    required_permissions = set(config.get("required_permissions", []))
    if required_permissions:
        middleware.set_required_permissions(required_permissions)
    
    return middleware