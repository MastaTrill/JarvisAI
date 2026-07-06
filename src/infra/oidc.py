"""
OpenID Connect (OIDC) / OAuth2.0 SSO support for JarvisAI
"""
import json
import secrets
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode, parse_qs, urlparse

from authlib.integrations.starlette_client import OAuth, OAuthError
from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6749 import grants
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from src.infra.authentication import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
)
from src.infra.database import get_db
from src.infra.database_models import User
from src.infra.auth_helpers import get_current_user


# OAuth configuration
oauth = OAuth()

# Common OIDC providers configuration
OIDC_PROVIDERS = {
    "google": {
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
        "client_id": None,  # Set via environment variable
        "client_secret": None,  # Set via environment variable
        "scope": ["openid", "email", "profile"],
    },
    "azure_ad": {
        "server_metadata_url": None,  # Format: https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration
        "client_id": None,
        "client_secret": None,
        "scope": ["openid", "email", "profile"],
    },
    "okta": {
        "server_metadata_url": None,  # Format: https://{yourOktaDomain}.com/oauth2/default/.well-known/openid-configuration
        "client_id": None,
        "client_secret": None,
        "scope": ["openid", "email", "profile"],
    },
}


def init_oauth():
    """Initialize OAuth clients for configured providers"""
    for provider_name, config in OIDC_PROVIDERS.items():
        if config["client_id"] and config["client_secret"]:
            oauth.register(
                name=provider_name,
                client_id=config["client_id"],
                client_secret=config["client_secret"],
                server_metadata_url=config["server_metadata_url"],
                client_kwargs={
                    "scope": " ".join(config["scope"]),
                    "code_challenge_method": "S256",
                },
            )


def get_oidc_provider(provider_name: str):
    """Get configured OIDC provider"""
    if not oauth._clients:
        init_oauth()
    
    if provider_name not in oauth._clients:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OIDC provider '{provider_name}' not configured"
        )
    
    return oauth.create_client(provider_name)


async def token_parse_id_token(client, token, request):
    """Parse and validate ID token from OAuth2 token response"""
    try:
        # This is a simplified version - in production you'd want to properly validate the JWT
        # For now, we'll rely on the userinfo endpoint
        return None
    except Exception:
        return None


async def oauth_login(request: Request, provider: str):
    """
    Initiate OIDC login flow
    Redirects user to identity provider for authentication
    """
    client = get_oidc_provider(provider)
    
    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    request.session["oauth_provider"] = provider
    
    # Redirect to identity provider
    redirect_uri = request.url_for("oauth_callback", provider=provider)
    return await client.authorize_redirect(request, redirect_uri, state=state)


async def oauth_callback(request: Request, provider: str, db: Session = Depends(get_db)):
    """
    Handle OIDC callback from identity provider
    Processes the authentication response and creates/login user
    """
    try:
        client = get_oidc_provider(provider)
        
        # Verify state to prevent CSRF
        state = request.query_params.get("state")
        expected_state = request.session.get("oauth_state")
        if not state or state != expected_state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter"
            )
        
        # Clear session state
        request.session.pop("oauth_state", None)
        request.session.pop("oauth_provider", None)
        
        # Get token from provider
        token = await client.authorize_access_token(request)
        
        # Get user info from provider
        user_info = await token_parse_id_token(client, token, request)
        if not user_info:
            # Fallback to userinfo endpoint
            user_info = await client.userinfo(token=token)
        
        # Extract user information
        email = user_info.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by identity provider"
            )
        
        # Get or create user
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Create new user from OIDC info
            username = user_info.get("preferred_username") or email.split("@")[0]
            # Ensure username is unique
            base_username = username
            counter = 1
            while db.query(User).filter(User.username == username).first():
                username = f"{base_username}{counter}"
                counter += 1
            
            user = User(
                email=email,
                username=username,
                full_name=user_info.get("name"),
                is_active=True,
                # OIDC users don't have local passwords by default
                hashed_password=secrets.token_urlsafe(32),  # Placeholder
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Update user info from OIDC (optional)
        user.full_name = user_info.get("name", user.full_name)
        # Update last login time
        from datetime import datetime, timezone
        user.last_login = datetime.now(timezone.utc)
        db.commit()
        
        # Create access token for our system
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role.name if hasattr(user.role, 'name') and user.role else str(user.role)}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role.name if hasattr(user.role, 'name') and user.role else str(user.role)
            }
        }
    except OAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


def get_oidc_router():
    """Get FastAPI router for OIDC endpoints"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/oidc", tags=["authentication"])
    
    @router.get("/login/{provider}")
    async def login(provider: str, request: Request):
        """Initiate OIDC login for specified provider"""
        return await oauth_login(request, provider)
    
    @router.get("/callback/{provider}")
    async def callback(provider: str, request: Request, db: Session = Depends(get_db)):
        """Handle OIDC callback from provider"""
        return await oauth_callback(request, provider, db)
    
    @router.get("/providers")
    async def list_providers():
        """List configured OIDC providers"""
        configured = []
        for name, config in OIDC_PROVIDERS.items():
            if config["client_id"] and config["client_secret"]:
                configured.append(name)
        return {"providers": configured}
    
    return router