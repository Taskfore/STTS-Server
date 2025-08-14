# STTS Server Library Integration Documentation

## Overview

The STTS Server has been enhanced with a comprehensive library integration system that bridges the existing codebase with the `realtime_conversation` library patterns. This integration provides a clean, modular architecture with advanced monitoring, middleware support, and enterprise-grade patterns.

## Documentation Structure

- **[Architecture Overview](architecture.md)** - System design and component relationships
- **[Adapter Pattern](adapters.md)** - Bridge adapters for legacy engine integration
- **[Middleware System](middleware.md)** - Request processing pipeline and built-in middleware
- **[Router Organization](routers.md)** - Clean router structure and responsibilities
- **[API Reference](api-reference.md)** - Comprehensive endpoint documentation
- **[Configuration](configuration.md)** - Enhanced configuration management
- **[Monitoring & Analytics](monitoring.md)** - Performance tracking and usage analytics
- **[Migration Guide](migration.md)** - Moving from legacy to library-integrated architecture
- **[Development Guide](development.md)** - Adding new features and extending the system

## Quick Start

### Running the Library-Integrated Server

```bash
# Install dependencies (if needed)
pip install -r requirements.txt

# Run the new library-integrated server
python server_v2.py

# Server will be available at http://localhost:8004
# API documentation at http://localhost:8004/docs
```

### Testing the Integration

```bash
# Run integration tests
python test_library_integration.py

# Test specific components
python test_new_routers.py
```

## Key Features

### 🔧 Adapter Pattern
- **Bridge Adapters**: Clean interfaces between legacy engines and library protocols
- **Dependency Injection**: No global state, proper separation of concerns
- **Protocol Compliance**: Implements realtime_conversation library interfaces

### ⚙️ Middleware Pipeline
- **Request Processing**: Complete request lifecycle management
- **Built-in Middleware**: Timing, logging, analytics, and more
- **Extensible**: Easy to add custom middleware components

### 🏗️ Clean Architecture
- **Core Routers**: Business logic with library integration (`routers/core/`)
- **Management Routers**: System management with enhanced features (`routers/management/`)
- **WebSocket Routers**: Organized real-time endpoints (`routers/websocket/`)

### 📊 Enhanced Monitoring
- **Request Context**: Complete request tracking with correlation IDs
- **Performance Metrics**: Timing, resource usage, error rates
- **Usage Analytics**: Voice usage patterns, format preferences, system insights

### 🔒 Backward Compatibility
- **Legacy Endpoints**: All existing endpoints continue to work
- **Zero Downtime**: Gradual migration without service interruption
- **Feature Flags**: Enable/disable new features via configuration

## Architecture Comparison

| Aspect | Legacy (server.py) | Phase 1 (server_clean.py) | Phase 2 (server_v2.py) |
|--------|-------------------|---------------------------|------------------------|
| **Architecture** | Monolithic | Clean Routers | Library Integration |
| **State Management** | Global Variables | Mixed | Dependency Injection |
| **Monitoring** | Basic Logging | Enhanced Logging | Full Lifecycle Tracking |
| **Extensibility** | Hard to Extend | Router-based | Middleware Pipeline |
| **Error Handling** | Inconsistent | Standardized | Context-aware |
| **Performance Insights** | None | Basic | Comprehensive Analytics |

## Getting Started

1. **Read the [Architecture Overview](architecture.md)** to understand the system design
2. **Review the [API Reference](api-reference.md)** for endpoint details
3. **Check the [Migration Guide](migration.md)** for transitioning from legacy
4. **Explore [Middleware System](middleware.md)** for request processing features
5. **See [Development Guide](development.md)** for extending the system

## Support and Contribution

- **Issues**: Report issues and bugs in the project repository
- **Feature Requests**: Suggest new features and improvements
- **Development**: Follow the development guide for contributing

## License

This documentation is part of the STTS Server project and follows the same licensing terms.